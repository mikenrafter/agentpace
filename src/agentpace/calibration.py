"""Calibration of dispatch estimates against actuals: the evidence for the weekly brief.

Reads the observation store written by `learn`, collapses it to one row per execution, and
reports per model and kind: how far the estimates made at dispatch were from what happened this
window versus earlier windows (mean pinball loss, coverage of P50/P80/P90), which models and
kinds have little history, the largest misses with pointers to open them, and drift flags.

The report states evidence. Flags are thresholded observations for a reader to judge; nothing
here changes an estimator or a rule. Pointers (thread id, transcript path, vigil item, plan task)
are included, prompt text never is.

Invariants:
- Honest absence: a group with no estimate has `None` loss and coverage, never 0.
- A hand-set ceiling (a vigil `--expect-s` with no estimator output behind it) is never a P80: it
  is counted and scored apart (`ceiling`: how often the run finished within it), and is not in
  `n_estimated`, loss or quantile coverage.
- Usage-wait runs are counted by category but not scored: their duration is a provider wait,
  not an estimate error (the owner's labels exclude evidenced usage waits).
- Windows are disjoint: `this` is `now - window < end <= now`; `previous` is `end <= now - window`.

Mating parts: `learn.read_store` and `learn.collapse` produce the rows read here.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentpace import learn
from agentpace import local_runs as lr
from agentpace import paths, render

QUANTILES = (("p50", 0.5), ("p80", 0.8), ("p90", 0.9))
# Thresholds for flags. They are reading aids, not rules: the weekly reader decides what matters.
MIN_HISTORY = 10  # fewer run-level observations than this is "thin"
MIN_FLAG_N = 5  # a flag needs this many scored runs in the window it judges
UNDERCOVERAGE = 0.6  # P80 coverage below this, against a target of 0.8
LOSS_UP_RATIO = 1.5  # this window's scaled loss against the previous windows'
LOSS_UP_FLOOR = 0.05  # and at least this much larger in absolute scaled loss
DEFAULT_TOP = 10


def parse_since(text: str) -> int:
    """`7d`, `12h`, `2w` -> seconds. Anything else is a usage error."""
    match = re.fullmatch(r"(\d+)([hdw])", text.strip())
    if not match or int(match[1]) == 0:
        raise ValueError(f"--since must look like 7d, 12h or 2w: {text!r}")
    return int(match[1]) * {"h": 3600, "d": 86_400, "w": 604_800}[match[2]]


def _scored(row: Mapping[str, Any]) -> bool:
    return row["category"] != "usage-wait"


def _estimated(row: Mapping[str, Any]) -> bool:
    return any(v is not None for v in row["estimate"].values())


def _row_loss(row: Mapping[str, Any]) -> float:
    """Mean pinball over the quantiles this estimate carries."""
    losses = [lr.pinball(row["actual_s"], row["estimate"][name], q) for name, q in QUANTILES if row["estimate"].get(name) is not None]
    return sum(losses) / len(losses)


def _reference_estimate(row: Mapping[str, Any]) -> float:
    """The number a reader compares the actual with: P80, else P50, else P90."""
    for name in ("p80", "p50", "p90"):
        if row["estimate"].get(name) is not None:
            return float(row["estimate"][name])
    raise ValueError("row has no estimate")


def stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Error of estimates against actuals for one population of collapsed rows."""
    scored = [r for r in rows if _scored(r)]
    estimated = [r for r in scored if _estimated(r)]
    coverage: dict[str, dict[str, Any]] = {}
    for name, _ in QUANTILES:
        with_q = [r for r in estimated if r["estimate"].get(name) is not None]
        hits = sum(r["actual_s"] <= r["estimate"][name] for r in with_q)
        coverage[name] = {"hits": hits, "n": len(with_q), "rate": hits / len(with_q) if with_q else None}
    capped = [r for r in scored if r.get("ceiling_s")]
    under = sum(r["actual_s"] <= r["ceiling_s"] for r in capped)
    mean_loss = statistics.fmean(_row_loss(r) for r in estimated) if estimated else None
    mean_actual = statistics.fmean(r["actual_s"] for r in estimated) if estimated else None
    return {
        "n": len(scored),
        "n_estimated": len(estimated),
        "mean_actual_s": mean_actual,
        "mean_pinball_s": mean_loss,
        "relative_loss": mean_loss / mean_actual if estimated and mean_actual else None,
        "coverage": coverage,
        "ceiling": {"hits": under, "n": len(capped), "rate": under / len(capped) if capped else None},
    }


def _flags(this: Mapping[str, Any], previous: Mapping[str, Any]) -> list[str]:
    flags = []
    p80 = this["coverage"]["p80"]
    if p80["n"] >= MIN_FLAG_N and p80["rate"] < UNDERCOVERAGE:
        flags.append("p80-undercoverage")
    now_rel, before_rel = this["relative_loss"], previous["relative_loss"]
    if (
        this["n_estimated"] >= MIN_FLAG_N
        and previous["n_estimated"] >= MIN_FLAG_N
        and now_rel is not None
        and before_rel is not None
        and now_rel >= LOSS_UP_RATIO * before_rel
        and now_rel - before_rel >= LOSS_UP_FLOOR
    ):
        flags.append("loss-up")
    return flags


def _miss(row: Mapping[str, Any]) -> dict[str, Any]:
    reference = _reference_estimate(row)
    return {
        "model": row["model"],
        "kind": row["kind"],
        "category": row["category"],
        "source": row["source"],
        "label_source": row["label_source"],
        "actual_s": row["actual_s"],
        "estimate": dict(row["estimate"]),
        "estimate_source": row.get("estimate_source"),
        "actual_over_estimate": row["actual_s"] / reference,
        "direction": "over-run" if row["actual_s"] > reference else "finished-early",
        "mean_pinball_s": _row_loss(row),
        "end": row["end"],
        "pointers": dict(row.get("pointers") or {}),
    }


def calibrate(
    observations: Sequence[Mapping[str, Any]],
    *,
    now: float,
    window_s: float,
    top: int = DEFAULT_TOP,
) -> dict[str, Any]:
    """The calibration report for runs finished in `(now - window_s, now]` against earlier ones."""
    collapsed = learn.collapse(observations)
    boundary = now - window_s
    window_all = [r for r in collapsed if boundary < r["end"] <= now]
    # Only agent-run samples are grouped, scored and labelled; other roles are counted by category,
    # and subagents nested in a scored parent run (whose actual holds their time) are counted apart.
    rows = [r for r in collapsed if learn.is_sample(r)]
    in_window = [r for r in rows if boundary < r["end"] <= now]
    earlier = [r for r in rows if r["end"] <= boundary]

    def keyed(population: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in population:
            grouped[(row["model"], row["kind"])].append(row)
            grouped[(row["model"], "*")].append(row)
        return grouped

    this_groups, earlier_groups = keyed(in_window), keyed(earlier)
    groups = []
    for model, kind in sorted(this_groups, key=lambda k: (k[0], k[1] != "*", k[1])):
        this, previous = stats(this_groups[(model, kind)]), stats(earlier_groups.get((model, kind), []))
        groups.append({"model": model, "kind": kind, "this": this, "previous": previous, "flags": _flags(this, previous)})

    history: dict[tuple[str, str], list[Mapping[str, Any]]] = keyed(rows)
    new = []
    for model, kind in sorted(this_groups, key=lambda k: (k[0], k[1] != "*", k[1])):
        everything = history[(model, kind)]
        first_seen = min(r["end"] for r in everything)
        runs = sum(r["source"] in learn.RUN_LEVEL_SOURCES for r in everything)
        if first_seen > boundary:
            reason = "first-seen-in-window"
        elif runs < MIN_HISTORY:
            reason = "thin-history"
        else:
            continue
        new.append(
            {
                "model": model,
                "kind": kind,
                "reason": reason,
                "runs": runs,
                "with_estimate": sum(_scored(r) and _estimated(r) for r in everything),
                "first_seen": first_seen,
                "n_this_window": len(this_groups[(model, kind)]),
            }
        )

    candidates = [r for r in in_window if _scored(r) and _estimated(r)]
    misses = sorted((_miss(r) for r in candidates), key=lambda m: (-m["mean_pinball_s"], -m["end"]))[:top]
    return {
        "generated_at": now,
        "window": {"since_s": window_s, "start": boundary, "end": now},
        "observations": {
            "stored": len(observations),
            "executions": len(collapsed),
            "agent_runs": len(rows),
            "nested_subagents": sum(bool(r.get("nested_in")) for r in window_all),
            "vigil_holds": sum(r["source"] == "vigil-ledger" for r in window_all) + sum(r.get("estimate_link") == "vigil-hold" for r in window_all),
            "window": len(in_window),
            "previous": len(earlier),
        },
        "estimates": {
            "scored": sum(_scored(r) for r in in_window),
            "estimated": sum(_scored(r) and _estimated(r) for r in in_window),
            "ceiling": sum(_scored(r) and bool(r.get("ceiling_s")) and not _estimated(r) for r in in_window),
            "linked_by_id": sum(_scored(r) and r.get("link_source") == "meta-run" and r.get("estimate_link") == "vigil-hold" for r in in_window),
            "sources": dict(sorted(Counter(r["estimate_source"] for r in in_window if _scored(r) and r.get("estimate_source")).items())),
        },
        # Each run once: a hold left unscored is not a second copy of the run it watched, and a
        # subagent nested in a delegated run is inside that run's time. Machine jobs and
        # check-ins are holds that are counted, apart, as what they are.
        "categories": {c: sum(r["category"] == c for r in window_all if _counts_as_run(r)) for c in learn.CATEGORIES},
        "labels": {
            "kind_source": dict(Counter(r["kind_source"] for r in in_window)),
            "model_source": dict(Counter(r["model_source"] for r in in_window)),
            "kind_other": sum(r["kind"] == "other" for r in in_window),
            "model_unknown": sum(r["model"] == "unknown" for r in in_window),
        },
        "holds": _hold_counts(window_all),
        "groups": groups,
        "new": new,
        "misses": misses,
        "thresholds": {
            "min_history": MIN_HISTORY,
            "min_flag_n": MIN_FLAG_N,
            "p80_undercoverage_below": UNDERCOVERAGE,
            "loss_up_ratio": LOSS_UP_RATIO,
            "loss_up_floor": LOSS_UP_FLOOR,
        },
    }


def _counts_as_run(row: Mapping[str, Any]) -> bool:
    return learn.is_sample(row) or row["category"] in learn.ROLE_CATEGORIES


def _hold_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Vigil holds this window by fate: `scored` moved its estimate onto the one run it watched;
    every other status left the hold unscored (see `learn.collapse`)."""
    counts = Counter(str(r.get("link_status") or "unknown") for r in rows if r["source"] == "vigil-ledger")
    counts["scored"] = sum(r.get("estimate_link") == "vigil-hold" and learn.is_sample(r) for r in rows)
    return dict(sorted(counts.items()))


# --- rendering -----------------------------------------------------------------------------


def _day(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%d")


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f}"


def _coverage(entry: Mapping[str, Any]) -> str:
    return "-" if entry["n"] == 0 else f"{entry['hits']}/{entry['n']}"


def _pointer_text(pointers: Mapping[str, Any]) -> str:
    return "; ".join(f"{k}={v}" for k, v in pointers.items()) or "none"


def render_markdown(report: Mapping[str, Any]) -> str:
    window, counts = report["window"], report["observations"]
    lines = [
        "# agentpace calibration",
        "",
        f"Window: runs finished {_day(window['start'])} to {_day(window['end'])} UTC "
        f"({window['since_s'] / 86_400:g} days), against everything earlier. "
        f"{counts['executions']} executions in the store, {counts.get('agent_runs', counts['executions'])} of them agent runs "
        f"({counts['window']} this window, {counts['previous']} earlier).",
        "",
        "This is evidence, not a verdict. Pointers lead to the runs; the reader decides what to queue. "
        "Private: holds thread ids, transcript paths and vigil labels.",
        "",
    ]
    if counts["executions"] == 0:
        return "\n".join(lines + ["No observations yet. Run `agentpace learn` first."]) + "\n"

    cats = report["categories"]
    lines += [
        "## Categories this window",
        "",
        " | ".join(f"{name} {cats[name]}" for name in learn.CATEGORIES),
        "",
        "Usage-wait runs are counted here and left out of every error figure below; so are orchestrator turns "
        "(main-thread T3 turns, which include owner waits), machine jobs and check-ins, which are not agent runs. "
        f"{report['estimates']['estimated']} of {report['estimates']['scored']} scored runs this window had a dispatch estimate"
        f" and {report['estimates'].get('ceiling', 0)} more only a hand-set ceiling.",
        "",
        "## Calibration by model and kind",
        "",
        "Coverage is hits/n, the share of actuals at or under the estimate (target 50/80/90 for P50/P80/P90). "
        "Loss is mean pinball in seconds over the quantiles the estimate carried; `-` means no estimate existed.",
        "",
        "| model | kind | n (estimated) | earlier n (est.) | loss s | earlier loss s | P50 | P80 | P90 | flags |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    shown = [g for g in report["groups"] if g["this"]["n_estimated"] or g["previous"]["n_estimated"]]
    for g in shown:
        this, before = g["this"], g["previous"]
        cov = this["coverage"]
        lines.append(
            f"| {g['model']} | {g['kind']} | {this['n']} ({this['n_estimated']}) | {before['n']} ({before['n_estimated']}) "
            f"| {_seconds(this['mean_pinball_s'])} | {_seconds(before['mean_pinball_s'])} "
            f"| {_coverage(cov['p50'])} | {_coverage(cov['p80'])} | {_coverage(cov['p90'])} | {', '.join(g['flags']) or ''} |"
        )
    omitted = len(report["groups"]) - len(shown)
    if omitted:
        lines += ["", f"{render.count(omitted, 'group')} with runs but no dispatch estimate {'is' if omitted == 1 else 'are'} omitted here; the JSON lists them."]
    t = report["thresholds"]
    holds = report.get("holds") or {}
    if holds:
        unscored = ", ".join(f"{k} {v}" for k, v in holds.items() if k != "scored") or "none"
        lines += [
            "",
            "## Vigil holds",
            "",
            f"{holds.get('scored', 0)} holds watched exactly one run and their estimate (or hand-set ceiling) is scored against "
            f"that run's actual; {report['estimates'].get('linked_by_id', 0)} of the runs were found by the id the orchestrator recorded. Not scored: {unscored}. A hold's own open-to-done time is the orchestrator's wait, so it is never "
            "an actual; `multi-run` holds covered several runs, `not-an-agent-run` are check-ins and machine jobs, and the "
            "other statuses could not be tied to one run.",
        ]
    capped = [g for g in report["groups"] if g["this"].get("ceiling", {}).get("n") or g["previous"].get("ceiling", {}).get("n")]
    lines += [
        "",
        "## Hand-set ceilings",
        "",
        "A ceiling is a `--expect-s` an orchestrator typed with no estimate behind it. It is a budget, not a P80, so it is "
        "left out of every coverage and loss figure above; this table only says how often runs finished within it. A ceiling that "
        "nearly every run meets is a loose alarm, not evidence of a good estimate.",
        "",
    ]
    if capped:
        lines += ["| model | kind | within ceiling (this) | within ceiling (earlier) |", "|---|---|---|---|"]
        for g in capped:
            lines.append(f"| {g['model']} | {g['kind']} | {_coverage(g['this']['ceiling'])} | {_coverage(g['previous']['ceiling'])} |")
    else:
        lines.append("None this window or before.")
    lines += ["", "## Drift flags", ""]
    flagged = [g for g in report["groups"] if g["flags"]]
    if flagged:
        for g in flagged:
            this, before = g["this"], g["previous"]
            for flag in g["flags"]:
                if flag == "p80-undercoverage":
                    p80 = this["coverage"]["p80"]
                    detail = f"P80 covered {p80['hits']}/{p80['n']} actuals (target 0.8, flag below {t['p80_undercoverage_below']})"
                else:
                    detail = f"scaled loss {this['relative_loss']:.2f} vs {before['relative_loss']:.2f} earlier (flag at x{t['loss_up_ratio']} and +{t['loss_up_floor']})"
                lines.append(f"- {g['model']} / {g['kind']}: **{flag}**, {detail}")
    else:
        lines.append(f"None. A flag needs at least {t['min_flag_n']} scored runs in the window it judges.")
    lines += ["", "## New or thin history", ""]
    if report["new"]:
        lines += ["| model | kind | reason | run-level history | with estimate | first seen | runs this window |", "|---|---|---|---|---|---|---|"]
        for n in report["new"]:
            lines.append(f"| {n['model']} | {n['kind']} | {n['reason']} | {n['runs']} | {n['with_estimate']} | {_day(n['first_seen'])} | {n['n_this_window']} |")
        lines += ["", f"History counts runs from T3 and transcripts (thin below {t['min_history']}); hold-ledger copies of the same runs are not added."]
    else:
        lines.append("None.")
    lines += ["", f"## Largest misses ({len(report['misses'])})", ""]
    if report["misses"]:
        lines += ["| model | kind | category | direction | actual s | estimate s | actual / estimate | loss s | pointers |", "|---|---|---|---|---|---|---|---|---|"]
        for m in report["misses"]:
            estimate = ", ".join(f"{q} {_seconds(v)}" for q, v in m["estimate"].items() if v is not None)
            lines.append(
                f"| {m['model']} | {m['kind']} | {m['category']} | {m['direction']} | {_seconds(m['actual_s'])} | {estimate} "
                f"| {m['actual_over_estimate']:.2f} | {_seconds(m['mean_pinball_s'])} | {_pointer_text(m['pointers'])} |"
            )
    else:
        lines.append("No scored run had a dispatch estimate this window.")
    labels = report["labels"]
    lines += [
        "",
        "## How runs were labelled",
        "",
        f"Kind source: {', '.join(f'{k} {v}' for k, v in sorted(labels['kind_source'].items())) or 'none'}. "
        f"Model source: {', '.join(f'{k} {v}' for k, v in sorted(labels['model_source'].items())) or 'none'}. "
        f"{labels['kind_other']} runs have kind `other` and {labels['model_unknown']} have model `unknown`: "
        "the keyword labelling did not recognise them.",
    ]
    return "\n".join(lines) + "\n"


# --- CLI -----------------------------------------------------------------------------------


def _parse_at(text: str | None) -> float:
    """`--at`: an epoch or ISO time; anything else, or a non-finite number, is a usage error."""
    if text is None:
        return time.time()
    try:
        value = float(text)
    except ValueError:
        try:
            value = lr.epoch(text)
        except (ValueError, TypeError, AttributeError):
            raise ValueError(f"--at must be an epoch or ISO time: {text!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"--at must be a finite time: {text!r}")
    return value


def _write_private(target: Path, text: str) -> None:
    """Write the report owner-only (0600), also over an existing file: it holds thread ids,
    transcript paths and vigil labels."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)


def cmd_calibration(args: Any) -> int:
    window_s = parse_since(args.since)
    now = _parse_at(args.at)
    if args.top < 1:
        raise ValueError(f"--top must be at least 1: {args.top}")
    state = Path(args.state) if args.state else paths.state_dir()
    report = calibrate(learn.read_store(state), now=now, window_s=window_s, top=args.top)
    text = json.dumps(report, indent=2) + "\n" if args.json else render_markdown(report)
    if args.out:
        target = Path(args.out)
        _write_private(target, text)
        print(f"wrote {target}")
    else:
        print(text, end="")
    return 0


def add_command(commands: Any) -> None:
    parser = commands.add_parser("calibration", help="estimates vs actuals per model and kind, for the weekly brief")
    parser.add_argument("--since", default="7d", help="window of runs to judge, e.g. 7d, 12h, 2w (default: %(default)s)")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="print JSON")
    output.add_argument("--md", action="store_true", help="print Markdown (default)")
    parser.add_argument("--out", help="write the report to this file instead of printing it")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help="how many largest misses to list (default: %(default)s)")
    parser.add_argument("--state", help="state directory (default: AGENTPACE_STATE or the XDG state dir)")
    parser.add_argument("--at", help="treat this epoch or ISO time as now")
    parser.set_defaults(handler=cmd_calibration)
