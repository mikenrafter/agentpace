"""`agentpace calibration --html`: the summary a person reads first, as cards.

The embedded snapshot is the report `calibrate` returns, the same object `--json` prints. Like the
Markdown report it holds pointers (thread ids, transcript paths, vigil labels), so the caller
writes it owner-only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agentpace import calibration as cal
from agentpace import render as fmt
from agentpace.state_html import esc, page

ALL_KINDS = "*"


def _percent(hits: int, n: int) -> str:
    return "-" if n == 0 else f"{hits / n:.0%}"


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header = "".join(f"<th>{esc(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def _overall_coverage(report: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """Hits and n per quantile, summed over each model's all-kinds row (each run counted once)."""
    totals = {name: (0, 0) for name, _ in cal.QUANTILES}
    for group in report["groups"]:
        if group["kind"] != ALL_KINDS:
            continue
        for name, _ in cal.QUANTILES:
            hits, n = totals[name]
            entry = group["this"]["coverage"][name]
            totals[name] = (hits + entry["hits"], n + entry["n"])
    return totals


def _hero(report: Mapping[str, Any]) -> str:
    window, counts = report["window"], report["observations"]
    title = f"Calibration, {cal._day(window['start'])} to {cal._day(window['end'])} UTC"
    if counts["executions"] == 0:
        return f'<header class="card" data-role="summary"><h1>{esc(title)}</h1><p>No observations yet. Run <code>agentpace learn</code> first.</p></header>'
    figures = "".join(
        f'<div class="figure{" lead" if name == "p80" else ""}"><span>{name.upper()} coverage (target {int(q * 100)}%)</span>'
        f"<strong>{esc(_percent(*totals))}</strong><span>{totals[0]}/{totals[1]}</span></div>"
        for (name, q), totals in zip(cal.QUANTILES, _overall_coverage(report).values())
    )
    estimates = report["estimates"]
    flagged = sum(bool(g["flags"]) for g in report["groups"])
    note = (
        f"{estimates['estimated']} of {estimates['scored']} scored runs this window had a dispatch estimate; "
        f"{fmt.count(len(report['misses']), 'largest miss', 'largest misses')} and {fmt.count(flagged, 'drift flag')} below. "
        "Evidence, not a verdict. Private: holds thread ids, transcript paths and vigil labels."
    )
    return (
        f'<header class="card" data-role="summary"><h1>{esc(title)}</h1>'
        f'<div class="figures">{figures}</div><p class="muted">{esc(note)}</p></header>'
    )


def _coverage_card(report: Mapping[str, Any]) -> str:
    rows = []
    for g in report["groups"]:
        this, before = g["this"], g["previous"]
        if not (this["n_estimated"] or before["n_estimated"]):
            continue
        coverage = this["coverage"]
        flags = "".join(f'<span class="badge warn">{esc(f)}</span>' for f in g["flags"])
        rows.append([
            esc(g["model"]),
            "all kinds" if g["kind"] == ALL_KINDS else esc(g["kind"]),
            f"{this['n']} ({this['n_estimated']})",
            esc(cal._seconds(this["mean_pinball_s"])),
            esc(cal._coverage(coverage["p50"])),
            esc(cal._coverage(coverage["p80"])),
            esc(cal._coverage(coverage["p90"])),
            flags,
        ])
    body = (
        _table(["model", "kind", "n (estimated)", "loss s", "P50", "P80", "P90", "flags"], rows)
        if rows
        else "<p>No scored run had a dispatch estimate.</p>"
    )
    return (
        '<section class="card" data-role="coverage"><h2>Coverage by model and kind</h2>'
        '<p class="muted">hits/n: the share of actuals at or under the estimate, target 50/80/90. Loss is mean pinball in seconds.</p>'
        f"{body}</section>"
    )


def _flags_card(report: Mapping[str, Any]) -> str:
    t = report["thresholds"]
    items = []
    for g in report["groups"]:
        this, before = g["this"], g["previous"]
        for flag in g["flags"]:
            if flag == "p80-undercoverage":
                p80 = this["coverage"]["p80"]
                detail = f"P80 covered {p80['hits']}/{p80['n']} actuals (target 0.8, flag below {t['p80_undercoverage_below']})"
            else:
                detail = f"scaled loss {this['relative_loss']:.2f} vs {before['relative_loss']:.2f} earlier (flag at x{t['loss_up_ratio']} and +{t['loss_up_floor']})"
            kind = "all kinds" if g["kind"] == ALL_KINDS else g["kind"]
            items.append(f'<li>{esc(g["model"])} / {esc(kind)}: <strong>{esc(flag)}</strong>, {esc(detail)}</li>')
    body = (
        f"<ul>{''.join(items)}</ul>"
        if items
        else f"<p>None. A flag needs at least {t['min_flag_n']} scored runs in the window it judges.</p>"
    )
    return f'<section class="card" data-role="flags"><h2>Drift flags</h2>{body}</section>'


def _new_card(report: Mapping[str, Any]) -> str:
    t = report["thresholds"]
    rows = [
        [esc(n["model"]), "all kinds" if n["kind"] == ALL_KINDS else esc(n["kind"]), esc(n["reason"]),
         str(n["runs"]), str(n["with_estimate"]), esc(cal._day(n["first_seen"])), str(n["n_this_window"])]
        for n in report["new"]
    ]
    body = (
        _table(["model", "kind", "reason", "run-level history", "with estimate", "first seen", "runs this window"], rows)
        + f'<p class="muted">History counts runs from T3 and transcripts; thin below {t["min_history"]}.</p>'
        if rows
        else "<p>None.</p>"
    )
    return f'<section class="card" data-role="new"><h2>New or thin history</h2>{body}</section>'


def _miss_card(miss: Mapping[str, Any]) -> str:
    estimate = ", ".join(f"{q.upper()} {cal._seconds(v)} s" for q, v in miss["estimate"].items() if v is not None)
    pointers = esc(cal._pointer_text(miss["pointers"]))
    return (
        '<article class="item">'
        f'<div class="head"><h3>{esc(miss["model"])} / {esc(miss["kind"])}</h3><span class="badge warn">{esc(miss["direction"])}</span></div>'
        f'<p>Actual {esc(cal._seconds(miss["actual_s"]))} s against {esc(estimate)}: {miss["actual_over_estimate"]:.2f}x.</p>'
        f'<p class="muted">{esc(miss["category"])}, loss {esc(cal._seconds(miss["mean_pinball_s"]))} s</p>'
        f'<p class="muted">Open it: {pointers}</p></article>'
    )


def _misses_card(report: Mapping[str, Any]) -> str:
    misses = report["misses"]
    body = (
        f'<div class="cards">{"".join(_miss_card(m) for m in misses)}</div>'
        if misses
        else "<p>No scored run had a dispatch estimate this window.</p>"
    )
    return f'<section class="card" data-role="misses"><h2>Largest misses ({len(misses)})</h2>{body}</section>'


def render(report: Mapping[str, Any]) -> str:
    cards = [_hero(report)]
    if report["observations"]["executions"]:
        cards += [_misses_card(report), _flags_card(report), _coverage_card(report), _new_card(report)]
    return page("agentpace calibration", cards, report)
