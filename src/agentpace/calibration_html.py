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


def _width(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") + "%"


def _bars(coverage: Mapping[str, Mapping[str, Any]]) -> str:
    """One row per quantile: a bar for the share of actuals at or under the estimate, a line at the target."""
    rows = []
    for name, target in cal.QUANTILES:
        entry = coverage[name]
        rate = entry["rate"] or 0.0
        low = " low" if entry["n"] and rate < target else ""
        rows.append(
            f"<span>{name.upper()}</span>"
            f'<span class="track"><span class="fill{low}" style="width:{_width(rate * 100)}"></span>'
            f'<span class="target" style="left:{_width(target * 100)}"></span></span>'
            f"<span>{esc(cal._coverage(entry))}</span>"
        )
    return f'<div class="cov">{"".join(rows)}</div>'


def _group_card(model: str, kind: str, group: Mapping[str, Any] | None, thin: Mapping[str, Any] | None, thresholds: Mapping[str, Any]) -> str:
    """One model and kind: flags, one muted line, coverage bars; thin history adds its own badge and line."""
    badges = [f'<span class="badge warn">{esc(flag)}</span>' for flag in (group["flags"] if group else [])]
    if thin:
        badges.append('<span class="badge">thin history</span>')
    lines = []
    bars = ""
    if group and group["this"]["n_estimated"]:
        this, before = group["this"], group["previous"]
        line = f"{fmt.count(this['n'], 'run')}, {this['n_estimated']} estimated · loss {cal._seconds(this['mean_pinball_s'])} s"
        if "loss-up" in group["flags"]:
            line += f" · scaled loss {before['relative_loss']:.2f} → {this['relative_loss']:.2f}"
        lines.append(line)
        bars = _bars(this["coverage"])
    elif group:
        lines.append(f"{fmt.count(group['this']['n'], 'run')}, none estimated this window")
    if thin:
        lines.append(
            f"{thin['runs']} run-level, {thin['with_estimate']} with an estimate · first seen {cal._day(thin['first_seen'])} · "
            f"{thin['n_this_window']} this window · {thin['reason']} · thin below {thresholds['min_history']} run-level observations"
        )
    heading = f"{model} · {'all kinds' if kind == ALL_KINDS else kind}"
    muted = "".join(f'<p class="muted">{esc(line)}</p>' for line in lines)
    return (
        f'<div class="group" data-group data-model="{esc(model)}" data-kind="{esc(kind)}">'
        f'<div class="head"><h3>{esc(heading)}</h3>{"".join(badges)}</div>'
        f"{muted}{bars}</div>"
    )


def _coverage_card(report: Mapping[str, Any]) -> str:
    """Every model and kind with a dispatch estimate, and every thin-history one, as cards in one grid."""
    thin = {(n["model"], n["kind"]): n for n in report["new"]}
    groups = {
        (g["model"], g["kind"]): g
        for g in report["groups"]
        if g["this"]["n_estimated"] or g["previous"]["n_estimated"]
    }
    keys = list(groups) + [key for key in thin if key not in groups]
    cards = [_group_card(model, kind, groups.get((model, kind)), thin.get((model, kind)), report["thresholds"]) for model, kind in keys]
    body = f'<div class="groups">{"".join(cards)}</div>' if cards else "<p>No scored run had a dispatch estimate.</p>"
    return (
        '<section class="card" data-role="coverage"><h2>Coverage by model and kind</h2>'
        '<p class="muted">Bar: the share of actuals at or under the estimate, hits/n beside it. Line: the target, 50/80/90. '
        'Loss is mean pinball in seconds. A thin-history card says why it is thin.</p>'
        f"{body}</section>"
    )


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
        cards += [_misses_card(report), _flags_card(report), _coverage_card(report)]
    return page("agentpace calibration", cards, report)
