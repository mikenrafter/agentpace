"""`agentpace eta --html`: the deliverable ETA as cards, with the `--json` result embedded.

The snapshot is `result` unchanged. Model, kind and names are not in the forecast result, so the
cards read them from the plan beside it. Every time and saving printed here is read from
`result["schedule"]`, `result["speedups"]` or a task's summary, all produced by `deliverable`; this
module only formats them and places bars. The agent's own notes are kept apart in a tinted list.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agentpace import render as fmt
from agentpace.state_html import esc, page

FINISHED = ("done", "removed")
CHART_COLOURS = 6
LANE_START_REM, LANE_SPAN_REM, LANE_STEP_REM = 2.2, 0.9, 0.4  # the rail is 3.25 rem; lanes fit right of the tick labels
LOW_CONFIDENCE = "Low-confidence plan: research tasks are unbounded; totals are illustrative, not a commitment."


def _percent(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") + "%"


def _colour(index: int) -> str:
    return f"var(--s-c{index % CHART_COLOURS + 1})"


def _compact(seconds: float) -> str:
    """Tick labels on a 3.25 rem rail: `35m`, `2.4h`."""
    if seconds == 0:
        return "0"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _figure(label: str, seconds: float, lead: bool = False) -> str:
    return f'<div class="figure{" lead" if lead else ""}"><span>{esc(label)}</span><strong>{esc(fmt.duration(seconds))}</strong></div>'


def _hero(result: Mapping[str, Any], plan: Mapping[str, Any], finished: int, open_tasks: int) -> str:
    notes = [f"{fmt.count(open_tasks, 'task')} open, {finished} finished. From now; what changes it is listed at the foot."]
    schedule = result.get("schedule", {})
    if open_tasks:
        notes.append(f"Agent cap {schedule.get('agent_cap', result.get('agent_cap', 1))}.")
    if schedule.get("critical_path"):
        notes.append(f"Critical path: {' → '.join(map(str, schedule['critical_path']))}.")
    if result.get("elapsed_s"):
        notes.append(f"{fmt.duration(result['elapsed_s'])} already elapsed: whole-deliverable median {fmt.duration(result['total_p50'])}, slip {fmt.duration(result['total_p80'])}.")
    if result.get("deadline_met") is False:
        notes.append("Target is not met. No undeclared work was removed.")
    elif result.get("deadline_met") is True:
        notes.append("Commit meets the target.")
    banner = "" if result.get("reliable", True) else f'<p class="banner">{esc(LOW_CONFIDENCE)}</p>'
    return (
        '<header class="card" data-role="summary">'
        f"<h1>{esc(plan.get('deliverable', 'Deliverable'))}</h1>"
        '<div class="figures">'
        f"{_figure('Commit', result['commit'], lead=True)}{_figure('Median', result['p50'])}{_figure('Slip (P80)', result['p80'])}"
        "</div>"
        f"{banner}"
        f'<p class="muted">{esc(" ".join(notes))}</p>'
        "</header>"
    )


def _task_notes(applied: Sequence[Mapping[str, Any]], kept: str | None) -> list[str]:
    parts = []
    for entry in applied:
        parts.append(
            f'<p class="note"><span class="badge accent">Shortcut applied</span> saves {esc(fmt.duration(entry["savings_s"]))} on the task '
            f'(ETA {esc(fmt.duration(entry["eta_savings_s"]))}); risk: {esc(entry["risk"])}; basis: {esc(entry["basis"])}</p>'
        )
    if kept:
        parts.append(f'<p><span class="badge">Kept in full</span> {esc(kept)}</p>')
    return parts


NOTE_LABELS = {"cut": "Cut", "parallel": "Parallel", "note": "Note"}


def _agent_notes(notes: Sequence[Mapping[str, Any]]) -> str:
    """The agent's notes in a tinted list. A saving is the agent's claim unless the scheduler reproduced it."""
    if not notes:
        return ""
    items = []
    for note in notes:
        line = f'<b>{esc(NOTE_LABELS.get(note["kind"], note["kind"]))}:</b> {esc(note["text"])}'
        if note.get("saves_s") is not None:
            saving = esc(fmt.duration(note["saves_s"]))
            if note.get("backed"):
                line += f' <span class="badge accent">confirmed by the scheduler</span> saves {saving}'
            else:
                line += f' <span class="badge">claimed</span> saves {saving}; scheduler finds {esc(fmt.duration(note["scheduler_saves_s"]))}'
        items.append(f"<li>{line}</li>")
    return f'<p class="muted">Agent notes (the agent\'s words, not computed)</p><ul class="notes">{"".join(items)}</ul>'


def _card(
    summary: Mapping[str, Any],
    task: Mapping[str, Any],
    applied: Sequence[Mapping[str, Any]],
    kept: str | None,
    span: Mapping[str, Any] | None,
    critical: bool,
    colour: int | None,
) -> str:
    status = summary["status"]
    badges = [f'<span class="badge accent">{esc(status)}</span>', f'<span class="badge">{esc(summary["type"])}</span>']
    if task.get("model") or task.get("kind"):
        badges.append(f'<span class="badge">{esc(" · ".join(str(task[k]) for k in ("model", "kind") if task.get(k)))}</span>')
    if summary.get("needs_owner"):
        badges.append('<span class="badge warn">needs owner</span>')
    if not summary.get("reliable", True):
        badges.append('<span class="badge warn">low confidence</span>')
    heading = [f'<h3>{esc(summary["id"])}</h3>']
    if critical:
        heading.append('<span class="badge crit">critical path</span>')
    if span and span["wait_s"] > 0:
        heading.append('<span class="badge">waits for a free agent</span>')
    subtitle = next((str(task[k]) for k in ("name", "label", "what") if task.get(k)), "")
    facts = []
    if span and status not in FINISHED:
        facts.append(("Runs", f"{fmt.duration(span['start_s'])} → {fmt.duration(span['end_s'])}"))
    if status not in FINISHED:
        facts += [
            ("Commit cut", f"{fmt.duration(summary['commit'])} at P{summary['quantile'] * 100:g}, {summary['appetite']} ({summary['source']})"),
            ("Median", fmt.duration(summary["p50"])),
            ("P80", fmt.duration(summary["p80"])),
        ]
    if status == "running":
        facts.append(("Elapsed", fmt.duration(summary["elapsed_s"])))
    if status in FINISHED and task.get("actual_s") is not None:
        facts.append(("Actual", fmt.duration(float(task["actual_s"]))))
    if task.get("depends_on"):
        facts.append(("After", ", ".join(map(str, task["depends_on"]))))
    lines = [f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in facts]
    extra = []
    if subtitle:
        extra.append(f'<p class="muted">{esc(subtitle)}</p>')
    for key in ("note", "notes", "basis"):
        if isinstance(task.get(key), str) and task[key]:
            extra.append(f'<p class="muted">{esc(task[key])}</p>')
    if summary.get("warning"):
        extra.append(f'<p class="banner">{esc(summary["warning"])}</p>')
    if summary.get("needs_owner"):
        extra.append('<p class="muted">Requested appetite needs the owner; the configured cut is kept.</p>')
    extra.extend(_task_notes(applied, kept))
    if span and status not in FINISHED and span["start_with_extra_agent_s"] < span["start_s"]:
        extra.append(f'<p class="muted">One more agent would start this at {esc(fmt.duration(span["start_with_extra_agent_s"]))} (dashed line on the rail).</p>')
    style = f' style="--c:{_colour(colour)}"' if colour is not None else ""
    return (
        f'<article class="card{" task" if colour is not None else ""}" data-task="{esc(summary["id"])}" data-status="{esc(status)}"{style}>'
        f'<div class="head">{"".join(heading)}</div>'
        f'<div class="badges">{"".join(badges)}</div>{"".join(extra)}'
        f'<dl class="facts">{"".join(lines)}</dl>{_agent_notes(summary.get("notes", []))}</article>'
    )


def _kept_reason(task: Mapping[str, Any], plan: Mapping[str, Any], keep: Sequence[str], no_defer: Sequence[str]) -> str | None:
    if task["id"] in no_defer:
        return "this task was named in --no-defer."
    for entry in plan.get("tradeoffs", []):
        protected = sorted(set(entry.get("protects", [])) & set(keep))
        if entry.get("task") == task["id"] and entry["id"] in plan.get("apply", []) and protected:
            return f"shortcut {entry['id']} was not applied: it would cut {', '.join(protected)}."
    return None


def _speedups_card(result: Mapping[str, Any]) -> str:
    """What would shorten the commit: every row is a makespan difference the scheduler computed."""
    entries = result.get("speedups", [])
    helps = [e for e in entries if e["saves_s"] > 0]
    if not helps:
        body = '<p class="muted">No computed speed-up: more agents or the plan\'s catalogued cuts would not shorten the commit.</p>'
        return f'<section class="card whatif" data-role="speedups"><h2>What would make it faster</h2>{body}</section>'
    rows = []
    for entry in helps:
        owner = ' <span class="badge warn">needs an owner grant</span>' if entry.get("needs_owner") else ""
        risk = f' <span class="muted">risk: {esc(entry["risk"])}</span>' if entry.get("risk") else ""
        rows.append(
            f'<div class="row"><span>{esc(entry["label"])}{risk}{owner}</span>'
            f'<strong class="gain">−{esc(fmt.duration(entry["saves_s"]))} → {esc(fmt.duration(entry["commit_s"]))}</strong></div>'
        )
    idle = [e for e in entries if e["saves_s"] <= 0 and e["kind"] != "combined"]
    notes = [
        "Computed by the scheduler from dependencies and the cap; each row is the saving and the commit it would give. "
        "Raising the cap needs an authorized extra-agents entry. Notes on the cards below are the agent's."
    ]
    if idle:
        names = ", ".join(f'{e["label"]} ({e["id"]})' for e in idle)
        notes.append(f"No effect on the commit: {names}.")
    return (
        f'<section class="card whatif" data-role="speedups"><h2>What would make it faster</h2>{"".join(rows)}'
        f'<p class="muted">{esc(" ".join(notes))}</p></section>'
    )


def _lanes(spans: Mapping[str, Mapping[str, Any]], ids: Sequence[str]) -> dict[str, int]:
    """Lane per task so bars that overlap in time (a ghost counts from its earlier start) never share a line."""
    occupied = {i: (min(spans[i]["start_with_extra_agent_s"], spans[i]["start_s"]), spans[i]["end_s"]) for i in ids}
    last_end: list[float] = []
    lanes = {}
    for i in sorted(ids, key=lambda i: (occupied[i][0], spans[i]["start_s"], ids.index(i))):
        free = next((n for n, end in enumerate(last_end) if end <= occupied[i][0]), None)
        if free is None:
            last_end.append(occupied[i][1])
            free = len(last_end) - 1
        else:
            last_end[free] = occupied[i][1]
        lanes[i] = free
    return lanes


def _rail(schedule: Mapping[str, Any], ids: Sequence[str]) -> str:
    """Time runs down; one thin bar per open task from its scheduled start to end, a dashed ghost where one more agent starts it earlier."""
    makespan = schedule["makespan_s"]
    spans = schedule["tasks"]
    lanes = _lanes(spans, ids)
    step = min(LANE_STEP_REM, LANE_SPAN_REM / max(1, max(lanes.values(), default=0))) if lanes else LANE_STEP_REM
    marks = [
        f'<span class="tick" style="top:{_percent(100 * tick / makespan)}">{esc(_compact(tick))}</span>' for tick in schedule["ticks_s"]
    ]
    bars = []
    for index, task_id in enumerate(ids):
        span = spans[task_id]
        place = f"left:{LANE_START_REM + lanes[task_id] * step:g}rem;top:{_percent(100 * span['start_s'] / makespan)};height:{_percent(100 * (span['end_s'] - span['start_s']) / makespan)}"
        bars.append(f'<span class="bar" data-bar="{esc(task_id)}" style="{place};background:{_colour(index)}"></span>')
        if span["start_with_extra_agent_s"] < span["start_s"]:
            ghost = (
                f"left:{LANE_START_REM + lanes[task_id] * step:g}rem;top:{_percent(100 * span['start_with_extra_agent_s'] / makespan)};"
                f"height:{_percent(100 * (span['end_s'] - span['start_s']) / makespan)}"
            )
            bars.append(f'<span class="ghost" data-ghost="{esc(task_id)}" style="{ghost};color:{_colour(index)}"></span>')
    return f'<div class="rail" aria-label="Schedule from now, time runs down">{"".join(marks)}{"".join(bars)}</div>'


def render(result: Mapping[str, Any], plan: Mapping[str, Any], *, keep: Sequence[str] = (), no_defer: Sequence[str] = ()) -> str:
    tasks = {str(task["id"]): task for task in plan.get("tasks", [])}
    schedule = result.get("schedule", {})
    spans = schedule.get("tasks", {})
    critical = set(schedule.get("critical_path", []))
    open_ids = [s["id"] for s in result["tasks"] if s["status"] not in FINISHED]
    open_cards, finished_cards = [], []
    for summary in result["tasks"]:
        task = dict(tasks.get(summary["id"], {}), id=summary["id"])
        applied = [entry for entry in result.get("tradeoffs", []) if entry.get("task") == summary["id"]]
        is_open = summary["status"] not in FINISHED
        card = _card(
            summary, task, applied, _kept_reason(task, plan, keep, no_defer), spans.get(summary["id"]),
            summary["id"] in critical, open_ids.index(summary["id"]) if is_open else None,
        )
        (open_cards if is_open else finished_cards).append(card)
    cards = [_hero(result, plan, len(finished_cards), len(open_cards))]
    if open_cards:
        cards.append(_speedups_card(result))
        rail = _rail(schedule, open_ids) if schedule.get("makespan_s") else "<div></div>"
        cards.append(f'<section class="timeline" data-role="tasks">{rail}<div class="tasks">{"".join(open_cards)}</div></section>')
    spend = [
        f'<li>Available with <code>{esc(e["available_with"])}</code> and an owner grant: saves {esc(fmt.duration(e.get("eta_gain_s", 0)))}; spends {esc(e["cost"])}.</li>'
        for e in result.get("available_spend", [])
    ]
    if spend:
        cards.append(f'<section class="card" data-role="spend"><h2>Needs an owner grant</h2><ul>{"".join(spend)}</ul></section>')
    if finished_cards:
        cards.append(
            f'<details class="card" data-role="finished"><summary>Finished and removed ({len(finished_cards)})</summary>'
            f'<div class="cards">{"".join(finished_cards)}</div></details>'
        )
    changes = "".join(f"<li>{esc(item)}</li>" for item in result.get("what_would_change_it", []))
    cards.append(f'<section class="card" data-role="changes"><h2>What changes this ETA</h2><ul>{changes}</ul></section>')
    return page(f"{plan.get('deliverable', 'Deliverable')} ETA", cards, result)
