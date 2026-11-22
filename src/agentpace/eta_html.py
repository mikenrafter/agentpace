"""`agentpace eta --html`: the deliverable ETA as cards, with the `--json` result embedded.

The snapshot is `result` unchanged. Model, kind, names and notes are not in the forecast result,
so the cards read them from the plan beside it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agentpace import render as fmt
from agentpace.state_html import esc, page

FINISHED = ("done", "removed")
LOW_CONFIDENCE = "Low-confidence plan: research tasks are unbounded; totals are illustrative, not a commitment."


def _figure(label: str, seconds: float, lead: bool = False) -> str:
    return f'<div class="figure{" lead" if lead else ""}"><span>{esc(label)}</span><strong>{esc(fmt.duration(seconds))}</strong></div>'


def _hero(result: Mapping[str, Any], plan: Mapping[str, Any], finished: int, open_tasks: int) -> str:
    notes = [f"{fmt.count(open_tasks, 'task')} open, {finished} finished. From now; what changes it is listed at the foot."]
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


def _task_notes(task: Mapping[str, Any], applied: Sequence[Mapping[str, Any]], kept: str | None) -> list[str]:
    parts = []
    for entry in applied:
        parts.append(
            f'<p><span class="badge accent">Shortcut applied</span> saves {esc(fmt.duration(entry["savings_s"]))} on the task '
            f'(ETA {esc(fmt.duration(entry["eta_savings_s"]))}); risk: {esc(entry["risk"])}; basis: {esc(entry["basis"])}</p>'
        )
    if kept:
        parts.append(f'<p><span class="badge">Kept in full</span> {esc(kept)}</p>')
    return parts


def _card(summary: Mapping[str, Any], task: Mapping[str, Any], applied: Sequence[Mapping[str, Any]], kept: str | None) -> str:
    status = summary["status"]
    badges = [f'<span class="badge accent">{esc(status)}</span>', f'<span class="badge">{esc(summary["type"])}</span>']
    if task.get("model") or task.get("kind"):
        badges.append(f'<span class="badge">{esc(" · ".join(str(task[k]) for k in ("model", "kind") if task.get(k)))}</span>')
    if summary.get("needs_owner"):
        badges.append('<span class="badge warn">needs owner</span>')
    if not summary.get("reliable", True):
        badges.append('<span class="badge warn">low confidence</span>')
    subtitle = next((str(task[k]) for k in ("name", "label", "what") if task.get(k)), "")
    facts = []
    if status not in FINISHED:
        facts += [
            ("Commit cut", f"{fmt.duration(summary['commit'])} at P{summary['quantile'] * 100:g}, {summary['appetite']} ({summary['source']})"),
            ("Median", fmt.duration(summary["p50"])),
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
    extra.extend(_task_notes(task, applied, kept))
    return (
        f'<article class="card" data-task="{esc(summary["id"])}" data-status="{esc(status)}">'
        f'<div class="head"><h3>{esc(summary["id"])}</h3></div>'
        f'<div class="badges">{"".join(badges)}</div>{"".join(extra)}'
        f'<dl class="facts">{"".join(lines)}</dl></article>'
    )


def _kept_reason(task: Mapping[str, Any], plan: Mapping[str, Any], keep: Sequence[str], no_defer: Sequence[str]) -> str | None:
    if task["id"] in no_defer:
        return "this task was named in --no-defer."
    for entry in plan.get("tradeoffs", []):
        protected = sorted(set(entry.get("protects", [])) & set(keep))
        if entry.get("task") == task["id"] and entry["id"] in plan.get("apply", []) and protected:
            return f"shortcut {entry['id']} was not applied: it would cut {', '.join(protected)}."
    return None


def render(result: Mapping[str, Any], plan: Mapping[str, Any], *, keep: Sequence[str] = (), no_defer: Sequence[str] = ()) -> str:
    tasks = {str(task["id"]): task for task in plan.get("tasks", [])}
    open_cards, finished_cards = [], []
    for summary in result["tasks"]:
        task = tasks.get(summary["id"], {})
        applied = [entry for entry in result.get("tradeoffs", []) if entry.get("task") == summary["id"]]
        card = _card(summary, dict(task, id=summary["id"]), applied, _kept_reason(dict(task, id=summary["id"]), plan, keep, no_defer))
        (finished_cards if summary["status"] in FINISHED else open_cards).append(card)
    cards = [_hero(result, plan, len(finished_cards), len(open_cards))]
    if open_cards:
        cards.append(f'<section class="cards" data-role="tasks">{"".join(open_cards)}</section>')
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
