"""Incremental, idempotent ingest of finished runs into a local observation store.

`agentpace learn` reads every source agentpace already knows (the T3 database, the vigil item
ledger, Claude subagent and Codex transcripts, plan files) for runs finished since the last
cursor, and appends one observation per run to `<state_dir>/observations.jsonl`. An observation
keeps the estimate that existed at dispatch (when there was one), the actual duration, the model,
the kind, how each label was obtained, and a category. `calibration` reads the store and puts
the evidence in the weekly brief; nothing here tunes a rule.

Rows come from the shared readers (`t3_history.read_database`, `usage_gap.load_t3_runs`,
`local_runs.ledger_row`, `run_labels.codex_turns` / `claude_turns`), so a learned row carries the
features the estimator weights; `extend` adds the learn fields. Names go through
`local_runs.model_name` / `kind_name`, the normalizer estimate-run uses.

Invariants:
- Read-only sources: SQLite opens with `mode=ro`; transcripts, ledger and plans are only read.
- Pointers, never prompt text: briefs are read in memory for the kind keyword and the derived
  features, then dropped. Paths and thread ids are kept as pointers (the store is private state).
- Idempotent: an observation's id is derived from its source identity, never from ingest time; a
  row is appended only when its id is new or its content changed (the later line supersedes).
- Crash-safe: rows are appended and synced before the cursor moves (atomic replace, then a
  directory sync). A crash in between re-reads the same window; unchanged rows are not re-added.
- Agent runs only are samples: orchestrator turns, machine jobs and check-ins are stored under
  their own category; a vigil hold is scored only through the one run it watched.

Mating parts:
- `calibration` reads this store through `read_store` and `collapse`.
- `local_runs.cmd_estimate_run --learned` feeds rows from `reference_rows` to the estimator; a
  learned row is a `local_runs.make_row` row plus the fields below, so both stores are one shape.
- `plan_state.sync_ledger` writes `ledger_id` events; `vigil_execution_key` must give the same
  value for the same ledger item.
- `t3_history.hold_index(agents=True)` / `call_index` link a vigil item to its run.
"""

from __future__ import annotations

import copy
import fcntl
import functools
import json
import math
import os
import re
import sqlite3
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentpace import local_runs as lr
from agentpace import paths
from agentpace import run_labels
from agentpace import t3_history
from agentpace import usage_gap as ug

STORE_SCHEMA = 4
SOURCES = ("t3-database", "vigil-ledger", "claude-subagent", "codex-transcript", "plan")
RUN_LEVEL_SOURCES = ("t3-database", "claude-subagent", "codex-transcript")
TIME_SOURCES = ("t3-database", "vigil-ledger")  # cursor: latest finished time, re-read with an overlap
FILE_SOURCES = ("claude-subagent", "codex-transcript", "plan")  # cursor: per-file (mtime_ns, size)
# Rows that are not agent runs: stored and counted, never an agent-run sample or a scored
# estimate. A main-thread T3 turn includes owner interaction; machine jobs and check-in timers
# are holds, not runs.
ROLE_CATEGORIES = ("orchestrator-turn", "machine", "checkin")
CATEGORIES = ("usage-wait", "over-12h", "research", "ordinary") + ROLE_CATEGORIES
OVER_LONG_S = ug.DEFAULT_MAX_GAP_S
HOLD_SOURCES = ("vigil-ledger", "plan")

# Clock skew between a writer and this reader: a run that finished just before the cursor can be
# written just after it. Re-reading the window is free because ids dedupe.
DATABASE_OVERLAP_S = 6 * 3600.0
LEDGER_OVERLAP_S = 6 * 3600.0

DONE_OUTCOMES = ("done", "completed", "success", "exited")


@dataclass(frozen=True)
class Sources:
    """Where each source lives; `None` means the source is not configured."""

    t3_db: Path | None = None
    vigil_ledger: Path | None = None
    claude_projects: Path | None = None
    codex_sessions: Path | None = None
    plan_dirs: Sequence[Path] = ()


def default_sources() -> Sources:
    """Sources from the environment, then the harness defaults.

    `AGENTPACE_T3_DB`, `AGENTPACE_VIGIL_LEDGER`, `AGENTPACE_CLAUDE_PROJECTS`,
    `AGENTPACE_CODEX_SESSIONS`, `AGENTPACE_PLAN_DIRS` (os.pathsep separated). Plans default to
    every `<project>.research/plans` directory beside the current directory's project, the layout
    the agentpace skill recommends.
    """
    env = os.environ
    home = Path.home()
    state_home = Path(env.get("XDG_STATE_HOME") or home / ".local" / "state")
    plan_dirs = env.get("AGENTPACE_PLAN_DIRS")
    return Sources(
        t3_db=Path(env.get("AGENTPACE_T3_DB") or home / ".t3" / "userdata" / "statev2.sqlite"),
        vigil_ledger=Path(env.get("AGENTPACE_VIGIL_LEDGER") or state_home / "vigil" / "items.jsonl"),
        claude_projects=Path(env.get("AGENTPACE_CLAUDE_PROJECTS") or home / ".claude" / "projects"),
        codex_sessions=Path(env.get("AGENTPACE_CODEX_SESSIONS") or home / ".codex" / "sessions"),
        plan_dirs=tuple(Path(p) for p in plan_dirs.split(os.pathsep) if p)
        if plan_dirs is not None
        else tuple(sorted(Path.cwd().parent.glob("*.research/plans"))),
    )


def categorize(kind: str, actual_s: float, usage_limit: bool, role: str | None = None) -> str:
    """Exclusive category: a role that is not an agent run, then an evidenced usage-limit ending,
    then run length, then research."""
    if role:
        if role not in ROLE_CATEGORIES:
            raise ValueError(f"unknown role category: {role!r}")
        return role
    if usage_limit:
        return "usage-wait"
    if actual_s > OVER_LONG_S:
        return "over-12h"
    if kind == "research":
        return "research"
    return "ordinary"


def is_agent_run(row: Mapping[str, Any]) -> bool:
    """True for a row that samples agent work (not an orchestrator turn, machine job or check-in)."""
    return row.get("category") not in ROLE_CATEGORIES


def vigil_execution_key(hold: str, label: str, start: float) -> str:
    """Identity of one vigil ledger item; equals `plan_state.sync_ledger`'s `ledger_id`."""
    return lr.identity((hold, label, float(start)))


def _seconds(value: Any) -> float | None:
    """A positive finite number of seconds, else None (bool, inf, nan, zero and text are not)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value > 0 else None


def extend(
    row: Mapping[str, Any],
    *,
    key: Iterable[Any],
    source: str | None = None,
    model_raw: str,
    kind_source: str,
    label_source: str,
    model_source: str = "source",
    provider: str = "",
    native_session: str | None = None,
    execution_key: str | None = None,
    estimate: Mapping[str, Any] | None = None,
    estimate_source: str | None = None,
    ceiling_s: Any = None,
    estimator: str | None = None,
    usage_limit: bool = False,
    role: str | None = None,
    pointers: Mapping[str, Any] | None = None,
    delegated: bool | None = None,
    ingested_at: float = 0.0,
) -> dict[str, Any]:
    """One observation from a shared reader's row (`local_runs.make_row` shape, plus whatever
    features that reader adds), extended with the learn fields. Matching aids (brief hash and
    anchors) are dropped: they are for in-memory matching, not storage.

    `estimate` holds quantiles an estimator produced. `ceiling_s` is a hand-set overdue alarm (a
    vigil `--expect-s` with no estimate behind it): kept apart, because it is a budget the owner
    chose, not a P80 whose coverage means anything."""
    result = {k: v for k, v in row.items() if k not in ("brief_hash", "brief_anchors", "brief_link")}
    quantiles = {q: _seconds((estimate or {}).get(q)) for q in ("p50", "p80", "p90")}
    ceiling = _seconds(ceiling_s)
    result.update(
        schema=STORE_SCHEMA,
        source=source or result["source"],
        id=lr.identity(tuple(key)),
        model_raw=model_raw,
        model_source=model_source,
        provider=provider or result.get("provider") or "",
        kind_source=kind_source,
        label_source=label_source,
        native_session=native_session,
        execution_key=execution_key,
        estimate=quantiles,
        estimate_source=estimate_source if ceiling is not None or any(v is not None for v in quantiles.values()) else None,
        ceiling_s=ceiling,
        estimator=estimator,
        expect_s=quantiles["p80"] or ceiling,
        category=categorize(result["kind"], result["actual_s"], usage_limit, role),
        censor_reason="usage-limit" if usage_limit else "none",
        delegated=delegated if delegated is not None else result.get("delegated"),
        pointers={k: v for k, v in (pointers or {}).items() if v is not None},
        ingested_at=ingested_at,
    )
    return result


def observation(
    *,
    source: str,
    key: Iterable[Any],
    start: float,
    end: float,
    model_raw: str,
    kind: str,
    model: str | None = None,
    project: str = "",
    session: str = "",
    brief: str = "",
    run_id: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """An observation built from scratch (plan tasks, synthetic tests): a `make_row` row plus the
    learn fields of `extend`. Raises ValueError for a run without positive finite duration."""
    row = lr.make_row(start=start, end=end, model=model_raw, kind=kind, brief=brief, project=project, session=session, source=source)
    if model:
        row["model"] = lr.model_name(model)
    if run_id:
        row["run_id"] = run_id
    return extend(row, key=key, model_raw=model_raw, **fields)


# --- store ---------------------------------------------------------------------------------


def _open_private(path: Path, flags: int):
    """Open a text file for writing, created owner-only: the store holds ids and local paths."""
    descriptor = os.open(path, flags | os.O_WRONLY | os.O_CREAT, 0o600)
    os.chmod(path, 0o600)
    return os.fdopen(descriptor, "a" if flags & os.O_APPEND else "w", encoding="utf-8")


def _make_state_dir(state: Path) -> None:
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state, 0o700)
    for name in ("observations.jsonl", "cursor.json", "lock"):  # tighten files written by an older run
        if (state / name).exists():
            os.chmod(state / name, 0o600)


def _store_path(state: Path) -> Path:
    return Path(state) / "observations.jsonl"


def _cursor_path(state: Path) -> Path:
    return Path(state) / "cursor.json"


def _store_lines(state: Path) -> list[dict[str, Any]]:
    path = _store_path(state)
    if not path.exists():
        return []
    return [r for r in lr.json_lines(path) if r.get("id")]


def read_store(state: Path) -> list[dict[str, Any]]:
    """The current version of every observation, in first-seen order.

    A later line with the same id supersedes an earlier one (a resumed transcript appends its
    updated observation). A torn or foreign line is skipped."""
    latest: dict[str, dict[str, Any]] = {}
    for row in _store_lines(state):
        latest[row["id"]] = row
    return list(latest.values())


def read_cursor(state: Path) -> dict[str, Any]:
    """Per source: a finished-time watermark (T3, ledger) or per-file state `{path: [mtime_ns,
    size]}` (transcripts, plans). Anything malformed reads as absent, which only costs a re-read."""
    try:
        data = json.loads(_cursor_path(state).read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    cursor: dict[str, Any] = {}
    for name, value in data.items():
        if name in TIME_SOURCES and _finite_number(value):
            cursor[name] = float(value)
        elif name in FILE_SOURCES and isinstance(value, dict):
            cursor[name] = {str(k): list(v) for k, v in value.items() if isinstance(v, list) and len(v) == 2}
    return cursor


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _canonical(row: Mapping[str, Any]) -> str:
    """A row's content for change detection; the ingest time is not content."""
    return json.dumps({k: v for k, v in row.items() if k != "ingested_at"}, sort_keys=True)


def append_observations(state: Path, rows: Sequence[Mapping[str, Any]], current: dict[str, Any] | None = None) -> list[Mapping[str, Any]]:
    """Append each row whose id is new or whose content changed, durably; return those rows.

    `current` (id -> stored row) is read from the store when not given, and is updated in place.
    A torn final line (crash mid-append) is closed with a newline first so the next row does not
    fuse with it.
    """
    state = Path(state)
    _make_state_dir(state)
    if current is None:
        current = {r["id"]: r for r in read_store(state)}
    fresh: list[Mapping[str, Any]] = []
    for row in rows:
        stored = current.get(row["id"])
        if stored is not None and _canonical(stored) == _canonical(row):
            continue
        current[row["id"]] = dict(row)
        fresh.append(row)
    if not fresh:
        return []
    path = _store_path(state)
    needs_newline = False
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            needs_newline = handle.read(1) != b"\n"
    with _open_private(path, os.O_APPEND) as handle:
        handle.write(("\n" if needs_newline else "") + "".join(json.dumps(r, sort_keys=True) + "\n" for r in fresh))
        handle.flush()
        os.fsync(handle.fileno())
    return fresh


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_cursor(state: Path, cursor: Mapping[str, Any]) -> None:
    """Replace the cursor atomically, and sync the directory so the rename itself is durable."""
    path = _cursor_path(state)
    temporary = path.with_suffix(".tmp")
    with _open_private(temporary, os.O_TRUNC) as handle:
        handle.write(json.dumps({"schema": STORE_SCHEMA, **dict(sorted(cursor.items()))}, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _schema_of_cursor(state: Path) -> int | None:
    try:
        data = json.loads(_cursor_path(state).read_text())
    except (OSError, ValueError):
        return None
    return int(data.get("schema", 1)) if isinstance(data, dict) and _finite_number(data.get("schema", 1)) else 1


def _migrate(state: Path) -> int | None:
    """Set aside a store (and cursor) written by an older schema, so it is rebuilt from the
    sources. The store is derived data; mixing schemas would count runs twice. Returns the old
    schema, or None when nothing was moved."""
    lines = _store_lines(state)
    found = int(lines[0].get("schema", 1)) if lines else _schema_of_cursor(state)
    if found is None or found == STORE_SCHEMA:
        return None
    if _store_path(state).exists():
        os.replace(_store_path(state), state / f"observations.v{found}.jsonl")
    if _cursor_path(state).exists():
        os.replace(_cursor_path(state), state / f"cursor.v{found}.json")
    _fsync_directory(state)
    return found


def collapse(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per execution, across sources. Works on deep copies; the input is not changed.

    - A Codex transcript row whose session is the native session of a T3 run is dropped (T3
      wins: it has the run id and the usage-limit signals).
    - Rows sharing an `execution_key` (a vigil item and the plan task closed from it) merge: the
      vigil row stays; missing estimate quantiles, pointers and explicit labels come from the other.
    - A vigil hold linked to exactly one run moves its estimate (meta P50/P80) or hand-set
      ceiling onto that run and is dropped: the run's actual is what it predicted. Otherwise the
      hold stays, with a `link_status` saying why, and is never scored (`is_sample` is false).
    - A Claude subagent is linked to the T3 turn that ran it (`parent_run_id`). When that parent
      is itself an agent run, the child is `nested_in` it: the parent's actual already holds the
      child's time, so the child is not a second sample (`is_sample`).
    """
    rows = copy.deepcopy(list(rows))
    native = {r["native_session"] for r in rows if r.get("source") == "t3-database" and r.get("native_session")}
    kept = [r for r in rows if not (r.get("source") == "codex-transcript" and r.get("session") in native)]
    result = _resolve_holds(_merge_execution_keys(kept))
    _link_subagents(result)
    return result


def _merge_execution_keys(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    result: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: (r.get("source") != "vigil-ledger", r.get("start", 0))):
        key = row.get("execution_key")
        if not key:
            result.append(row)
        elif key not in merged:
            merged[key] = row
            result.append(row)
        else:
            _absorb(merged[key], row)
    return result


def _absorb(base: dict[str, Any], other: Mapping[str, Any]) -> None:
    """Fill what `base` lacks from another record of the same execution. The other record's
    estimate source is named only when it gave a quantile or the ceiling."""
    sources = {base["estimate_source"]} if base.get("estimate_source") else set()
    contributed = False
    for quantile, value in other["estimate"].items():
        if base["estimate"].get(quantile) is None and value is not None:
            base["estimate"][quantile] = value
            contributed = True
    for field in ("ceiling_s", "estimator"):
        if base.get(field) is None and other.get(field) is not None:
            base[field] = other[field]
            contributed |= field == "ceiling_s"
    if contributed and other.get("estimate_source"):
        sources.add(other["estimate_source"])
    base["estimate_source"] = "+".join(sorted(sources)) or None
    base["expect_s"] = base["estimate"].get("p80") or base.get("ceiling_s")
    base["pointers"] = {**other.get("pointers", {}), **base.get("pointers", {})}
    if base.get("model") == "unknown" and other.get("model", "unknown") != "unknown":
        base.update(model=other["model"], model_raw=other.get("model_raw", ""), model_source=other.get("model_source", "none"))
    if base.get("kind") == "other" and other.get("kind", "other") != "other":
        base.update(kind=other["kind"], kind_source=other.get("kind_source", "none"))


# T3 names a delegated task `command:mcp:<call>:delegate-task:<id>` (a follow-up sent with
# t3_thread_send: `...:thread-send:<id>`) and hands it out url-encoded behind a prefix: the task
# id `node:delegated-task:`, the parent's background item `turn-item:delegated-task:`, the child
# thread `thread:delegated-task:`. A run id is `run:thread:` plus the thread id encoded once more,
# then `:ordinal:<n>`. A follow-up run stores its own task id as `payload.delegatedTaskId`.
_DELEGATED = "delegated-task:"
_TASK_PREFIXES = ("node:", "turn-item:")


def _decoded(text: Any) -> str:
    """Percent-decoded until nothing changes (a run id nests the thread id encoded twice).
    (`urllib` is off limits: no module here may import a network library.)"""
    text = str(text or "").strip()
    for _ in range(4):
        decoded = re.sub(r"%([0-9A-Fa-f]{2})", lambda m: chr(int(m[1], 16)), text)
        if decoded == text:
            break
        text = decoded
    return text


def _bare(decoded: str) -> str:
    """A decoded T3 thread or task id without its prefix: `command:mcp:...`, or a top-level thread id."""
    for prefix in ("thread:", *_TASK_PREFIXES, ""):
        if decoded.startswith(prefix + _DELEGATED):
            return decoded[len(prefix + _DELEGATED) :]
    return decoded


def recorded_keys(recorded: Any) -> set[str]:
    """The index keys a recorded `meta.run` names. A prefixed T3 id says what it is: `run:` one
    run; `node:` or `turn-item:` one task (the dispatch that created a thread, or a follow-up);
    `thread:` every run on the thread, for the hold's window to choose from. Any other id may be a
    run, task or thread without its prefix, a Claude agent id (`abc123` or `agent-abc123`), or a
    Codex or native session id (hashed, as the store keeps sessions)."""
    raw = str(recorded or "").strip()
    text = _decoded(raw)
    if not text:
        return set()
    if text.startswith("run:"):
        return {"run:" + text}
    if text.startswith(_TASK_PREFIXES):
        return {"task:" + _bare(text)}
    if text.startswith("thread:"):
        return {"thread:" + _bare(text)}
    bare = _bare(text)
    return {"run:" + text, "task:" + bare, "thread:" + bare, "agent:" + text.removeprefix("agent-"), f"session:{lr.identity(raw)}", f"session:{lr.identity(text)}"}


def _t3_aliases(run_id: Any, thread_id: Any, delegated_task_id: Any, ordinal: Any = 1) -> set[str]:
    thread = _decoded(thread_id)
    aliases = {"run:" + _decoded(run_id), "thread:" + _bare(thread)}
    if delegated_task_id:
        aliases.add("task:" + _bare(_decoded(delegated_task_id)))
    elif thread.startswith("thread:" + _DELEGATED) and ordinal in (1, None):
        # The delegate_task call that created the thread started its first run. A later run with
        # no task id of its own (T3 starts it, not a dispatch) is not named by that call.
        aliases.add("task:" + _bare(thread))
    return aliases


def _run_aliases(row: Mapping[str, Any]) -> set[str]:
    """The keys (`recorded_keys`) under which a dispatch may have recorded this run-level row."""
    aliases: set[str] = set()
    if row.get("source") == "t3-database":
        aliases = _t3_aliases(row.get("run_id"), row.get("thread_id") or row.get("pointers", {}).get("thread_id"), row.get("delegated_task_id"), row.get("ordinal"))
        if row.get("native_session"):
            aliases.add(f"session:{row['native_session']}")
    elif row.get("source") == "claude-subagent":
        stem = Path(str(row.get("pointers", {}).get("transcript") or "")).stem
        aliases.add("agent:" + stem.removeprefix("agent-"))
    elif row.get("source") == "codex-transcript" and row.get("session"):
        aliases.add(f"session:{row['session']}")
    return {a for a in aliases if not a.endswith(":")}


def _runs_named(index: Mapping[str, list[dict[str, Any]]], recorded: str) -> tuple[list[dict[str, Any]], bool]:
    """The distinct run-level rows a recorded id names, and whether it named them as a thread
    (several runs over time, so only those the hold watched are its)."""
    found: dict[int, dict[str, Any]] = {}
    by_thread = False
    for key in recorded_keys(recorded):
        rows = index.get(key, [])
        by_thread |= key.startswith("thread:") and bool(rows)
        for row in rows:
            found[id(row)] = row
    return list(found.values()), by_thread


def _dispatched_by(run: Mapping[str, Any], hold: Mapping[str, Any]) -> bool:
    """A run started within the dispatch window before the hold opened (or after it): the only
    runs a hold registered after its dispatch can have watched."""
    return run["start"] >= hold["start"] - _HOLD_DISPATCH_MAX_S


def _watched(run: Mapping[str, Any], hold: Mapping[str, Any]) -> bool:
    return _overlaps(run, hold) and _dispatched_by(run, hold)


def _span(pair: Sequence[Any]) -> dict[str, Any]:
    """A stopped run kept on a hold as `[start, end]` (end `None` while unknown)."""
    return {"start": pair[0], "end": pair[1]}


def _overlaps(run: Mapping[str, Any], hold: Mapping[str, Any]) -> bool:
    return run["start"] <= hold["end"] and (run["end"] is None or run["end"] > hold["start"])


def _resolve_holds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs = {r["run_id"]: r for r in rows if r.get("source") == "t3-database" and r.get("run_id")}
    subagents = {r["pointers"]["transcript"]: r for r in rows if r.get("source") == "claude-subagent" and r.get("pointers", {}).get("transcript")}
    by_thread: dict[str, list[dict[str, Any]]] = {}
    for run in runs.values():
        by_thread.setdefault(str(run.get("thread_id") or run["pointers"].get("thread_id")), []).append(run)
    by_alias: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("source") in RUN_LEVEL_SOURCES:
            for alias in _run_aliases(row):
                by_alias.setdefault(alias, []).append(row)

    def resolve(hold: Mapping[str, Any]) -> tuple[dict[str, Any] | None, int | None]:
        """The run a hold names and, for an id that names several runs (a thread, a session),
        how many of them fall inside the hold. A recorded id is exact: it is never matched by label or time,
        except to choose among the runs it names."""
        link = hold.get("link") or {}
        if "run" not in link:
            return runs.get(link.get("t3_run_id")) or subagents.get(link.get("subagent")), None
        named, by_thread = _runs_named(by_alias, link["run"])
        # A thread id names every run on its thread, so the hold takes the runs it watched: they
        # overlap it and started no earlier than the dispatch window before it. The rule never
        # depends on how many runs the thread has. Any other id is exact, and only a Codex session
        # (several turns) still has to choose among them.
        if by_thread or len(named) > 1:
            overlapping = [r for r in named if _watched(r, hold)]
        else:
            overlapping = [r for r in named if _overlaps(r, hold)]
        if by_thread:
            # Runs that stopped (cancelled, failed, interrupted) are not stored, but the hold may
            # have watched one: it cannot then say which run its estimate was for.
            stopped = [run for run in link.get("stopped") or [] if _watched(_span(run), hold)]
            if overlapping and stopped:
                return None, len(overlapping) + len(stopped)
        if len(overlapping) == 1:
            return overlapping[0], 1
        if overlapping:
            return None, len(overlapping)
        if len(named) == 1 and not by_thread and named[0]["end"] <= hold["start"]:
            return named[0], 1  # the one run an exact id names was over first: the ended-first guard says so
        return None, None

    linked = [r for r in rows if r.get("source") == "vigil-ledger" and r.get("link_status") == "linked"]
    resolved = {id(hold): resolve(hold) for hold in linked}
    # A hold opened after its run ended watched nothing: it is not a claimant, so it cannot void the hold that did.
    claims = Counter(id(t) for h in linked for t, _ in [resolved[id(h)]] if t is not None and t["end"] > h["start"])
    dropped: set[int] = set()
    for hold in linked:
        target, named = resolved[id(hold)]
        by_id = "run" in (hold.get("link") or {})
        if target is None and named:
            hold.update(link_status="multi-run", linked_runs=named)
            continue
        if target is None:
            # Nothing stored for the target: it was cancelled (the hold has nothing to score), or
            # the link is broken (the dispatch names a run or subagent that is not there).
            hold["link_status"] = "target-cancelled" if (hold.get("link") or {}).get("target") == "cancelled" else "link-failed"
            continue
        if target["end"] <= hold["start"]:
            # Opened after the run had finished: the estimate was made with the answer known.
            hold["link_status"] = "target-ended-first"
            continue
        if claims[id(target)] > 1:
            hold["link_status"] = "shared-target"
            continue
        if "unknown" not in (hold["model"], target["model"]) and hold["model"] != target["model"]:
            hold["link_status"] = "model-mismatch"
            continue
        if not is_sample(target):
            hold["link_status"] = "target-not-agent-run"
            continue
        if target["source"] == "t3-database":
            thread = by_thread.get(str(target.get("thread_id") or target["pointers"].get("thread_id")), [])
            count = 1 if by_id else sum(hold["start"] - _HOLD_DISPATCH_MAX_S <= r["start"] <= hold["end"] for r in thread)
        else:
            count = int(target.get("turns", 1))
        if count != 1:
            hold.update(link_status="multi-run", linked_runs=count)
            continue
        if all(v is None for v in hold["estimate"].values()) and hold.get("ceiling_s") is None:
            # Found its run but carries nothing to score (no quantile, no ceiling).
            hold["link_status"] = "linked-no-estimate"
            continue
        for quantile, value in hold["estimate"].items():
            if value is not None and target["estimate"].get(quantile) is None:
                target["estimate"][quantile] = value
        for field in ("ceiling_s", "estimator"):
            if target.get(field) is None:
                target[field] = hold.get(field)
        target["estimate_source"] = hold["estimate_source"] or target.get("estimate_source")
        target["expect_s"] = target["estimate"].get("p80") or target.get("ceiling_s")
        target["expect_is_prerun"] = hold["start"] <= target["start"] + _HOLD_STAMP_SLACK[1]
        target["estimate_link"] = "vigil-hold"
        target["link_source"] = hold.get("link_source")
        target["pointers"] = {**target.get("pointers", {}), **{k: v for k, v in hold.get("pointers", {}).items() if k != "thread_id"}}
        dropped.add(id(hold))
    return [r for r in rows if id(r) not in dropped]


# A subagent's first record can be written a moment before its parent turn's start is recorded.
_PARENT_SLACK_S = 5.0


def _link_subagents(rows: list[dict[str, Any]]) -> None:
    turns: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("source") == "t3-database" and row.get("native_session"):
            turns.setdefault(row["native_session"], []).append(row)
    for row in rows:
        if row.get("source") != "claude-subagent" or not row.get("parent_session"):
            continue
        around = [t for t in turns.get(row["parent_session"], []) if t["start"] - _PARENT_SLACK_S <= row["start"] <= t["end"] + _PARENT_SLACK_S]
        if not around:
            continue
        parent = max(around, key=lambda t: t["start"])
        row["parent_run_id"] = parent.get("run_id") or parent["id"]
        if is_agent_run(parent):
            row["nested_in"] = row["parent_run_id"]


def is_sample(row: Mapping[str, Any]) -> bool:
    """One duration sample of agent work: a run-level row (T3 run, subagent, Codex turn) or a
    plan task with its own recorded actual, that is an agent run and not nested inside another
    agent run. A vigil hold is never a sample: its lifetime is the orchestrator's wait."""
    run_level = row.get("source") in RUN_LEVEL_SOURCES or (row.get("source") == "plan" and row.get("label_source") == "plan-record")
    return run_level and is_agent_run(row) and not row.get("nested_in")


def reference_rows(state: Path, *, after: float) -> list[dict[str, Any]]:
    """Learned history for the estimator: collapsed, uncensored run-level agent runs (never an
    orchestrator turn, machine job or check-in) finished after `after`. The replayed store already holds everything older, so taking only the tail never
    double-counts it."""
    return [
        r
        for r in collapse(read_store(state))
        if r["end"] > after and r.get("censor_reason", "none") == "none" and r["source"] in RUN_LEVEL_SOURCES and is_sample(r)
    ]


# --- readers -------------------------------------------------------------------------------
# Each reader takes (location, its cursor entry or None, now) and returns (observations, status
# counters, new cursor entry). A reader raises only for a source that is configured but
# unreadable; one bad file or row is skipped and counted.


def _kind_source(kind: str, found: str) -> str:
    return "none" if kind == "other" else found


def read_t3(db_path: Path, since: float | None, now: float) -> tuple[list[dict[str, Any]], dict[str, int], float | None]:
    """T3 runs finished since the cursor (less an overlap), through the shared readers:
    `t3_history.read_database` for the row and its estimator features, `usage_gap.load_t3_runs`
    for usage-limit endings and the native session. Both filter by the cursor in SQL."""
    since_s = max(0.0, since - DATABASE_OVERLAP_S) if since is not None else None
    stats: dict[str, int] = {}
    runs = t3_history.read_database(db_path, since=since_s, stats=stats)
    limits = {r.run_id: r for r in ug.load_t3_runs(db_path, since=since_s)}
    counters = {"seen": 0, "added": 0, "skipped_censored": 0, "skipped_invalid": stats.get("skipped_invalid", 0)}
    rows: list[dict[str, Any]] = []
    cursor: float | None = None
    for run in runs:
        if run["end"] is None or (since_s is not None and run["end"] < since_s):
            continue  # in flight, or context read only for the concurrency feature
        counters["seen"] += 1
        cursor = run["end"] if cursor is None else max(cursor, run["end"])
        limit = limits.get(run["run_id"])
        ended_on_limit = bool(limit and limit.usage_limit_end)
        if run["status"] != "completed" and not ended_on_limit:
            counters["skipped_censored"] += 1
            continue
        rows.append(
            extend(
                run,
                key=("t3", run["run_id"]),
                model_raw=run["model_raw"],
                kind_source=_kind_source(run["kind"], "brief-keyword"),
                label_source="t3-database",
                native_session=lr.identity(limit.session_id) if limit and limit.session_id else None,
                usage_limit=ended_on_limit,
                role=None if run["delegated"] else "orchestrator-turn",
                pointers={"thread_id": run["thread_id"], "run_id": run["run_id"]},
                delegated=run["delegated"],
                ingested_at=now,
            )
        )
    return rows, counters, cursor


# A check-in timer or an owner decision is a wait on the owner, never an agent run.
_CHECKIN_LABEL = re.compile(r"(^|[-_.])(check-?in\d*|owner)([-_.]|$)")
# A hold registered more than this after its dispatch is not that dispatch's hold.
_HOLD_DISPATCH_MAX_S = 900.0
# vigil stamps the hold when it registers it, just after the tool call that asked for it.
_HOLD_STAMP_SLACK = (-5.0, 60.0)


def _meta_seconds(value: Any) -> float | None:
    """Seconds from a vigil `--meta` value (always text): positive and finite, else None."""
    try:
        return _seconds(float(str(value).strip()))
    except ValueError:
        return None


def _text(value: Any) -> str | None:
    """A short label from meta (`agentpace-0.13.1`), else None."""
    text = str(value or "").strip()
    return text[:64] if text else None


def hold_role(label: str, model_raw: str) -> str | None:
    """`checkin` for a check-in or owner wait, `machine` for a hold that names no model (a suite,
    an emulator job, a remote build), else None (an agent hold). Decided from the label and meta, never
    from a link: hold_index binds every hold to the last dispatch before it."""
    if _CHECKIN_LABEL.search(label.lower()):
        return "checkin"
    return None if model_raw else "machine"


class _Dispatches:
    """Links a vigil item to the run its hold watched: the hold registration in the orchestrator's
    transcript (`t3_history.hold_index`), the dispatch before it, and the run or subagent that
    dispatch's result names (`t3_history.call_index`). Each transcript is indexed once."""

    def __init__(self, projects: Path | None):
        self.projects = Path(projects) if projects else None
        self.cache: dict[str, Any] = {}

    def _indexed(self, session: str) -> Any:
        if session not in self.cache:
            found = sorted(self.projects.glob(f"*/{session}.jsonl")) if self.projects and re.fullmatch(r"[\w.-]+", session) else []
            entry = None
            if found:
                try:
                    entry = (found[0], t3_history.hold_index(found, agents=True), t3_history.call_index(found))
                except (OSError, ValueError, TypeError, KeyError, AttributeError):
                    entry = None
            self.cache[session] = entry
        return self.cache[session]

    def link(self, session: str, label: str, start: float) -> tuple[str, dict[str, str] | None]:
        entry = self._indexed(session)
        if entry is None:
            return "no-transcript", None
        path, holds, calls = entry
        low, high = _HOLD_STAMP_SLACK
        found = [(key, call) for key, call in holds.items() if key[2] == label and low <= start - key[1] <= high]
        if not found:
            return "no-hold-record", None
        if len(found) > 1:
            return "ambiguous-hold-record", None
        (key, call), = found
        if key[1] - call[1] > _HOLD_DISPATCH_MAX_S:
            return "stale-dispatch", None
        ids = calls.get(call, [])
        if len(ids) != 1:
            return "no-dispatch-result", None
        if ids[0].get("runId"):
            return "linked", {"t3_run_id": str(ids[0]["runId"])}
        return "linked", {"subagent": str(path.with_suffix("") / "subagents" / f"agent-{ids[0]['agentId']}.jsonl")}


def _subagent_cancelled(path: Path) -> bool:
    """A subagent transcript that never ended a turn (stopped, or still running)."""
    return not any(t.get("censor_reason", "none") in ("none", "usage-limit") for t in run_labels.claude_turns(path))


_STOPPED = ("cancelled", "canceled", "failed", "interrupted")


def _cancelled_targets(rows: Sequence[Mapping[str, Any]], t3_db: Path | None, transcripts: Path | None = None) -> None:
    """Read the T3 runs that stopped without finishing (cancelled, failed, interrupted) and the
    Claude subagents that never ended a turn; neither is stored as an observation. On a linked hold:
    `link["target"] = "cancelled"` when its run was cancelled, so a hold that later finds no stored
    target can say why; and, for a hold named by a thread id, `link["stopped"]` as `[start, end]`
    pairs of every stopped run on that thread, so a stopped run beside a stored one makes the hold
    `multi-run` rather than a score for the wrong run. A recorded id (`link["run"]`) is matched as
    `collapse` matches it: a run named as a thread counts only when it overlaps the hold; a Claude
    agent id finds its transcript under `transcripts`. Both sources are only read."""
    wanted: dict[str, list[Mapping[str, Any]]] = {}
    named: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for row in rows:
        link = row.get("link") or {}
        try:
            if link.get("run"):
                named.append((row, link))
                agent = _decoded(link["run"]).removeprefix("agent-")
                if transcripts and re.fullmatch(r"[\w-]+", agent):
                    found = sorted(Path(transcripts).glob(f"*/*/subagents/agent-{agent}.jsonl"))
                    if len(found) == 1 and _subagent_cancelled(found[0]):
                        link["target"] = "cancelled"
            elif link.get("t3_run_id"):
                wanted.setdefault(str(link["t3_run_id"]), []).append(link)
            elif link.get("subagent") and Path(link["subagent"]).is_file() and _subagent_cancelled(Path(link["subagent"])):
                link["target"] = "cancelled"
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    if not (wanted or named) or t3_db is None or not Path(t3_db).exists():
        return
    try:
        connection = sqlite3.connect(Path(t3_db).resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
        try:
            connection.execute("PRAGMA query_only=ON")
            stopped = [
                tuple(record)
                for record in connection.execute(
                    "SELECT run_id, thread_id, ordinal, status, requested_at, completed_at, payload_json FROM orchestration_v2_projection_runs"
                    " WHERE lower(status) IN (" + ", ".join("?" * len(_STOPPED)) + ")",
                    _STOPPED,
                )
            ]
        finally:
            connection.close()
    except sqlite3.Error:
        return
    by_key: dict[str, list[dict[str, Any]]] = {}
    for run_id, thread_id, ordinal, status, requested, completed, payload in stopped:
        cancelled = str(status).lower() in ("cancelled", "canceled")
        if cancelled:
            for link in wanted.get(str(run_id), []):
                link["target"] = "cancelled"
        try:
            body = json.loads(payload) if payload else {}
            body = body if isinstance(body, dict) else {}
            start = lr.epoch(body.get("startedAt") or requested)
            finish = body.get("completedAt") or completed
            run = {"start": start, "end": lr.epoch(finish) if finish else None, "cancelled": cancelled}
        except (ValueError, TypeError, AttributeError):
            continue
        for key in _t3_aliases(run_id, thread_id, body.get("delegatedTaskId"), ordinal):
            by_key.setdefault(key, []).append(run)
    for row, link in named:
        spans: dict[int, dict[str, Any]] = {}
        for key in recorded_keys(link["run"]):
            for run in by_key.get(key, []):
                if run["cancelled"] and (not key.startswith("thread:") or _overlaps(run, row)):
                    link["target"] = "cancelled"
                if key.startswith("thread:"):
                    spans[id(run)] = run
        if spans:
            link["stopped"] = [[run["start"], run["end"]] for run in spans.values()]


def read_vigil(
    ledger: Path, since: float | None, now: float, transcripts: Path | None = None, t3_db: Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, int], float | None]:
    """Done vigil items through the shared `local_runs.ledger_row` (brief features, machine
    time), with label provenance, a role (check-in, machine job) and the link to the run the hold
    watched. Linking needs the orchestrator transcripts (`transcripts`, the Claude projects
    folder); `collapse` decides whether the estimate scores that run."""
    since_s = since - LEDGER_OVERLAP_S if since is not None else None
    counters = {"seen": 0, "added": 0, "skipped": 0, "linked": 0}
    rows: list[dict[str, Any]] = []
    cursor: float | None = None
    dispatches = _Dispatches(transcripts)
    for line in Path(ledger).read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
            row = lr.ledger_row(record)
            hold, label = str(record["hold"]), str(record["label"])
        except (ValueError, KeyError, TypeError, AttributeError, OSError):
            counters["skipped"] += 1
            continue
        cursor = row["end"] if cursor is None else max(cursor, row["end"])
        if since_s is not None and row["end"] < since_s:
            continue
        counters["seen"] += 1
        meta = record.get("meta") or {}
        kind_source = "vigil-meta" if meta.get("kind") else _kind_source(row["kind"], "label-keyword")
        model_raw = str(meta.get("model") or "")
        model_source = "vigil-meta" if model_raw else "label-keyword"
        if not model_raw:
            model_raw = lr.model_word(label) or ""
            model_source = "label-keyword" if model_raw else "none"
            row["model"] = lr.model_name(model_raw)
        recorded = str(meta.get("run") or "").strip()
        # The label only guesses a check-in or machine job; a recorded run id says what the hold watched.
        role = None if recorded else hold_role(label, model_raw)
        if recorded:
            # The orchestrator wrote the run's id when it dispatched: exact, so no transcript, label or time is consulted.
            status, link, link_source = "linked", {"run": recorded}, "meta-run"
        elif role:
            status, link, link_source = "not-an-agent-run", None, None
        else:
            status, link = dispatches.link(str(record.get("session_id") or ""), label, row["start"])
            link_source = "transcript" if status == "linked" else None
        counters["linked"] += status == "linked"
        quantiles = {q: _meta_seconds(meta.get(q)) for q in ("p50", "p80")}
        estimated = any(v is not None for v in quantiles.values())
        # A hold with the estimator's own quantiles was estimated; one with only `--expect-s` has a
        # hand-set ceiling, which is reported apart and never scored as a P80.
        ceiling = None if estimated else _seconds(record.get("expect_s"))
        observed = extend(
            row,
            key=("vigil", hold, label, row["start"]),
            model_raw=model_raw,
            model_source=model_source,
            kind_source=kind_source,
            label_source="vigil-hold",
            execution_key=vigil_execution_key(hold, label, row["start"]),
            estimate=quantiles,
            estimate_source="agentpace" if estimated else "ceiling",
            ceiling_s=ceiling,
            estimator=_text(meta.get("est")) if estimated else None,
            role=role,
            pointers={"vigil_hold": hold, "vigil_label": label, "plan_task": meta.get("task")},
            ingested_at=now,
        )
        observed.update(link=link, link_status=status, link_source=link_source)
        rows.append(observed)
    _cancelled_targets(rows, t3_db, transcripts)
    return rows, counters, cursor


def _scan(paths_: Iterable[Path], seen: Mapping[str, Any] | None) -> Iterator[tuple[Path, list[int], bool]]:
    """(path, signature, changed) for each file. The signature is (mtime_ns, size); a file is
    changed when it differs from the signature recorded after its last successful parse, so a
    file written in the same clock tick as another, or after a failed parse, is still read."""
    for path in sorted(paths_):
        try:
            stat = path.stat()
        except OSError:
            continue
        signature = [stat.st_mtime_ns, stat.st_size]
        yield path, signature, (seen or {}).get(str(path)) != signature


def _subagent_observation(path: Path, turns: Sequence[Mapping[str, Any]], now: float) -> dict[str, Any] | None:
    """One observation per subagent transcript, updated when the subagent is resumed: start of
    the first ended turn, end of the last, and the agent work in between (the sum of the turns;
    the time between turns was the parent's). A turn ends by completing or on a usage limit;
    API-error stubs and in-flight turns are not work. None until one turn has ended."""
    ended = [t for t in turns if t.get("censor_reason", "none") in ("none", "usage-limit")]
    if not ended:
        return None
    named = next((t for t in ended if t.get("model_raw")), ended[0])
    spans = [[float(t["start"]), float(t["end"])] for t in ended]
    base = dict(ended[0], start=spans[0][0], end=spans[-1][1], actual_s=sum(end - start for start, end in spans), model=named["model"])
    row = extend(
        base,
        key=("claude", str(path)),
        source="claude-subagent",
        model_raw=str(named.get("model_raw") or ""),
        kind_source=_kind_source(base["kind"], "brief-keyword"),
        label_source="transcript",
        provider="claudeAgent",
        # Ended on a usage limit, as usage_gap marks it: a provider wait, not an estimate error.
        usage_limit=ended[-1].get("censor_reason") == "usage-limit",
        pointers={"transcript": str(path)},
        ingested_at=now,
    )
    row.update(turns=len(spans), spans=spans, wall_s=spans[-1][1] - spans[0][0], parent_session=lr.identity(path.parent.parent.name))
    return row


def read_claude(projects: Path, seen: Mapping[str, Any] | None, now: float) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    """Claude subagent transcripts through the shared `run_labels.claude_turns`."""
    counters = {"seen": 0, "added": 0, "skipped_files": 0, "skipped_incomplete": 0}
    rows: list[dict[str, Any]] = []
    files: dict[str, Any] = {}
    for path, signature, changed in _scan(Path(projects).glob("*/*/subagents/*.jsonl"), seen):
        if not changed:
            files[str(path)] = signature
            continue
        counters["seen"] += 1
        try:
            row = _subagent_observation(path, run_labels.claude_turns(path), now)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            counters["skipped_files"] += 1
            continue
        files[str(path)] = signature
        if row is None:
            counters["skipped_incomplete"] += 1
        else:
            rows.append(row)
    return rows, counters, files


def read_codex(sessions: Path, seen: Mapping[str, Any] | None, now: float) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    """Codex turns through the shared `run_labels.codex_turns`. In-flight and errored turns are
    skipped; a turn that ended on a usage limit is kept as usage-wait."""
    counters = {"seen": 0, "added": 0, "skipped_files": 0, "skipped_censored": 0, "skipped_incomplete": 0}
    rows: list[dict[str, Any]] = []
    files: dict[str, Any] = {}
    for path, signature, changed in _scan(Path(sessions).rglob("rollout-*.jsonl"), seen):
        if not changed:
            files[str(path)] = signature
            continue
        counters["seen"] += 1
        try:
            turns = run_labels.codex_turns(path)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            counters["skipped_files"] += 1
            continue
        files[str(path)] = signature
        for turn in turns:
            reason = turn.get("censor_reason", "none")
            if reason == "incomplete":
                counters["skipped_incomplete"] += 1
                continue
            if reason not in ("none", "usage-limit"):
                counters["skipped_censored"] += 1
                continue
            rows.append(
                extend(
                    turn,
                    key=("codex", str(path), turn.get("turn_id"), turn["start"]),
                    source="codex-transcript",
                    model_raw=str(turn.get("model_raw") or ""),
                    kind_source=_kind_source(turn["kind"], "brief-keyword"),
                    label_source="transcript",
                    provider="codex",
                    usage_limit=reason == "usage-limit",
                    pointers={"transcript": str(path), "turn_id": turn.get("turn_id")},
                    ingested_at=now,
                )
            )
    return rows, counters, files


def read_plans(plan_dirs: Sequence[Path], seen: Mapping[str, Any] | None, now: float) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    """Done agent tasks of plan files. A file is recorded in the cursor only once it parsed."""
    counters = {"seen": 0, "added": 0, "skipped_files": 0, "skipped": 0}
    rows: list[dict[str, Any]] = []
    files: dict[str, Any] = {}
    paths_ = [p for directory in plan_dirs if Path(directory).is_dir() for p in Path(directory).glob("*.json")]
    for path, signature, changed in _scan(paths_, seen):
        if not changed:
            files[str(path)] = signature
            continue
        try:
            plan = json.loads(path.read_text())
            tasks = plan["tasks"]
            if not isinstance(tasks, list):
                raise TypeError("tasks is not a list")
        except (OSError, ValueError, KeyError, TypeError):
            counters["skipped_files"] += 1
            continue
        files[str(path)] = signature
        counters["seen"] += 1
        links = {e.get("task"): e.get("ledger_id") for e in plan.get("events") or [] if isinstance(e, dict) and e.get("action") == "ledger-done"}
        for task in tasks:
            if not isinstance(task, dict) or task.get("type") != "agent" or task.get("status") != "done":
                continue
            try:
                start = lr.epoch(task["started_at"])
                end = lr.epoch(task.get("ended_at") or start + float(task["actual_s"]))
                rows.append(
                    observation(
                        source="plan",
                        key=("plan", str(path), task["id"], start),
                        start=start,
                        end=end,
                        model_raw=str(task.get("model") or ""),
                        kind=str(task.get("kind") or "other"),
                        model_source="plan" if task.get("model") else "none",
                        kind_source="plan" if task.get("kind") else "none",
                        label_source="vigil-hold" if task.get("actual_source") == "vigil-ledger" else "plan-record",
                        project=str(plan.get("cwd") or ""),
                        session=str(plan.get("id") or path.stem),
                        execution_key=links.get(task["id"]),
                        estimate={"p50": task.get("forecast_s")},
                        estimate_source="plan-forecast",
                        pointers={"plan": str(path), "task": task["id"]},
                        ingested_at=now,
                    )
                )
            except (ValueError, KeyError, TypeError, AttributeError):
                counters["skipped"] += 1
    return rows, counters, files


# --- run -----------------------------------------------------------------------------------


def _readers(sources: Sources) -> list[tuple[str, Any, Any]]:
    """(name, configured location, reader taking (location, since, now))."""
    return [
        ("t3-database", sources.t3_db, read_t3),
        ("vigil-ledger", sources.vigil_ledger, functools.partial(read_vigil, transcripts=sources.claude_projects, t3_db=sources.t3_db)),
        ("claude-subagent", sources.claude_projects, read_claude),
        ("codex-transcript", sources.codex_sessions, read_codex),
        ("plan", tuple(sources.plan_dirs) or None, read_plans),
    ]


def _is_present(location: Any) -> bool:
    if location is None:
        return False
    if isinstance(location, tuple):
        return any(Path(p).exists() for p in location)
    return Path(location).exists()


def run(state: Path, sources: Sources, now: float | None = None) -> dict[str, Any]:
    """Ingest runs finished since each source's cursor. Returns counts only.

    CLAIM(learn-crash-safe): rows are synced to the store before the cursor is replaced, and a
    row already stored with the same content is never added again, so a crash at any point loses
    and doubles nothing [tests/test_learn.py:test_crash_before_the_cursor_moves_loses_and_doubles_nothing].
    """
    state = Path(state)
    _make_state_dir(state)
    now = time.time() if now is None else float(now)
    with (state / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        migrated = _migrate(state)
        current = {r["id"]: r for r in read_store(state)}
        # A cursor with no rows behind it (the store was lost) would skip everything it covers.
        cursor = read_cursor(state) if current else {}
        new_cursor = dict(cursor)
        gathered: list[dict[str, Any]] = []
        summary: dict[str, Any] = {"state": str(state), "sources": {}}
        read_ok: list[str] = []
        for name, location, reader in _readers(sources):
            if not _is_present(location):
                summary["sources"][name] = {"status": "absent"}
                continue
            try:
                rows, counters, latest = reader(location, cursor.get(name), now)
            except Exception as error:  # noqa: BLE001 - one failing source must not cost the others their rows
                summary["sources"][name] = {"status": f"error: {type(error).__name__}"}
                continue
            gathered.extend(rows)
            read_ok.append(name)
            summary["sources"][name] = {"status": "ok", **counters}
            if name in FILE_SOURCES:
                new_cursor[name] = latest
            elif latest is not None:
                new_cursor[name] = max(latest, cursor.get(name, latest))
        before = set(current)
        # CLAIM(learn-crash-safe): rows first, cursor second.
        appended = append_observations(state, gathered, current)
        for name in read_ok:
            mine = [r for r in appended if r["source"] == name]
            summary["sources"][name]["added"] = sum(r["id"] not in before for r in mine)
            summary["sources"][name]["updated"] = sum(r["id"] in before for r in mine)
        if new_cursor != cursor or migrated or not _cursor_path(state).exists():
            _write_cursor(state, new_cursor)  # ↳ CLAIM(learn-crash-safe)
        summary["added"] = sum(r["id"] not in before for r in appended)
        summary["updated"] = sum(r["id"] in before for r in appended)
        summary["total"] = len(current)
        summary["cursor"] = {k: (v if k in TIME_SOURCES else {"files": len(v)}) for k, v in new_cursor.items()}
        if migrated is not None:
            summary["migrated_from_schema"] = migrated
    return summary


# --- CLI -----------------------------------------------------------------------------------


def cmd_learn(args: Any) -> int:
    defaults = default_sources()
    sources = Sources(
        t3_db=Path(args.t3_db) if args.t3_db else defaults.t3_db,
        vigil_ledger=Path(args.vigil_ledger) if args.vigil_ledger else defaults.vigil_ledger,
        claude_projects=Path(args.claude_projects) if args.claude_projects else defaults.claude_projects,
        codex_sessions=Path(args.codex_sessions) if args.codex_sessions else defaults.codex_sessions,
        plan_dirs=tuple(Path(p) for p in args.plans) if args.plans else defaults.plan_dirs,
    )
    summary = run(Path(args.state) if args.state else paths.state_dir(), sources)
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0
    print(f"learn: {summary['added']} new observations, {summary['total']} in {summary['state']}")
    for name, info in summary["sources"].items():
        detail = ", ".join(f"{k} {v}" for k, v in info.items() if k != "status")
        print(f"  {name}: {info['status']}" + (f" ({detail})" if detail else ""))
    return 0


def add_command(commands: Any) -> None:
    parser = commands.add_parser("learn", help="ingest runs finished since the last cursor into the local observation store")
    parser.add_argument("--state", help="state directory (default: AGENTPACE_STATE or the XDG state dir)")
    parser.add_argument("--t3-db", help="T3 SQLite database (default: AGENTPACE_T3_DB)")
    parser.add_argument("--vigil-ledger", help="vigil items.jsonl (default: AGENTPACE_VIGIL_LEDGER)")
    parser.add_argument("--claude-projects", help="Claude projects folder (default: AGENTPACE_CLAUDE_PROJECTS)")
    parser.add_argument("--codex-sessions", help="Codex sessions folder (default: AGENTPACE_CODEX_SESSIONS)")
    parser.add_argument("--plans", action="append", default=[], help="plan folder, repeatable (default: AGENTPACE_PLAN_DIRS)")
    parser.add_argument("--json", action="store_true", help="print the counts as JSON")
    parser.set_defaults(handler=cmd_learn)
