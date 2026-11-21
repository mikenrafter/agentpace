"""SYNTHETIC sources for the learn/calibration tests. Nothing here came from a real run.

Builds a fake T3 database, vigil ledger, Claude subagent transcript, Codex rollout and plan file.
Briefs carry CANARY_BRIEF so a test can prove no prompt text reaches the learned store.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

CANARY = "CANARY_BRIEF_TEXT"
T0 = 1_800_000_000.0  # fixed epoch for every fixture (2027-01-15T08:00:00Z)


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def make_t3_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE orchestration_v2_projection_runs(
            run_id TEXT, thread_id TEXT, ordinal INTEGER, provider TEXT,
            status TEXT, requested_at TEXT, completed_at TEXT, payload_json TEXT);
        CREATE TABLE orchestration_v2_projection_messages(message_id TEXT, payload_json TEXT);
        CREATE TABLE orchestration_v2_projection_threads(thread_id TEXT, project_id TEXT);
        CREATE TABLE projection_projects(project_id TEXT, workspace_root TEXT);
        CREATE TABLE orchestration_v2_projection_provider_threads(provider_thread_id TEXT, payload_json TEXT);
        CREATE TABLE orchestration_v2_projection_run_attempts(payload_json TEXT);
        CREATE TABLE orchestration_v2_projection_turn_items(
            run_id TEXT, type TEXT, status TEXT, payload_json TEXT);
        INSERT INTO projection_projects VALUES('p1', '/work/CANARY_PATH');
        """
    )
    connection.commit()
    connection.close()


def add_t3_run(
    path: Path,
    run_id: str,
    *,
    start: float,
    end: float | None,
    provider: str = "claudeAgent",
    model: str = "claude-opus-5-5",
    status: str = "completed",
    brief: str = "implement the thing",
    thread_id: str | None = None,
    native_id: str | None = None,
    usage_limit: bool = False,
    effort: str | None = None,
    ordinal: int = 1,
    requested: float | None = None,
) -> None:
    thread_id = thread_id or f"thread:delegated-task:{run_id}"  # a delegated agent run unless told otherwise
    connection = sqlite3.connect(path)
    payload = {
        "startedAt": iso(start),
        "userMessageId": f"m-{run_id}",
        "modelSelection": {"model": model, **({"options": [{"id": "reasoningEffort", "value": effort}]} if effort else {})},
        "providerThreadId": f"pt-{run_id}",
    }
    if end is not None:
        payload["completedAt"] = iso(end)
    connection.execute(
        "INSERT INTO orchestration_v2_projection_runs VALUES(?,?,?,?,?,?,?,?)",
        (run_id, thread_id, ordinal, provider, status, iso(requested if requested is not None else start), iso(end) if end else None, json.dumps(payload)),
    )
    connection.execute(
        "INSERT INTO orchestration_v2_projection_messages VALUES(?,?)",
        (f"m-{run_id}", json.dumps({"text": f"{brief} {CANARY}"})),
    )
    if connection.execute("SELECT count(*) FROM orchestration_v2_projection_threads WHERE thread_id=?", (thread_id,)).fetchone()[0] == 0:
        connection.execute("INSERT INTO orchestration_v2_projection_threads VALUES(?, 'p1')", (thread_id,))
    if native_id:
        connection.execute(
            "INSERT INTO orchestration_v2_projection_provider_threads VALUES(?,?)",
            (f"pt-{run_id}", json.dumps({"nativeThreadRef": {"nativeId": native_id}})),
        )
    if usage_limit:
        connection.execute(
            "INSERT INTO orchestration_v2_projection_turn_items VALUES(?,?,?,?)",
            (run_id, "error", "failed", json.dumps({"failure": {"class": "usage_limit"}})),
        )
    connection.commit()
    connection.close()


def vigil_line(
    hold: str,
    label: str,
    *,
    start: float,
    end: float,
    expect_s: float | None = None,
    meta: dict | None = None,
    outcome: str = "done",
) -> str:
    record = {
        "cwd": "/work/CANARY_PATH",
        "ended_at": iso(end),
        "expect_s": expect_s,
        "hold": hold,
        "label": label,
        "meta": meta or {},
        "outcome": outcome,
        "session_id": "s-orchestrator",
        "started_at": iso(start),
        "v": 1,
    }
    return json.dumps(record, sort_keys=True) + "\n"


def claude_turn(start: float, end: float | None, *, model: str = "claude-sonnet-5-5", brief: str = "review the diff", ending: str = "end_turn", meta: str | None = None) -> list[dict]:
    """One subagent turn: the prompt, then an assistant record. `ending` is end_turn, synthetic
    (an API error stub), usage-limit (a synthetic usage-limit stub) or none (still running).
    `meta` makes the prompt a harness-injected record (`isMeta`, as a resumed subagent receives
    it): `coordinator` (a message from the parent) or `task-notification` (a system notification)."""
    records = [{"type": "user", "timestamp": iso(start), "cwd": "/work/CANARY_PATH", "message": {"content": f"{brief} {CANARY}"}}]
    if meta:
        records[0].update(isMeta=True, origin={"kind": meta})
    if end is None:
        return records
    if ending == "end_turn":
        records.append({"type": "assistant", "timestamp": iso(end), "message": {"model": model, "stop_reason": "end_turn", "content": []}})
    elif ending in ("synthetic", "usage-limit"):
        text = "You've hit your usage limit · resets 3pm" if ending == "usage-limit" else "API Error: 500 internal"
        records.append({"type": "assistant", "timestamp": iso(end), "isApiErrorMessage": True, "message": {"model": "<synthetic>", "stop_reason": "stop_sequence", "content": [{"type": "text", "text": text}]}})
    elif ending == "none":
        records.append({"type": "assistant", "timestamp": iso(end), "message": {"model": model, "stop_reason": "tool_use", "content": []}})
    return records


def write_claude_subagent(
    directory: Path,
    name: str,
    *,
    start: float,
    end: float,
    model: str = "claude-sonnet-5-5",
    session: str = "session",
    turns: list[list[dict]] | None = None,
) -> Path:
    """A subagent transcript under `<directory>/proj/<session>/subagents/`. `turns` (from
    claude_turn) replaces the default single completed turn."""
    path = directory / "proj" / session / "subagents" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [r for turn in (turns or [claude_turn(start, end, model=model)]) for r in turn]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def write_codex_rollout(
    directory: Path,
    name: str,
    *,
    session_id: str,
    start: float,
    end: float,
    model: str = "gpt-6.1-sol",
    usage_limit: bool = False,
    brief: str = "fix the bug",
    brief_as_response_item: bool = False,
) -> Path:
    path = directory / f"rollout-{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    final = "You've hit your usage limit" if usage_limit else "done"
    records = [
        {"type": "session_meta", "timestamp": iso(start), "payload": {"id": session_id, "cwd": "/work/CANARY_PATH"}},
        {"type": "turn_context", "timestamp": iso(start), "payload": {"model": model, "effort": "high"}},
        {"type": "event_msg", "timestamp": iso(start), "payload": {"type": "task_started", "turn_id": "turn-1", "started_at": iso(start)}},
        {"type": "response_item", "timestamp": iso(start), "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": f"{brief} {CANARY}"}]}}
        if brief_as_response_item
        else {"type": "event_msg", "timestamp": iso(start), "payload": {"type": "user_message", "message": f"{brief} {CANARY}"}},
        {"type": "event_msg", "timestamp": iso(end), "payload": {"type": "task_complete", "turn_id": "turn-1", "completed_at": iso(end), "last_agent_message": final}},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def write_plan(directory: Path, name: str, tasks: list[dict], events: list[dict] | None = None) -> Path:
    path = directory / f"{name}.json"
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"id": name, "cwd": "/work/CANARY_PATH", "tasks": tasks, "events": events or [], "forecasts": [], "outcomes": []}))
    return path


def write_orchestrator_transcript(directory: Path, session: str, records: list[dict]) -> Path:
    """The orchestrator's own Claude transcript `<directory>/proj/<session>.jsonl` (where its
    dispatch calls and vigil hold registrations live)."""
    path = directory / "proj" / f"{session}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def tool_use(at: float, call_id: str, name: str, payload: dict) -> dict:
    return {"type": "assistant", "timestamp": iso(at), "message": {"content": [{"type": "tool_use", "id": call_id, "name": name, "input": payload}]}}


def tool_result(at: float, call_id: str, text: str, result: dict | None = None) -> dict:
    record = {"type": "user", "timestamp": iso(at), "message": {"content": [{"type": "tool_result", "tool_use_id": call_id, "content": text}]}}
    if result is not None:
        record["toolUseResult"] = result
    return record


def delegate(at: float, call_id: str, run_id: str, thread_id: str) -> list[dict]:
    """A delegate_task call and its result naming the child run."""
    return [
        tool_use(at, call_id, "mcp__t3-code__delegate_task", {"task": f"do it {CANARY}"}),
        tool_result(at + 1, call_id, json.dumps({"childRunId": run_id, "childThreadId": thread_id})),
    ]


def bash_hold(at: float, label: str, expect_s: int, call_id: str = "h") -> dict:
    return tool_use(at, f"{call_id}-{label}", "Bash", {"command": f"vigil hold add W --for none --expect-s {expect_s} --label {label}"})
