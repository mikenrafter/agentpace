"""agentpace learn: incremental, idempotent, crash-safe ingest into a local state store.

SYNTHETIC fixtures only (tests/learn_fixtures.py). Each test names the bug it guards.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agentpace import calibration, cli, learn, paths
from tests import learn_fixtures as fx

HOUR = 3600.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LearnCase(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.state = self.root / "state"
        self.db = self.root / "statev2.sqlite"
        self.ledger = self.root / "items.jsonl"
        self.claude = self.root / "claude-projects"
        self.codex = self.root / "codex-sessions"
        self.plans = self.root / "plans"
        fx.make_t3_db(self.db)

    def sources(self, **override):
        base = dict(t3_db=self.db, vigil_ledger=self.ledger, claude_projects=self.claude, codex_sessions=self.codex, plan_dirs=(self.plans,))
        base.update(override)
        return learn.Sources(**base)

    def learn(self, **override):
        return learn.run(self.state, self.sources(**override), now=fx.T0 + 10 * HOUR)

    def store(self):
        return learn.read_store(self.state)


class T3Source(LearnCase):
    def test_completed_run_becomes_an_observation_without_prompt_text(self):
        # why: the store feeds estimates and a weekly brief; it must hold derived facts and
        #      pointers, never prompt text (privacy invariant of local_runs).
        # catches: persisting the brief, or dropping model/kind/label provenance.
        fx.add_t3_run(self.db, "r1", start=fx.T0, end=fx.T0 + 600, brief="implement the parser", native_id="nat-1")
        summary = self.learn()
        (row,) = self.store()
        self.assertEqual(summary["sources"]["t3-database"]["added"], 1)
        self.assertEqual((row["model"], row["kind"], row["kind_source"]), ("opus", "implementation", "brief-keyword"))
        self.assertEqual((row["actual_s"], row["label_source"], row["category"]), (600.0, "t3-database", "ordinary"))
        self.assertIsNone(row["estimate"]["p80"])
        self.assertEqual(row["pointers"]["thread_id"], "thread:delegated-task:r1")
        self.assertEqual(row["pointers"]["run_id"], "r1")
        stored = (self.state / "observations.jsonl").read_text()
        self.assertNotIn("CANARY", stored)
        self.assertNotIn("/work", stored)

    def test_antigravity_neon_run_is_ingested_under_its_model(self):
        # why: new models (Gemini 4 Neon via the antigravity provider) have no history and must
        #      show up so the digest can flag them as thin.
        # catches: filtering T3 runs to the claude/codex providers.
        fx.add_t3_run(self.db, "g1", start=fx.T0, end=fx.T0 + 300, provider="antigravity", model="gemini-4-neon")
        self.learn()
        (row,) = self.store()
        self.assertEqual((row["model"], row["provider"], row["model_raw"]), ("neon", "antigravity", "gemini-4-neon"))

    def test_run_in_flight_is_picked_up_after_it_finishes(self):
        # why: the cursor must never pass a run that had not finished (end is NULL), or the
        #      longest runs would be the ones never learned.
        # catches: advancing the cursor on started_at, or ingesting a null end as zero.
        fx.add_t3_run(self.db, "done", start=fx.T0, end=fx.T0 + 100)
        fx.add_t3_run(self.db, "slow", start=fx.T0 + 50, end=None, status="running")
        self.learn()
        self.assertEqual([r["pointers"]["run_id"] for r in self.store()], ["done"])
        import sqlite3

        connection = sqlite3.connect(self.db)
        payload = json.loads(connection.execute("SELECT payload_json FROM orchestration_v2_projection_runs WHERE run_id='slow'").fetchone()[0])
        payload["completedAt"] = fx.iso(fx.T0 + 5000)
        connection.execute(
            "UPDATE orchestration_v2_projection_runs SET status='completed', completed_at=?, payload_json=? WHERE run_id='slow'",
            (fx.iso(fx.T0 + 5000), json.dumps(payload)),
        )
        connection.commit()
        connection.close()
        summary = self.learn()
        self.assertEqual(summary["sources"]["t3-database"]["added"], 1)
        self.assertEqual(sorted(r["pointers"]["run_id"] for r in self.store()), ["done", "slow"])

    def test_failed_run_is_skipped_unless_it_ended_on_a_usage_limit(self):
        # why: a failed run is censored, not a duration; one that ended on a usage limit is the
        #      usage-wait category the brief wants kept apart.
        # catches: ingesting every failure as a completed run, or dropping usage-limit endings.
        fx.add_t3_run(self.db, "crash", start=fx.T0, end=fx.T0 + 50, status="failed")
        fx.add_t3_run(self.db, "limit", start=fx.T0, end=fx.T0 + 70, status="failed", usage_limit=True)
        summary = self.learn()
        (row,) = self.store()
        self.assertEqual((row["pointers"]["run_id"], row["category"]), ("limit", "usage-wait"))
        self.assertEqual(summary["sources"]["t3-database"]["skipped_censored"], 1)

    def test_category_precedence_usage_wait_then_over_12h_then_research(self):
        # why: the four categories are exclusive; a research run over 12h is "over-12h", and a
        #      usage-limit ending beats both (brief: usage-wait / over-12h / research / ordinary).
        # catches: classifying by kind before duration, or letting research hide a usage wait.
        fx.add_t3_run(self.db, "ord", start=fx.T0, end=fx.T0 + 100, brief="implement a")
        fx.add_t3_run(self.db, "res", start=fx.T0, end=fx.T0 + 100, brief="research options")
        fx.add_t3_run(self.db, "long", start=fx.T0, end=fx.T0 + 43_200 + 1, brief="research options")
        fx.add_t3_run(self.db, "edge", start=fx.T0, end=fx.T0 + 43_200, brief="implement a")
        fx.add_t3_run(self.db, "wait", start=fx.T0, end=fx.T0 + 43_300, status="failed", brief="research options", usage_limit=True)
        self.learn()
        categories = {r["pointers"]["run_id"]: r["category"] for r in self.store()}
        self.assertEqual(categories, {"ord": "ordinary", "res": "research", "long": "over-12h", "edge": "ordinary", "wait": "usage-wait"})

    def test_learned_t3_row_carries_the_estimator_features(self):
        # why: learned rows feed estimate-run; a row missing effort, ordinal, hints or the
        #      concurrency and queue features weights differently from a replayed row (Neon
        #      review: learn.py was a second, partial copy of t3_history.read_database).
        # catches: building T3 rows in learn instead of through the shared reader.
        fx.add_t3_run(self.db, "a", start=fx.T0, end=fx.T0 + 900, thread_id="thread:delegated-task:a")
        fx.add_t3_run(self.db, "b", start=fx.T0 + 100, end=fx.T0 + 400, requested=fx.T0 + 40, effort="high", ordinal=3, thread_id="thread:delegated-task:b", brief="fix the failing test, human_minutes: 30")
        self.learn()
        row = {r["run_id"]: r for r in self.store()}["b"]
        self.assertEqual((row["effort"], row["ordinal"], row["delegated"]), ("high", 3, True))
        self.assertEqual((row["features"]["active_runs"], row["features"]["queue_wait_s"]), (1, 60.0))
        self.assertEqual((row["features"]["human_minutes"], row["features"]["verifiable"]), (30.0, 1))
        self.assertIn("hints", row)

    def test_incremental_run_reads_only_runs_near_the_cursor(self):
        # why: "incremental, without a full replay": every learn scanned and parsed the whole T3
        #      database (ug.load_t3_runs ran unfiltered), gigabytes on a large history.
        # catches: a reader that parses rows older than the cursor window.
        fx.add_t3_run(self.db, "old", start=fx.T0, end=fx.T0 + 60)
        self.learn()
        import sqlite3

        connection = sqlite3.connect(self.db)
        connection.execute("UPDATE orchestration_v2_projection_runs SET payload_json='{not json' WHERE run_id='old'")
        connection.commit()
        connection.close()
        fx.add_t3_run(self.db, "new", start=fx.T0 + 20 * HOUR, end=fx.T0 + 20 * HOUR + 60)
        summary = learn.run(self.state, self.sources(), now=fx.T0 + 30 * HOUR)
        t3 = summary["sources"]["t3-database"]
        self.assertEqual((t3["status"], t3["added"], t3["seen"]), ("ok", 1, 1))

    def test_main_thread_turn_is_stored_apart_and_kept_out_of_learned_history(self):
        # why: most T3 runs can be orchestrator main-thread turns (short, some waiting on
        #      the user); fed to --learned they would pull a model's P50 down.
        # catches: treating a non-delegated T3 turn as an agent run.
        fx.add_t3_run(self.db, "main", start=fx.T0, end=fx.T0 + 30, thread_id="thread:main-chat")
        fx.add_t3_run(self.db, "sub", start=fx.T0, end=fx.T0 + 900)
        self.learn()
        categories = {r["run_id"]: (r["category"], r["delegated"]) for r in self.store()}
        self.assertEqual(categories, {"main": ("orchestrator-turn", False), "sub": ("ordinary", True)})
        self.assertEqual([r["run_id"] for r in learn.reference_rows(self.state, after=0)], ["sub"])

    def test_orchestrator_turn_beats_usage_wait(self):
        # why: a main-thread turn that ended on a usage limit is still not an agent run; the role
        #      decides whether a row is an agent-run sample at all, before any duration category.
        # catches: categorising by usage limit first and scoring the turn as an agent wait.
        fx.add_t3_run(self.db, "main", start=fx.T0, end=fx.T0 + 30, status="failed", usage_limit=True, thread_id="thread:main-chat")
        self.learn()
        (row,) = self.store()
        self.assertEqual(row["category"], "orchestrator-turn")

    def test_t3_database_is_opened_read_only_and_never_created(self):
        # why: T3 is live; learn must not write it, and a missing path must not create a file.
        # catches: sqlite3.connect without mode=ro.
        fx.add_t3_run(self.db, "r1", start=fx.T0, end=fx.T0 + 100)
        before = digest(self.db)
        self.learn()
        self.assertEqual(digest(self.db), before)
        missing = self.root / "nope.sqlite"
        summary = self.learn(t3_db=missing)
        self.assertFalse(missing.exists())
        self.assertEqual(summary["sources"]["t3-database"]["status"], "absent")


class VigilSource(LearnCase):
    def test_item_keeps_its_dispatch_estimate_and_label_provenance(self):
        # why: the estimate at dispatch (vigil --expect-s, the run P80) is what calibration
        #      scores; kind from meta beats a keyword guess and says so.
        # catches: dropping expect_s, or labelling a keyword guess as explicit.
        self.ledger.write_text(
            fx.vigil_line("h1", "sol-build-thing", start=fx.T0, end=fx.T0 + 900, expect_s=1200, meta={"model": "sol", "kind": "implementation"})
            + fx.vigil_line("h2", "opus-review-diff", start=fx.T0, end=fx.T0 + 300)
        )
        self.learn()
        rows = {r["pointers"]["vigil_label"]: r for r in self.store()}
        first, second = rows["sol-build-thing"], rows["opus-review-diff"]
        self.assertEqual((first["estimate"]["p80"], first["estimate_source"], first["kind"], first["kind_source"]), (1200.0, "vigil-expect-s", "implementation", "vigil-meta"))
        self.assertEqual((first["model"], first["model_source"], first["label_source"]), ("sol", "vigil-meta", "vigil-hold"))
        self.assertEqual((second["kind"], second["kind_source"], second["model"], second["model_source"]), ("review", "label-keyword", "opus", "label-keyword"))
        self.assertIsNone(second["estimate"]["p80"])
        self.assertIsNone(second["estimate_source"])
        self.assertEqual(second["pointers"]["vigil_hold"], "h2")

    def test_abbreviated_meta_kind_joins_the_keyword_kind(self):
        # why: vigil meta `impl` and keyword `implementation` are one kind; split, each group
        #      has no history of its own.
        # catches: storing meta.kind raw.
        self.ledger.write_text(fx.vigil_line("h1", "neon-impl", start=fx.T0, end=fx.T0 + 60, expect_s=100, meta={"model": "gemini-4-neon", "kind": "impl"}))
        self.learn()
        (row,) = self.store()
        self.assertEqual((row["kind"], row["kind_source"], row["model"]), ("implementation", "vigil-meta", "neon"))

    def test_item_keeps_its_brief_features_and_machine_time(self):
        # why: the replay reads meta.brief (scope features) and meta.machine_s through
        #      local_runs.read_ledger; learn dropped both, so its rows weighted differently.
        # catches: a second ledger parser that skips the brief and machine time.
        brief = self.root / "brief.md"
        brief.write_text("- one\n- two\nedit src/a.py\n")
        os.utime(brief, (fx.T0 - 60, fx.T0 - 60))
        self.ledger.write_text(fx.vigil_line("h1", "sol-impl", start=fx.T0, end=fx.T0 + 600, expect_s=900, meta={"model": "sol", "kind": "implementation", "brief": str(brief), "machine_s": 45}))
        self.learn()
        (row,) = self.store()
        self.assertEqual((row["features"]["items"], row["features"]["files"], row["machine_s"]), (2, 1, 45.0))

    def test_unfinished_and_malformed_lines_are_skipped_and_counted(self):
        # why: the ledger is appended by other tools; a torn or non-done line must not crash or
        #      become a duration.
        # catches: raising on a bad line, or ingesting outcome != done.
        self.ledger.write_text(
            fx.vigil_line("h1", "ok", start=fx.T0, end=fx.T0 + 60, expect_s=100)
            + fx.vigil_line("h2", "killed", start=fx.T0, end=fx.T0 + 60, outcome="cancelled")
            + "{not json\n"
            + json.dumps({"v": 2, "outcome": "done"}) + "\n"
        )
        summary = self.learn()
        self.assertEqual(len(self.store()), 1)
        self.assertEqual(summary["sources"]["vigil-ledger"]["skipped"], 3)

    def test_duplicate_ledger_lines_collapse_to_one_observation(self):
        # why: ledger lines can repeat (retried writer); a repeat must not double-weight history.
        # catches: id derived from line number instead of (hold, label, start).
        line = fx.vigil_line("h1", "same", start=fx.T0, end=fx.T0 + 60, expect_s=100)
        self.ledger.write_text(line * 2)
        self.learn()
        self.assertEqual(len(self.store()), 1)


class VigilLinks(LearnCase):
    SESSION = "orch-1"
    THREAD = "thread:delegated-task:child"

    def hold(self, label, *, start, end, expect_s, meta=None):
        return fx.vigil_line("W", label, start=start, end=end, expect_s=expect_s, meta=meta).replace('"session_id": "s-orchestrator"', f'"session_id": "{self.SESSION}"')

    def test_hold_on_one_t3_run_scores_that_run_not_the_hold(self):
        # why: a hold's open-to-done time is the orchestrator's wait, not the run; matched to its
        #      run through the dispatch result, the estimate is scored against the run's actual
        #      and the hold is not a second copy of the run (a hold of
        #      1,780 s around a 409 s run).
        # catches: scoring hold lifetimes, or counting the hold and its run twice.
        fx.write_orchestrator_transcript(self.claude, self.SESSION, fx.delegate(fx.T0, "c1", "R1", self.THREAD) + [fx.bash_hold(fx.T0 + 4, "sol-fix", 1200)])
        fx.add_t3_run(self.db, "R1", start=fx.T0 + 2, end=fx.T0 + 902, thread_id=self.THREAD, model="gpt-6.1-sol")
        self.ledger.write_text(self.hold("sol-fix", start=fx.T0 + 5, end=fx.T0 + 2000, expect_s=1200))
        self.learn()
        (row,) = learn.collapse(self.store())
        self.assertEqual((row["source"], row["actual_s"], row["estimate"]["p80"], row["estimate_source"]), ("t3-database", 900.0, 1200.0, "vigil-expect-s"))
        self.assertEqual(row["pointers"]["vigil_label"], "sol-fix")
        report = calibration.calibrate(self.store(), now=fx.T0 + 10 * HOUR, window_s=7 * 86_400)
        self.assertEqual((report["estimates"]["estimated"], report["holds"]["scored"]), (1, 1))

    def test_plan_forecast_reaches_the_run_through_its_hold(self):
        # why: a plan task closed from a vigil item is the same execution; its P50 and the hold's
        #      P80 both predicted the run the hold watched.
        # catches: losing the plan estimate (or its pointers) when the hold moves onto its run.
        fx.write_orchestrator_transcript(self.claude, self.SESSION, fx.delegate(fx.T0, "c1", "R1", self.THREAD) + [fx.bash_hold(fx.T0 + 4, "sol-task", 1200)])
        fx.add_t3_run(self.db, "R1", start=fx.T0 + 2, end=fx.T0 + 902, thread_id=self.THREAD, model="gpt-6.1-sol")
        self.ledger.write_text(self.hold("sol-task", start=fx.T0 + 5, end=fx.T0 + 1000, expect_s=1200, meta={"task": "plan-a/t1"}))
        ledger_id = learn.vigil_execution_key("W", "sol-task", fx.T0 + 5)
        fx.write_plan(
            self.plans,
            "plan-a",
            [{"id": "t1", "type": "agent", "model": "sol", "kind": "fix", "status": "done", "actual_s": 995, "started_at": fx.T0 + 5, "ended_at": fx.T0 + 1000, "actual_source": "vigil-ledger", "forecast_s": 800}],
            events=[{"action": "ledger-done", "task": "t1", "at": fx.T0 + 1000, "ledger_id": ledger_id}],
        )
        self.learn()
        (row,) = learn.collapse(self.store())
        self.assertEqual((row["source"], row["actual_s"], row["estimate"]["p50"], row["estimate"]["p80"]), ("t3-database", 900.0, 800.0, 1200.0))
        self.assertEqual((row["pointers"]["plan"], row["pointers"]["task"], row["pointers"]["run_id"]), (str(self.plans / "plan-a.json"), "t1", "R1"))

    def test_hold_spanning_several_runs_is_kept_unscored(self):
        # why: `expect_s` is a single-run P80 only when the hold covers one run; a hold over 17
        #      runs against a 64,800 s budget made every estimate look like a hit.
        # catches: scoring a multi-run hold against one of its runs or its own lifetime.
        fx.write_orchestrator_transcript(self.claude, self.SESSION, fx.delegate(fx.T0, "c1", "R1", self.THREAD) + [fx.bash_hold(fx.T0 + 4, "opus-fix", 64_800)])
        fx.add_t3_run(self.db, "R1", start=fx.T0 + 2, end=fx.T0 + 900, thread_id=self.THREAD)
        fx.add_t3_run(self.db, "R2", start=fx.T0 + 1000, end=fx.T0 + 1900, thread_id=self.THREAD, ordinal=2)
        self.ledger.write_text(self.hold("opus-fix", start=fx.T0 + 5, end=fx.T0 + 5000, expect_s=64_800))
        self.learn()
        rows = learn.collapse(self.store())
        hold = next(r for r in rows if r["source"] == "vigil-ledger")
        self.assertEqual((hold["link_status"], hold["linked_runs"]), ("multi-run", 2))
        self.assertTrue(all(r["estimate"]["p80"] is None for r in rows if r["source"] == "t3-database"))
        report = calibration.calibrate(self.store(), now=fx.T0 + 10 * HOUR, window_s=7 * 86_400)
        self.assertEqual((report["estimates"]["estimated"], report["holds"]["multi-run"]), (0, 1))

    def test_hold_on_a_background_agent_scores_its_subagent(self):
        # why: many agent holds wrap a background Agent, not a T3 run; the
        #      Agent result names the subagent transcript, which holds the run's actual.
        # catches: hold_index dropping the dispatch at an Agent call.
        call = fx.tool_use(fx.T0, "a1", "Agent", {"prompt": f"fix it {fx.CANARY}", "run_in_background": True})
        result = fx.tool_result(fx.T0 + 1, "a1", "Async agent launched successfully. agentId: abc123", {"agentId": "abc123", "status": "async_launched"})
        hold = fx.tool_use(fx.T0 + 3, "h1", "mcp__vigil__hold_add", {"watch_id": "W", "label": "opus-widget", "expect_s": 3600})
        fx.write_orchestrator_transcript(self.claude, self.SESSION, [call, result, hold])
        fx.write_claude_subagent(self.claude, "agent-abc123", start=fx.T0 + 2, end=fx.T0 + 1802, session=self.SESSION, model="claude-opus-5-5")
        self.ledger.write_text(self.hold("opus-widget", start=fx.T0 + 3, end=fx.T0 + 4000, expect_s=3600))
        self.learn()
        (row,) = learn.collapse(self.store())
        self.assertEqual((row["source"], row["model"], row["actual_s"], row["estimate"]["p80"]), ("claude-subagent", "opus", 1800.0, 3600.0))

    def test_hold_on_a_subagent_resumed_by_the_coordinator_is_multi_run(self):
        # why: "exactly one run" must see a resume that arrives as an `isMeta` coordinator message;
        #      the hold then spans two runs and its P80 is not a single-run estimate.
        # catches: scoring a hold against a subagent that was resumed inside the hold.
        call = fx.tool_use(fx.T0, "a1", "Agent", {"prompt": f"fix it {fx.CANARY}", "run_in_background": True})
        result = fx.tool_result(fx.T0 + 1, "a1", "Async agent launched successfully. agentId: abc123", {"agentId": "abc123", "status": "async_launched"})
        hold = fx.tool_use(fx.T0 + 3, "h1", "mcp__vigil__hold_add", {"watch_id": "W", "label": "opus-widget", "expect_s": 3600})
        fx.write_orchestrator_transcript(self.claude, self.SESSION, [call, result, hold])
        turns = [fx.claude_turn(fx.T0 + 2, fx.T0 + 302, model="claude-opus-5-5"), fx.claude_turn(fx.T0 + 3000, fx.T0 + 3100, model="claude-opus-5-5", meta="coordinator")]
        fx.write_claude_subagent(self.claude, "agent-abc123", start=0, end=0, session=self.SESSION, turns=turns)
        self.ledger.write_text(self.hold("opus-widget", start=fx.T0 + 3, end=fx.T0 + 4000, expect_s=3600))
        self.learn()
        rows = learn.collapse(self.store())
        hold_row = next(r for r in rows if r["source"] == "vigil-ledger")
        self.assertEqual((hold_row["link_status"], hold_row["linked_runs"]), ("multi-run", 2))

    def test_hold_registered_after_its_run_ended_is_not_scored(self):
        # why: a hold opened after the run it names had already finished predicted nothing about
        #      that run; the estimate was made with the answer known (Fable review r2).
        # catches: scoring a linked hold whose target ended before the hold started.
        fx.write_orchestrator_transcript(self.claude, self.SESSION, fx.delegate(fx.T0, "c1", "R1", self.THREAD) + [fx.bash_hold(fx.T0 + 400, "sol-fix", 1200)])
        fx.add_t3_run(self.db, "R1", start=fx.T0 + 2, end=fx.T0 + 302, thread_id=self.THREAD, model="gpt-6.1-sol")
        self.ledger.write_text(self.hold("sol-fix", start=fx.T0 + 401, end=fx.T0 + 500, expect_s=1200))
        self.learn()
        rows = learn.collapse(self.store())
        hold = next(r for r in rows if r["source"] == "vigil-ledger")
        self.assertEqual(hold["link_status"], "target-ended-first")
        self.assertTrue(all(r["estimate"]["p80"] is None for r in rows if r["source"] == "t3-database"))

    def test_hold_on_a_cancelled_t3_run_says_so(self):
        # why: 3 holds read "target-missing" with no way to tell a cancelled run (nothing to
        #      score, nothing wrong) from a link that failed (a bug to look at).
        # catches: one status for both.
        fx.write_orchestrator_transcript(self.claude, self.SESSION, fx.delegate(fx.T0, "c1", "R1", self.THREAD) + [fx.bash_hold(fx.T0 + 4, "sol-fix", 1200)])
        fx.add_t3_run(self.db, "R1", start=fx.T0 + 2, end=fx.T0 + 100, thread_id=self.THREAD, model="gpt-6.1-sol", status="cancelled")
        self.ledger.write_text(self.hold("sol-fix", start=fx.T0 + 5, end=fx.T0 + 200, expect_s=1200))
        self.learn()
        (hold,) = learn.collapse(self.store())
        self.assertEqual(hold["link_status"], "target-cancelled")

    def test_hold_on_a_subagent_with_no_ended_turn_is_target_cancelled(self):
        # why: a background agent that was stopped before it finished a turn has a transcript
        #      and no run; the hold has nothing to score.
        # catches: reporting a stopped agent as a failed link.
        call = fx.tool_use(fx.T0, "a1", "Agent", {"prompt": f"fix it {fx.CANARY}", "run_in_background": True})
        result = fx.tool_result(fx.T0 + 1, "a1", "Async agent launched successfully. agentId: abc123", {"agentId": "abc123", "status": "async_launched"})
        hold = fx.tool_use(fx.T0 + 3, "h1", "mcp__vigil__hold_add", {"watch_id": "W", "label": "opus-widget", "expect_s": 3600})
        fx.write_orchestrator_transcript(self.claude, self.SESSION, [call, result, hold])
        fx.write_claude_subagent(self.claude, "agent-abc123", start=0, end=0, session=self.SESSION, turns=[fx.claude_turn(fx.T0 + 2, None, model="claude-opus-5-5")])
        self.ledger.write_text(self.hold("opus-widget", start=fx.T0 + 3, end=fx.T0 + 400, expect_s=3600))
        self.learn()
        (hold_row,) = learn.collapse(self.store())
        self.assertEqual(hold_row["link_status"], "target-cancelled")

    def test_hold_whose_run_is_nowhere_is_link_failed(self):
        # why: a dispatch result that names a run the store never saw is a broken link, not a
        #      cancellation.
        # catches: calling every missing target cancelled.
        fx.write_orchestrator_transcript(self.claude, self.SESSION, fx.delegate(fx.T0, "c1", "R-gone", self.THREAD) + [fx.bash_hold(fx.T0 + 4, "sol-fix", 1200)])
        self.ledger.write_text(self.hold("sol-fix", start=fx.T0 + 5, end=fx.T0 + 200, expect_s=1200))
        self.learn()
        (hold,) = learn.collapse(self.store())
        self.assertEqual(hold["link_status"], "link-failed")

    def test_report_shows_cancelled_and_failed_links_apart(self):
        # why: the report is where a reader sees why holds were not scored.
        # catches: a status that is counted but not shown.
        fx.write_orchestrator_transcript(
            self.claude, self.SESSION,
            fx.delegate(fx.T0, "c1", "R1", self.THREAD) + [fx.bash_hold(fx.T0 + 4, "sol-a", 1200)] + fx.delegate(fx.T0 + 10, "c2", "R-gone", "thread:other") + [fx.bash_hold(fx.T0 + 14, "sol-b", 1200)],
        )
        fx.add_t3_run(self.db, "R1", start=fx.T0 + 2, end=fx.T0 + 100, thread_id=self.THREAD, model="gpt-6.1-sol", status="cancelled")
        self.ledger.write_text(self.hold("sol-a", start=fx.T0 + 5, end=fx.T0 + 200, expect_s=1200) + self.hold("sol-b", start=fx.T0 + 15, end=fx.T0 + 200, expect_s=1200))
        self.learn()
        report = calibration.calibrate(self.store(), now=fx.T0 + 10 * HOUR, window_s=7 * 86_400)
        self.assertEqual((report["holds"]["target-cancelled"], report["holds"]["link-failed"]), (1, 1))
        text = calibration.render_markdown(report)
        self.assertIn("target-cancelled 1", text)
        self.assertIn("link-failed 1", text)

    def test_checkin_machine_and_unlinked_holds_are_stored_not_scored(self):
        # why: most `unknown`-model ledger items are check-in timers or machine jobs (suite, emulator,
        #      remote build); scored as agent runs they made all three drift flags.
        # catches: treating every done ledger item as an agent run.
        self.ledger.write_text(
            self.hold("checkin5-widgets", start=fx.T0, end=fx.T0 + 360, expect_s=360)
            + self.hold("owner-decision-demo", start=fx.T0, end=fx.T0 + 9000, expect_s=7200)
            + self.hold("emu-import-test", start=fx.T0, end=fx.T0 + 900, expect_s=5400)
            + self.hold("land42-suite", start=fx.T0, end=fx.T0 + 900, expect_s=5400)
            + self.hold("rel42-rest", start=fx.T0, end=fx.T0 + 900, expect_s=1500)
            + self.hold("opus-parser-review", start=fx.T0, end=fx.T0 + 900, expect_s=3600)
        )
        self.learn()
        rows = {r["pointers"]["vigil_label"]: r for r in self.store()}
        self.assertEqual(
            {k: (r["category"], r["link_status"]) for k, r in rows.items()},
            {
                "checkin5-widgets": ("checkin", "not-an-agent-run"),
                "owner-decision-demo": ("checkin", "not-an-agent-run"),
                "emu-import-test": ("machine", "not-an-agent-run"),
                "land42-suite": ("machine", "not-an-agent-run"),
                "rel42-rest": ("machine", "not-an-agent-run"),
                "opus-parser-review": ("ordinary", "no-transcript"),
            },
        )
        report = calibration.calibrate(self.store(), now=fx.T0 + 10 * HOUR, window_s=7 * 86_400)
        self.assertEqual((report["groups"], report["misses"], report["estimates"]["estimated"]), ([], [], 0))
        self.assertEqual((report["categories"]["checkin"], report["categories"]["machine"], report["holds"]["no-transcript"]), (2, 3, 1))

    def test_hold_registered_long_after_the_last_dispatch_is_not_linked(self):
        # why: hold_index binds a hold to the last dispatch before it, however old; a hold made
        #      an hour later is for other work.
        # catches: linking without a bound on the dispatch-to-hold gap.
        fx.write_orchestrator_transcript(self.claude, self.SESSION, fx.delegate(fx.T0, "c1", "R1", self.THREAD) + [fx.bash_hold(fx.T0 + 3600, "sol-later", 600)])
        fx.add_t3_run(self.db, "R1", start=fx.T0 + 2, end=fx.T0 + 500, thread_id=self.THREAD, model="gpt-6.1-sol")
        self.ledger.write_text(self.hold("sol-later", start=fx.T0 + 3601, end=fx.T0 + 4000, expect_s=600))
        self.learn()
        (hold,) = [r for r in learn.collapse(self.store()) if r["source"] == "vigil-ledger"]
        self.assertEqual((hold["link_status"], hold["link"]), ("stale-dispatch", None))

    def test_hold_naming_another_model_than_its_run_is_not_scored(self):
        # why: a `sonnet-` hold bound to a Sol run is bound to the wrong dispatch; scoring it would
        #      grade one model's estimate with another model's run.
        # catches: trusting the link without checking the models agree.
        fx.write_orchestrator_transcript(self.claude, self.SESSION, fx.delegate(fx.T0, "c1", "R1", self.THREAD) + [fx.bash_hold(fx.T0 + 4, "sonnet-x", 600)])
        fx.add_t3_run(self.db, "R1", start=fx.T0 + 2, end=fx.T0 + 500, thread_id=self.THREAD, model="gpt-6.1-sol")
        self.ledger.write_text(self.hold("sonnet-x", start=fx.T0 + 5, end=fx.T0 + 600, expect_s=600))
        self.learn()
        (hold,) = [r for r in learn.collapse(self.store()) if r["source"] == "vigil-ledger"]
        self.assertEqual(hold["link_status"], "model-mismatch")

    def test_two_holds_bound_to_one_run_score_neither(self):
        # why: hold_index binds a hold to the last dispatch before it, so a later hold for other
        #      work can claim the same run; neither estimate is then known to be that run's.
        # catches: scoring the first or last of several holds on one run.
        fx.write_orchestrator_transcript(self.claude, self.SESSION, fx.delegate(fx.T0, "c1", "R1", self.THREAD) + [fx.bash_hold(fx.T0 + 4, "sol-a", 600), fx.bash_hold(fx.T0 + 30, "sol-b", 900)])
        fx.add_t3_run(self.db, "R1", start=fx.T0 + 2, end=fx.T0 + 500, thread_id=self.THREAD, model="gpt-6.1-sol")
        self.ledger.write_text(self.hold("sol-a", start=fx.T0 + 5, end=fx.T0 + 600, expect_s=600) + self.hold("sol-b", start=fx.T0 + 31, end=fx.T0 + 700, expect_s=900))
        self.learn()
        statuses = sorted(r["link_status"] for r in learn.collapse(self.store()) if r["source"] == "vigil-ledger")
        self.assertEqual(statuses, ["shared-target", "shared-target"])


class TranscriptSources(LearnCase):
    def test_claude_subagent_and_codex_rollout_are_ingested(self):
        # why: transcripts are the third source; Codex turns that ended on a usage limit are the
        #      usage-wait category.
        # catches: reading only one transcript family, or ignoring the usage-limit ending.
        fx.write_claude_subagent(self.claude, "a1", start=fx.T0, end=fx.T0 + 240)
        fx.write_codex_rollout(self.codex, "c1", session_id="nat-c1", start=fx.T0, end=fx.T0 + 500)
        fx.write_codex_rollout(self.codex, "c2", session_id="nat-c2", start=fx.T0, end=fx.T0 + 80, usage_limit=True)
        self.learn()
        rows = {(r["source"], r["category"]): r for r in self.store()}
        self.assertEqual(set(rows), {("claude-subagent", "ordinary"), ("codex-transcript", "ordinary"), ("codex-transcript", "usage-wait")})
        claude = rows[("claude-subagent", "ordinary")]
        self.assertEqual((claude["model"], claude["kind"], claude["actual_s"], claude["label_source"]), ("sonnet", "review", 240.0, "transcript"))
        self.assertTrue(claude["pointers"]["transcript"].endswith("a1.jsonl"))
        self.assertNotIn("CANARY", (self.state / "observations.jsonl").read_text())

    def test_codex_brief_in_a_response_item_sets_the_kind(self):
        # why: newer Codex rollouts carry the user prompt as a response_item; the shared
        #      run_labels.codex_turns reads it, learn's copy did not and labelled the run `other`.
        # catches: a second Codex parser that only reads event_msg user messages.
        fx.write_codex_rollout(self.codex, "c1", session_id="nat-c1", start=fx.T0, end=fx.T0 + 500, brief="review the diff", brief_as_response_item=True)
        self.learn()
        (row,) = self.store()
        self.assertEqual((row["kind"], row["effort"]), ("review", "high"))

    def test_unchanged_old_transcript_is_not_reread(self):
        # why: "incremental, without a full replay": files untouched since the cursor are skipped
        #      by mtime, so a nightly learn does not reparse 17 MB of history.
        # catches: parsing every transcript on every run (full replay).
        path = fx.write_claude_subagent(self.claude, "old", start=fx.T0, end=fx.T0 + 240)
        os.utime(path, (fx.T0, fx.T0))
        self.learn()
        fx.write_claude_subagent(self.claude, "old", start=fx.T0, end=fx.T0 + 999)  # same file, new content
        os.utime(path, (fx.T0, fx.T0))  # but its mtime says nothing changed
        self.learn()
        (row,) = self.store()
        self.assertEqual(row["actual_s"], 240.0)


class ClaudeSubagents(LearnCase):
    def test_api_error_stub_neither_names_the_model_nor_counts_as_work(self):
        # why: a `<synthetic>` API-error stub after a finished turn became the row's model
        #     ; usage_gap and run_labels already ignore stubs.
        # catches: taking the model from the last assistant record, synthetic or not.
        turns = [fx.claude_turn(fx.T0, fx.T0 + 300), fx.claude_turn(fx.T0 + 400, fx.T0 + 410, brief="again", ending="synthetic")]
        fx.write_claude_subagent(self.claude, "a1", start=0, end=0, turns=turns)
        self.learn()
        (row,) = self.store()
        self.assertEqual((row["model"], row["model_raw"], row["turns"], row["actual_s"]), ("sonnet", "claude-sonnet-5-5", 1, 300.0))

    def test_resume_by_a_coordinator_message_is_a_new_span_not_idle_work(self):
        # why: a subagent resumed by the parent arrives as an `isMeta` user record, which
        #      claude_turns did not treat as a turn start, so the 2,700 s idle gap before it was
        #      counted as work.
        # catches: merging a resumed turn into the one before it.
        turns = [fx.claude_turn(fx.T0, fx.T0 + 300), fx.claude_turn(fx.T0 + 3000, fx.T0 + 3100, brief="also fix the test", meta="coordinator")]
        fx.write_claude_subagent(self.claude, "a1", start=0, end=0, turns=turns)
        self.learn()
        (row,) = self.store()
        self.assertEqual((row["actual_s"], row["turns"], row["wall_s"]), (400.0, 2, 3100.0))

    def test_system_notification_after_a_finished_turn_also_starts_a_span(self):
        # why: a background-task notification wakes an idle subagent; the wait before it is not work.
        # catches: starting spans only at coordinator messages.
        turns = [fx.claude_turn(fx.T0, fx.T0 + 300), fx.claude_turn(fx.T0 + 3000, fx.T0 + 3100, brief="[SYSTEM NOTIFICATION] done", meta="task-notification")]
        fx.write_claude_subagent(self.claude, "a1", start=0, end=0, turns=turns)
        self.learn()
        (row,) = self.store()
        self.assertEqual((row["actual_s"], row["turns"]), (400.0, 2))

    def test_meta_record_inside_a_working_turn_does_not_split_it(self):
        # why: the harness injects `isMeta` records (skill text, reminders) between tool calls;
        #      only one after an end_turn is a resume.
        # catches: starting a span at every `isMeta` record.
        records = fx.claude_turn(fx.T0, fx.T0 + 50, ending="none") + fx.claude_turn(fx.T0 + 60, fx.T0 + 120, meta="coordinator")[:1] + fx.claude_turn(fx.T0 + 60, fx.T0 + 300)[1:]
        fx.write_claude_subagent(self.claude, "a1", start=0, end=0, turns=[records])
        self.learn()
        (row,) = self.store()
        self.assertEqual((row["actual_s"], row["turns"]), (300.0, 1))

    def test_stub_inside_a_turn_does_not_rename_the_turn_model(self):
        # why: run_labels.claude_turns took the model from every assistant record, so a stub at
        #      the end of a working turn renamed the turn `<synthetic>`.
        # catches: reading the model from synthetic records.
        from agentpace import run_labels

        records = fx.claude_turn(fx.T0, fx.T0 + 50, ending="none") + fx.claude_turn(fx.T0, fx.T0 + 60, ending="usage-limit")[1:]
        path = self.root / "t.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        (turn,) = run_labels.claude_turns(path)
        self.assertEqual((turn["model"], turn["censor_reason"], turn["end"] - turn["start"]), ("sonnet", "usage-limit", 60.0))

    def test_subagent_that_ended_on_a_usage_limit_is_a_usage_wait(self):
        # why: a usage-limit ending is a provider wait, not an estimate error; usage_gap marks
        #      it from the synthetic stub's text, learn dropped the run entirely.
        # catches: ignoring the usage-limit stub, or scoring the wait as agent work.
        fx.write_claude_subagent(self.claude, "a1", start=0, end=0, turns=[fx.claude_turn(fx.T0, fx.T0 + 900, ending="usage-limit")])
        self.learn()
        (row,) = self.store()
        self.assertEqual((row["category"], row["censor_reason"]), ("usage-wait", "usage-limit"))
        self.assertEqual(learn.reference_rows(self.state, after=0), [])

    def test_subagent_inside_a_delegated_run_is_linked_and_not_a_second_sample(self):
        # why: an Agent subagent runs inside its parent T3 turn; the parent's duration already
        #      holds the child's time, so both as samples count that time twice.
        # catches: dead dedupe (session compared with a path hash), or dropping the child.
        fx.add_t3_run(self.db, "parent", start=fx.T0, end=fx.T0 + 1000, native_id="sess-1")
        fx.write_claude_subagent(self.claude, "child", start=fx.T0 + 100, end=fx.T0 + 400, session="sess-1")
        self.learn()
        rows = {r["source"]: r for r in learn.collapse(self.store())}
        self.assertEqual((rows["claude-subagent"]["parent_run_id"], rows["claude-subagent"]["nested_in"]), ("parent", "parent"))
        self.assertEqual([r["source"] for r in learn.reference_rows(self.state, after=0)], ["t3-database"])

    def test_category_line_counts_each_run_once(self):
        # why: the category line counted an unscored hold and a nested subagent as runs beside
        #      the run each one watched or sat inside (Fable review r2).
        # catches: counting every collapsed row, not every run.
        fx.write_orchestrator_transcript(self.claude, "orch-1", fx.delegate(fx.T0, "c1", "R-gone", "thread:other") + [fx.bash_hold(fx.T0 + 4, "sol-fix", 1200)])
        self.ledger.write_text(fx.vigil_line("W", "sol-fix", start=fx.T0 + 5, end=fx.T0 + 300, expect_s=1200).replace('"session_id": "s-orchestrator"', '"session_id": "orch-1"'))
        fx.add_t3_run(self.db, "parent", start=fx.T0, end=fx.T0 + 1000, native_id="sess-1")
        fx.write_claude_subagent(self.claude, "child", start=fx.T0 + 100, end=fx.T0 + 400, session="sess-1")
        self.learn()
        report = calibration.calibrate(self.store(), now=fx.T0 + 10 * HOUR, window_s=7 * 86_400)
        self.assertEqual(report["holds"]["link-failed"], 1)
        self.assertEqual(report["observations"]["nested_subagents"], 1)
        self.assertEqual(sum(report["categories"][c] for c in ("usage-wait", "over-12h", "research", "ordinary")), 1)

    def test_subagent_of_an_orchestrator_turn_is_linked_and_kept_as_a_sample(self):
        # why: the orchestrator's turn is not an agent-run sample, so the child is the only
        #      sample of that work and must stay.
        # catches: dropping every child of a T3 turn.
        fx.add_t3_run(self.db, "turn", start=fx.T0, end=fx.T0 + 5000, native_id="sess-2", thread_id="thread:main-chat")
        fx.write_claude_subagent(self.claude, "child", start=fx.T0 + 100, end=fx.T0 + 400, session="sess-2")
        self.learn()
        child = next(r for r in learn.collapse(self.store()) if r["source"] == "claude-subagent")
        self.assertEqual((child["parent_run_id"], child.get("nested_in")), ("turn", None))
        self.assertEqual([r["source"] for r in learn.reference_rows(self.state, after=0)], ["claude-subagent"])


class FileCursor(LearnCase):
    def test_file_older_than_the_last_one_read_is_still_read(self):
        # why: one max-mtime cursor per source skipped forever any file written in the same tick
        #      as, or before, a file already read (`_files_newer_than` once used `>`).
        # catches: a global mtime watermark instead of per-file state.
        first = fx.write_claude_subagent(self.claude, "a1", start=fx.T0, end=fx.T0 + 100)
        os.utime(first, (fx.T0 + 500, fx.T0 + 500))
        self.learn()
        late = fx.write_claude_subagent(self.claude, "a2", start=fx.T0, end=fx.T0 + 200)
        os.utime(late, (fx.T0 + 500, fx.T0 + 500))  # same tick as the file already read
        self.learn()
        self.assertEqual(sorted(r["actual_s"] for r in self.store()), [100.0, 200.0])

    def test_plan_that_failed_to_parse_is_read_once_it_parses(self):
        # why: read_plans moved the cursor past a file before parsing it, so a plan caught
        #      mid-write was never read again.
        # catches: advancing the cursor before a successful parse.
        path = self.plans / "plan-a.json"
        self.plans.mkdir(parents=True)
        path.write_text("{torn")
        os.utime(path, (fx.T0, fx.T0))
        summary = self.learn()
        self.assertEqual(summary["sources"]["plan"]["skipped_files"], 1)
        fx.write_plan(self.plans, "plan-a", [{"id": "t1", "type": "agent", "model": "opus", "kind": "review", "status": "done", "actual_s": 700, "started_at": fx.T0, "ended_at": fx.T0 + 700}])
        os.utime(path, (fx.T0, fx.T0))
        self.learn()
        self.assertEqual([r["source"] for r in self.store()], ["plan"])

    def test_resumed_subagent_updates_its_observation(self):
        # why: a subagent resumed for a second turn is the same transcript; keying it by its end
        #      added a second observation for the same run.
        # catches: an id that includes the end, or a store that cannot supersede a row.
        turn1 = fx.claude_turn(fx.T0, fx.T0 + 300)
        fx.write_claude_subagent(self.claude, "a1", start=0, end=0, turns=[turn1])
        self.learn()
        fx.write_claude_subagent(self.claude, "a1", start=0, end=0, turns=[turn1, fx.claude_turn(fx.T0 + 3000, fx.T0 + 3100, brief="also fix the test")])
        summary = learn.run(self.state, self.sources(), now=fx.T0 + 20 * HOUR)
        (row,) = self.store()
        self.assertEqual(summary["sources"]["claude-subagent"]["updated"], 1)
        # Agent work only: 300 s + 100 s; the 2,600 s between turns was the parent's time.
        self.assertEqual((row["actual_s"], row["turns"], row["start"], row["end"]), (400.0, 2, fx.T0, fx.T0 + 3100))
        self.assertEqual(len((self.state / "observations.jsonl").read_text().splitlines()), 2)

    def test_store_written_by_an_older_schema_is_set_aside_and_rebuilt(self):
        # why: ids and categories changed with the shared readers; mixing old rows with new ones
        #      would count each subagent twice. The store is derived data, so it is rebuilt.
        # catches: reading old-schema rows as if they were current.
        fx.add_t3_run(self.db, "r1", start=fx.T0, end=fx.T0 + 600)
        self.state.mkdir(parents=True)
        (self.state / "observations.jsonl").write_text(json.dumps({"id": "old", "v": 1, "source": "claude-subagent"}) + "\n")
        (self.state / "cursor.json").write_text(json.dumps({"t3-database": fx.T0 + 9999}))
        summary = self.learn()
        self.assertEqual(summary["migrated_from_schema"], 1)
        self.assertEqual([r["run_id"] for r in self.store()], ["r1"])
        self.assertTrue((self.state / "observations.v1.jsonl").exists())


    def test_store_from_before_resumes_were_split_is_rebuilt(self):
        # why: schema 2 stored a coordinator-resumed subagent as one turn with its idle gap in the
        #      actual, and an unchanged transcript is never re-read, so the rows must be rebuilt.
        # catches: keeping schema-2 actuals that include idle gaps.
        turns = [fx.claude_turn(fx.T0, fx.T0 + 300), fx.claude_turn(fx.T0 + 3000, fx.T0 + 3100, meta="coordinator")]
        fx.write_claude_subagent(self.claude, "a1", start=0, end=0, turns=turns)
        self.state.mkdir(parents=True)
        (self.state / "observations.jsonl").write_text(json.dumps({"id": "old", "schema": 2, "source": "claude-subagent", "actual_s": 3100.0}) + "\n")
        (self.state / "cursor.json").write_text(json.dumps({"schema": 2}))
        summary = self.learn()
        self.assertEqual(summary["migrated_from_schema"], 2)
        (row,) = self.store()
        self.assertEqual(row["actual_s"], 400.0)


class ErrorPaths(LearnCase):
    def test_corrupt_t3_row_is_skipped_and_counted(self):
        # why: one row whose JSON does not parse crashed learn (json.loads outside any guard,
        #      and SQLite json_extract raising inside the join).
        # catches: a single bad row aborting the T3 source.
        fx.add_t3_run(self.db, "good", start=fx.T0, end=fx.T0 + 60)
        fx.add_t3_run(self.db, "bad", start=fx.T0, end=fx.T0 + 60)
        import sqlite3

        connection = sqlite3.connect(self.db)
        connection.execute("UPDATE orchestration_v2_projection_runs SET payload_json='{torn' WHERE run_id='bad'")
        connection.commit()
        connection.close()
        summary = self.learn()
        t3 = summary["sources"]["t3-database"]
        self.assertEqual((t3["status"], t3["added"], t3["skipped_invalid"]), ("ok", 1, 1))

    def test_unreadable_transcript_is_skipped_counted_and_retried(self):
        # why: one vanished or unreadable rollout raised out of the reader and lost the whole
        #      source and its cursor update.
        # catches: no per-file guard, or recording a file that was never read.
        fx.write_codex_rollout(self.codex, "c1", session_id="n1", start=fx.T0, end=fx.T0 + 100)
        locked = fx.write_codex_rollout(self.codex, "c2", session_id="n2", start=fx.T0, end=fx.T0 + 200)
        locked.chmod(0)
        self.addCleanup(locked.chmod, 0o600)
        summary = self.learn()
        codex = summary["sources"]["codex-transcript"]
        self.assertEqual((codex["status"], codex["added"], codex["skipped_files"]), ("ok", 1, 1))
        locked.chmod(0o600)
        summary = self.learn()
        self.assertEqual(summary["sources"]["codex-transcript"]["added"], 1)

    def test_one_bad_record_drops_its_turn_not_the_file(self):
        # why: a transcript record with a malformed timestamp made run_labels.finish raise, which
        #      threw away every other turn of that transcript.
        # catches: a per-file guard where a per-turn one is needed.
        bad = fx.claude_turn(fx.T0, fx.T0 + 100)
        bad[1]["timestamp"] = "not a time"
        bad[0]["timestamp"] = "also not a time"
        fx.write_claude_subagent(self.claude, "a1", start=0, end=0, turns=[bad, fx.claude_turn(fx.T0 + 500, fx.T0 + 800)])
        summary = self.learn()
        (row,) = self.store()
        self.assertEqual((row["actual_s"], summary["sources"]["claude-subagent"]["skipped_files"]), (300.0, 0))

    def test_unexpected_reader_error_reports_the_source_and_keeps_the_rest(self):
        # why: an error type a reader did not expect aborted the whole learn run, losing every
        #      other source's rows.
        # catches: catching only sqlite3.Error and OSError at the source boundary.
        fx.add_t3_run(self.db, "r1", start=fx.T0, end=fx.T0 + 60)
        self.ledger.write_text(fx.vigil_line("h", "opus-x", start=fx.T0, end=fx.T0 + 60))
        with mock.patch("agentpace.learn.lr.ledger_row", side_effect=RuntimeError("boom")):
            summary = self.learn()
        self.assertEqual(summary["sources"]["vigil-ledger"]["status"], "error: RuntimeError")
        self.assertEqual(summary["sources"]["t3-database"]["added"], 1)

    def test_per_source_added_counts_rows_after_dedupe(self):
        # why: a repeated ledger line was counted twice in the source's `added` but stored once.
        # catches: counting gathered rows instead of appended ones.
        line = fx.vigil_line("h1", "opus-same", start=fx.T0, end=fx.T0 + 60, expect_s=100)
        self.ledger.write_text(line * 2)
        summary = self.learn()
        self.assertEqual((summary["added"], summary["sources"]["vigil-ledger"]["added"]), (1, 1))


class Idempotence(LearnCase):
    def populate(self):
        fx.add_t3_run(self.db, "r1", start=fx.T0, end=fx.T0 + 600)
        self.ledger.write_text(fx.vigil_line("h1", "l", start=fx.T0, end=fx.T0 + 60, expect_s=90))
        fx.write_claude_subagent(self.claude, "a1", start=fx.T0, end=fx.T0 + 240)
        fx.write_codex_rollout(self.codex, "c1", session_id="nat-c1", start=fx.T0, end=fx.T0 + 500)
        fx.write_plan(self.plans, "plan-a", [{"id": "t1", "type": "agent", "model": "sol", "kind": "implementation", "status": "done", "actual_s": 700, "started_at": fx.T0, "ended_at": fx.T0 + 700, "forecast_s": 650}])

    def test_second_run_adds_nothing_and_changes_no_byte(self):
        # why: idempotent ingest (brief invariant): running learn twice adds nothing.
        # catches: id depending on ingest time, or appending rows already stored.
        self.populate()
        first = self.learn()
        self.assertEqual(first["added"], 5)
        before = (digest(self.state / "observations.jsonl"), digest(self.state / "cursor.json"))
        second = learn.run(self.state, self.sources(), now=fx.T0 + 20 * HOUR)
        self.assertEqual(second["added"], 0)
        self.assertEqual((digest(self.state / "observations.jsonl"), digest(self.state / "cursor.json")), before)

    def test_rerun_with_cursor_deleted_does_not_double_the_store(self):
        # why: the cursor is only an optimisation; the dedupe key is the safety net.
        # catches: dedupe by cursor position alone.
        self.populate()
        self.learn()
        (self.state / "cursor.json").unlink()
        summary = self.learn()
        self.assertEqual(summary["added"], 0)
        self.assertEqual(len(self.store()), 5)

    def test_store_deleted_but_cursor_kept_rebuilds_from_the_sources(self):
        # why: a cursor that says "read up to here" with no rows behind it left learn adding
        #      nothing for ever after the store was lost (Fable review r2).
        # catches: trusting the cursor when the store has no rows.
        fx.add_t3_run(self.db, "r1", start=fx.T0, end=fx.T0 + 600)
        fx.write_claude_subagent(self.claude, "a1", start=fx.T0, end=fx.T0 + 300)
        self.learn()
        self.assertEqual(len(self.store()), 2)
        (self.state / "observations.jsonl").unlink()
        summary = self.learn()
        self.assertEqual((summary["added"], len(self.store())), (2, 2))

    def test_crash_before_the_cursor_moves_loses_and_doubles_nothing(self):
        # why: brief invariant: "write then atomically move the cursor". A crash between the two
        #      leaves rows without a cursor; the retry must add each row exactly once.
        # catches: moving the cursor before the rows are durable, or no dedupe on retry.
        self.populate()
        with mock.patch("agentpace.learn.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.learn()
        self.assertFalse((self.state / "cursor.json").exists())
        self.assertEqual(len(self.store()), 5)
        summary = self.learn()
        self.assertEqual(summary["added"], 0)
        self.assertEqual(len(self.store()), 5)
        self.assertTrue((self.state / "cursor.json").exists())

    def test_torn_final_line_is_repaired_not_glued_to_the_next_row(self):
        # why: a crash mid-append leaves a partial line; the next row must start on a fresh line
        #      or both are lost.
        # catches: appending without checking the file ends in a newline.
        self.populate()
        self.learn()
        store = self.state / "observations.jsonl"
        with store.open("a") as handle:
            handle.write('{"v": 1, "id": "torn')
        fx.add_t3_run(self.db, "r2", start=fx.T0 + 1000, end=fx.T0 + 1600)
        summary = learn.run(self.state, self.sources(), now=fx.T0 + 20 * HOUR)
        self.assertEqual(summary["added"], 1)
        self.assertEqual(len(self.store()), 6)

    def test_store_is_owner_only(self):
        # why: the store holds thread ids, transcript paths and vigil labels; it is private state
        #      on a shared machine, not a world-readable file.
        # catches: default umask 0644 files in a 0755 directory.
        fx.add_t3_run(self.db, "r1", start=fx.T0, end=fx.T0 + 600)
        self.learn()
        modes = {path.name: path.stat().st_mode & 0o777 for path in (self.state, self.state / "observations.jsonl", self.state / "cursor.json")}
        self.assertEqual(modes, {"state": 0o700, "observations.jsonl": 0o600, "cursor.json": 0o600})

    def test_cursor_is_the_latest_finished_time_per_source(self):
        # why: the cursor is evidence of where the next run resumes; wrong by one source and a
        #      source is either replayed forever or skipped.
        # catches: a single global cursor, or the ingest time used as cursor.
        fx.add_t3_run(self.db, "r1", start=fx.T0, end=fx.T0 + 600)
        fx.add_t3_run(self.db, "r2", start=fx.T0, end=fx.T0 + 900)
        self.ledger.write_text(fx.vigil_line("h1", "l", start=fx.T0, end=fx.T0 + 60))
        self.learn()
        cursor = learn.read_cursor(self.state)
        self.assertEqual(cursor["t3-database"], fx.T0 + 900)
        self.assertEqual(cursor["vigil-ledger"], fx.T0 + 60)


class PlanSource(LearnCase):
    def test_done_agent_task_keeps_its_plan_forecast(self):
        # why: a plan ETA at dispatch is the second estimate source the brief names.
        # catches: ignoring plan files, or importing todo/removed/machine tasks as runs.
        fx.write_plan(
            self.plans,
            "plan-a",
            [
                {"id": "t1", "type": "agent", "model": "opus", "kind": "review", "status": "done", "actual_s": 700, "started_at": fx.T0, "ended_at": fx.T0 + 700, "forecast_s": 650},
                {"id": "t2", "type": "agent", "model": "opus", "kind": "review", "status": "todo"},
                {"id": "t3", "type": "machine", "status": "done", "actual_s": 90, "started_at": fx.T0, "ended_at": fx.T0 + 90},
                {"id": "t4", "type": "agent", "model": "opus", "kind": "review", "status": "removed"},
            ],
        )
        self.learn()
        (row,) = self.store()
        self.assertEqual((row["source"], row["estimate"]["p50"], row["estimate_source"], row["kind_source"]), ("plan", 650.0, "plan-forecast", "plan"))
        self.assertEqual((row["pointers"]["plan"], row["pointers"]["task"]), (str(self.plans / "plan-a.json"), "t1"))


class SourcesAndCli(LearnCase):
    def test_absent_sources_are_reported_not_fatal(self):
        # why: any source can be missing (no codex on a host); learn must still ingest the rest
        #      and say what it could not read rather than invent zeros.
        # catches: a missing path raising, or reported as "0 added" without saying absent.
        fx.add_t3_run(self.db, "r1", start=fx.T0, end=fx.T0 + 60)
        summary = self.learn()
        self.assertEqual(summary["sources"]["vigil-ledger"]["status"], "absent")
        self.assertEqual(summary["sources"]["codex-transcript"]["status"], "absent")
        self.assertEqual(summary["sources"]["t3-database"]["status"], "ok")
        self.assertEqual(summary["added"], 1)

    def test_state_dir_follows_the_xdg_state_home(self):
        # why: the store lives outside the repo, in the XDG state directory.
        # catches: a hard-coded path under the repo or ~/.agentpace (which holds shared rows).
        with mock.patch.dict(os.environ, {"XDG_STATE_HOME": "/xdg", "AGENTPACE_STATE": ""}):
            self.assertEqual(paths.state_dir(), Path("/xdg/agentpace"))
        with mock.patch.dict(os.environ, {"AGENTPACE_STATE": "/explicit"}):
            self.assertEqual(paths.state_dir(), Path("/explicit"))

    def test_cli_learn_reads_env_sources_and_prints_json_counts(self):
        # why: the env names (AGENTPACE_T3_DB, AGENTPACE_VIGIL_LEDGER) select sources;
        #      the summary is counts only.
        # catches: ignoring the env names, or printing observation contents.
        fx.add_t3_run(self.db, "r1", start=fx.T0, end=fx.T0 + 60)
        env = {"AGENTPACE_T3_DB": str(self.db), "AGENTPACE_VIGIL_LEDGER": str(self.ledger), "AGENTPACE_STATE": str(self.state), "AGENTPACE_PLAN_DIRS": str(self.plans), "AGENTPACE_CLAUDE_PROJECTS": str(self.claude), "AGENTPACE_CODEX_SESSIONS": str(self.codex)}
        out = io.StringIO()
        with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(out):
            code = cli.main(["learn", "--json"])
        self.assertEqual(code, 0)
        summary = json.loads(out.getvalue())
        self.assertEqual(summary["added"], 1)
        self.assertNotIn("CANARY", out.getvalue())


class CollapseAcrossSources(LearnCase):
    def test_codex_transcript_of_a_t3_run_is_dropped_at_read_time(self):
        # why: a T3 Codex run and its rollout file are one execution; counting both doubles
        #      history. T3 wins (it has the run id and usage-limit signals).
        # catches: no cross-source dedupe, or dropping the T3 row instead of the transcript.
        fx.add_t3_run(self.db, "r1", start=fx.T0, end=fx.T0 + 500, provider="codex", model="gpt-6.1-sol", native_id="nat-shared")
        fx.write_codex_rollout(self.codex, "c1", session_id="nat-shared", start=fx.T0, end=fx.T0 + 500)
        fx.write_codex_rollout(self.codex, "c2", session_id="nat-cli-only", start=fx.T0, end=fx.T0 + 300)
        self.learn()
        self.assertEqual(len(self.store()), 3)
        kept = learn.collapse(self.store())
        self.assertEqual(sorted((r["source"], r["actual_s"]) for r in kept), [("codex-transcript", 300.0), ("t3-database", 500.0)])

    def test_vigil_item_and_its_plan_task_merge_their_estimates(self):
        # why: a plan task closed from the vigil ledger is the same execution as the ledger item;
        #      merged, one row holds the plan P50 and the vigil P80.
        # catches: scoring the same run twice, or losing the P50 from the plan.
        start, end = fx.T0, fx.T0 + 900
        self.ledger.write_text(fx.vigil_line("h1", "sol-task", start=start, end=end, expect_s=1200, meta={"task": "plan-a/t1", "model": "sol", "kind": "implementation"}))
        ledger_id = learn.vigil_execution_key("h1", "sol-task", start)
        fx.write_plan(
            self.plans,
            "plan-a",
            [{"id": "t1", "type": "agent", "model": "sol", "kind": "implementation", "status": "done", "actual_s": 900, "started_at": start, "ended_at": end, "actual_source": "vigil-ledger", "forecast_s": 800}],
            events=[{"action": "ledger-done", "task": "t1", "at": end, "ledger_id": ledger_id}],
        )
        self.learn()
        self.assertEqual(len(self.store()), 2)
        (merged,) = learn.collapse(self.store())
        self.assertEqual((merged["estimate"]["p50"], merged["estimate"]["p80"]), (800.0, 1200.0))
        self.assertEqual(merged["estimate_source"], "plan-forecast+vigil-expect-s")


class SmallGuards(LearnCase):
    def test_collapse_leaves_its_input_untouched(self):
        # why: collapse filled the caller's nested estimate dict in place (a shallow copy), so
        #      reading the store twice gave different answers.
        # catches: dict(row) instead of a deep copy.
        vigil = learn.observation(source="vigil-ledger", key=("v",), start=0, end=100, model_raw="sol", kind="fix", kind_source="vigil-meta", label_source="vigil-hold", execution_key="E", estimate={"p80": 120})
        plan = learn.observation(source="plan", key=("p",), start=0, end=100, model_raw="sol", kind="fix", kind_source="plan", label_source="vigil-hold", execution_key="E", estimate={"p50": 90})
        before = json.dumps([vigil, plan], sort_keys=True)
        learn.collapse([vigil, plan])
        self.assertEqual(json.dumps([vigil, plan], sort_keys=True), before)

    def test_merge_keeps_plan_pointers_and_explicit_labels(self):
        # why: the vigil+plan merge kept only the vigil row's pointers and labels, so the plan
        #      task could not be opened and an explicit plan model lost to `unknown`.
        # catches: discarding the absorbed row's pointers, model or kind.
        vigil = learn.observation(source="vigil-ledger", key=("v",), start=0, end=100, model_raw="", kind="other", kind_source="none", label_source="vigil-hold", execution_key="E", pointers={"vigil_label": "x"})
        plan = learn.observation(source="plan", key=("p",), start=0, end=100, model_raw="claude-opus-5-5", kind="review", kind_source="plan", model_source="plan", label_source="vigil-hold", execution_key="E", pointers={"plan": "/p.json", "task": "t1"})
        (merged,) = learn.collapse([vigil, plan])
        self.assertEqual((merged["model"], merged["kind"], merged["model_source"], merged["kind_source"]), ("opus", "review", "plan", "plan"))
        self.assertEqual(merged["pointers"], {"vigil_label": "x", "plan": "/p.json", "task": "t1"})

    def test_bool_and_infinite_estimates_are_not_estimates(self):
        # why: `isinstance(x, (int, float)) and x > 0` accepted True (a 1 s P80) and infinity.
        # catches: any estimate check that admits bool or non-finite values.
        self.ledger.write_text(
            fx.vigil_line("h1", "opus-a", start=fx.T0, end=fx.T0 + 60).replace('"expect_s": null', '"expect_s": true')
            + fx.vigil_line("h2", "opus-b", start=fx.T0, end=fx.T0 + 60).replace('"expect_s": null', '"expect_s": Infinity')
        )
        self.learn()
        self.assertEqual([(r["estimate"]["p80"], r["estimate_source"]) for r in self.store()], [(None, None), (None, None)])

    def test_cursor_rename_is_followed_by_a_directory_sync(self):
        # why: os.replace is durable only once the directory entry is synced; without it a power
        #      loss can bring back the old cursor after the rows were counted as read.
        # catches: dropping the directory fsync after the rename.
        fx.add_t3_run(self.db, "r1", start=fx.T0, end=fx.T0 + 60)
        with mock.patch("agentpace.learn._fsync_directory", wraps=learn._fsync_directory) as synced:
            self.learn()
        synced.assert_called_with(self.state)


class FeedEstimates(LearnCase):
    def test_learned_rows_and_an_estimate_query_name_the_same_model(self):
        # why: learn stored `neon` while `estimate-run --model gemini-4-neon` queried
        #      `gemini-4-neon`, so the model weight never matched learned Neon history.
        # catches: learn and local_runs using different model normalizers.
        from agentpace import local_runs as lr

        for raw in ("gemini-4-neon", "claude-fable-5-2", "gpt-5.3-codex"):
            fx.add_t3_run(self.db, raw, start=fx.T0, end=fx.T0 + 60, model=raw, thread_id=f"thread:delegated-task:{raw}")
        self.learn()
        learned = {r["model_raw"]: r["model"] for r in learn.reference_rows(self.state, after=0)}
        queried = {raw: lr.make_row(start=0, end=1, model=raw)["model"] for raw in learned}
        self.assertEqual(learned, queried)
        self.assertEqual(sorted(learned.values()), ["codex", "fable", "neon"])

    def test_quantile_method_uses_learned_history_too(self):
        # why: `estimate-run --method quantile --learned` with the default --data loaded the
        #      fitted artifact's own reference rows and silently ignored --learned.
        # catches: deciding to refit from the --data path alone.
        from agentpace import local_runs as lr

        home = self.root / "home"
        (home / ".agentpace").mkdir(parents=True)
        replay = [lr.make_row(start=fx.T0 + i * 100, end=fx.T0 + i * 100 + 50, model="opus", kind="implementation", project="p", session=f"s{i}") for i in range(12)]
        lr.write_rows(home / ".agentpace" / "local-runs.jsonl", replay)
        (home / ".agentpace" / "quantile-model.json").write_text(json.dumps({"reference": replay[:3], "drop": []}))
        fx.add_t3_run(self.db, "new", start=fx.T0 + 5000, end=fx.T0 + 5600)
        self.learn()
        argv = ["estimate-run", "--model", "opus", "--kind", "implementation", "--method", "quantile", "--json", "--cwd", str(self.root), "--learned"]
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"AGENTPACE_STATE": str(self.state), "HOME": str(home), "AGENTPACE_HOME": str(home / ".agentpace")}), contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(argv), 0)
        self.assertEqual(json.loads(out.getvalue())["training_rows"], 13)

    def test_estimate_run_adds_only_learned_history_newer_than_the_replay(self):
        # why: "estimates use fresh history without a full replay": the learned store extends
        #      the replayed rows, and must not double-count what the replay already holds.
        # catches: concatenating the whole store (duplicates), or ignoring --learned.
        replay = self.root / "replay.jsonl"
        from agentpace import local_runs as lr

        rows = [lr.make_row(start=fx.T0 + i * 100, end=fx.T0 + i * 100 + 50, model="opus", kind="implementation", project="p", session=f"s{i}") for i in range(12)]
        lr.write_rows(replay, rows)
        fx.add_t3_run(self.db, "old", start=fx.T0, end=fx.T0 + 50)  # inside the replay window
        fx.add_t3_run(self.db, "new", start=fx.T0 + 5000, end=fx.T0 + 5600)  # after the replay watermark
        self.learn()
        out = io.StringIO()
        env = {"AGENTPACE_STATE": str(self.state)}
        argv = ["estimate-run", "--model", "opus", "--kind", "implementation", "--method", "reference", "--data", str(replay), "--json", "--cwd", str(self.root)]
        with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(out):
            cli.main(argv)
            baseline = json.loads(out.getvalue())["rows"]
            out.seek(0)
            out.truncate()
            cli.main(argv + ["--learned"])
            learned = json.loads(out.getvalue())["rows"]
        self.assertEqual((baseline, learned), (12, 13))


if __name__ == "__main__":
    unittest.main()
