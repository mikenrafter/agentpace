"""Between-run usage-limit gap pairing and summary tests (synthetic fixtures only)."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agentpace import activity_history as ah
from agentpace import usage_gap as ug


def _run(
    run_id: str,
    start: float,
    end: float | None,
    *,
    provider: str = "claude",
    thread_id: str | None = None,
    session_id: str | None = None,
    project: str | None = None,
    ordinal: int | None = None,
    status: str = "completed",
    usage_limit_end: bool = False,
    usage_limit_source: str | None = None,
    usage_limit_continuation_of: str | None = None,
    made_progress: bool | None = None,
    failure_class: str | None = None,
    failure_code: str | None = None,
    usage_limit_notice: bool = False,
    usage_limit_markers: int = 0,
    resumed_in_run: bool = False,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "start": start,
        "end": end,
        "provider": provider,
        "thread_id": thread_id,
        "session_id": session_id,
        "project": project,
        "ordinal": ordinal,
        "status": status,
        "usage_limit_end": usage_limit_end,
        "usage_limit_source": usage_limit_source,
        "usage_limit_continuation_of": usage_limit_continuation_of,
        "made_progress": made_progress,
        "failure_class": failure_class,
        "failure_code": failure_code,
        "usage_limit_notice": usage_limit_notice,
        "usage_limit_markers": usage_limit_markers,
        "resumed_in_run": resumed_in_run,
    }


class LimitDetectionTest(unittest.TestCase):
    def test_claude_monthly_spend_limit_is_detected_in_activity_history(self):
        # why: Claude CLI emits "You've hit your monthly spend limit ... resets ..."
        #      as a synthetic API-error assistant record; an older pattern missed it because
        #      LIMIT only allowed "usage " before "limit".
        # catches: regex that only matches "hit your usage limit" and misses Claude's
        #          monthly spend limit wording.
        # proves: CLAIM(limit-marker-channels)
        records = [
            {
                "timestamp": 100,
                "type": "assistant",
                "isApiErrorMessage": True,
                "message": {
                    "model": "<synthetic>",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You've hit your monthly spend limit · raise it at "
                                "https://claude.ai/settings/usage · your session limit resets 8am"
                            ),
                        }
                    ],
                },
            },
            {
                "timestamp": 110,
                "type": "assistant",
                "isApiErrorMessage": True,
                "message": {
                    "model": "<synthetic>",
                    "content": [
                        {
                            "type": "text",
                            "text": "Selected model is at capacity. Please try a different model.",
                        }
                    ],
                },
            },
            {
                "timestamp": 120,
                "type": "user",
                "message": {
                    "role": "user",
                    "content": "You've hit your monthly spend limit. Fix this string.",
                },
            },
        ]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "claude.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            trace = ah.parse_log(path)
        self.assertEqual([m["at"] for m in trace["markers"]], [100.0])

    def test_in_run_usage_wait_with_later_progress_is_not_a_usage_limit_end(self):
        # why: a run that pauses on a usage limit and resumes model progress inside the
        #      same run is an in-run wait, not a between-run usage-limit end.
        # catches: treating every run with usage_limit_markers > 0 as ending on a limit.
        # proves: CLAIM(usage-limit-end-terminal)
        in_run_trace = {
            "markers": [{"at": 150.0, "kind": "usage-limit"}],
            "progress": [110.0, 400.0],
        }
        ended_trace = {
            "markers": [{"at": 195.0, "kind": "usage-limit"}],
            "progress": [110.0, 180.0],
        }
        ended_in_run, reason_in_run = ug.run_ended_on_usage_limit(
            {"start": 100.0, "end": 500.0}, in_run_trace
        )
        ended_terminal, reason_terminal = ug.run_ended_on_usage_limit(
            {"start": 100.0, "end": 200.0}, ended_trace
        )
        self.assertFalse(ended_in_run)
        self.assertIsNone(reason_in_run)
        self.assertTrue(ended_terminal)
        self.assertEqual(reason_terminal, "transcript-limit-marker")

    def test_capacity_error_text_is_not_classified_as_usage_limit(self):
        # why: transient model capacity errors are not account usage-window limits.
        # catches: overly broad matching on "capacity" or generic API errors.
        # proves: CLAIM(limit-marker-channels)
        self.assertFalse(
            ug.is_usage_limit_text("Selected model is at capacity. Please try a different model.")
        )
        self.assertFalse(ug.is_usage_limit_text("API Error: 503 service unavailable"))
        self.assertTrue(
            ug.is_usage_limit_text(
                "You’ve hit your usage limit. Upgrade to Pro or try again at 3:00 PM."
            )
        )
        self.assertTrue(
            ug.is_usage_limit_text(
                "You've hit your monthly spend limit · your session limit resets 8am"
            )
        )


class PairingRulesTest(unittest.TestCase):
    def test_same_t3_thread_pairs_gap_from_first_end_to_resume_start(self):
        # why: primary pairing rule joins consecutive work in the same T3 thread;
        #      gap is resume.start - limit.end (2000 - 500 = 1500s).
        # catches: measuring start-to-start, end-to-end, or failing to match on thread_id.
        # proves: CLAIM(gap-definition)
        runs = [
            _run(
                "r1",
                100,
                500,
                thread_id="t1",
                ordinal=1,
                status="failed",
                failure_class="usage_limit",
            ),
            _run("r2", 2000, 2400, thread_id="t1", ordinal=2, status="completed"),
        ]
        result = ug.pair_usage_gaps(runs)
        self.assertEqual(len(result.pairs), 1)
        pair = result.pairs[0]
        self.assertEqual(pair.limit_run_id, "r1")
        self.assertEqual(pair.resume_run_id, "r2")
        self.assertEqual(pair.gap_s, 1500.0)
        self.assertEqual(pair.match_tier, "same-t3-thread")
        self.assertEqual(pair.strength, "strong")
        self.assertEqual(pair.classification, "usage-wait")
        self.assertEqual(pair.rejected_retries, 0)

    def test_explicit_continuation_link_certifies_limit_end_and_beats_other_tiers(self):
        # why: T3 writes usageLimitContinuationOfRunId on auto-resumed runs; this is
        #      the strongest signal even when the predecessor's trace is absent.
        # catches: ignoring usageLimitContinuationOfRunId or downgrading it to same-thread.
        # proves: CLAIM(tier-precedence)
        runs = [
            _run("r1", 0, 100, thread_id="t1", session_id="s1", ordinal=1, status="failed"),
            _run(
                "r2",
                1900,
                2100,
                thread_id="t1",
                session_id="s1",
                ordinal=2,
                usage_limit_continuation_of="r1",
            ),
        ]
        result = ug.pair_usage_gaps(runs)
        self.assertEqual(len(result.pairs), 1)
        pair = result.pairs[0]
        self.assertEqual(pair.gap_s, 1800.0)  # 1900 - 100
        self.assertEqual(pair.match_tier, "explicit-continuation")
        self.assertEqual(pair.strength, "strong")
        self.assertEqual(pair.limit_source, "t3-continuation-link")

    def test_tier_precedence_prefers_t3_thread_over_session_over_project_provider(self):
        # why: same T3 thread must take precedence over an unrelated run sharing a session
        #      or project, and same-session must beat weak project+provider matching.
        # catches: scanning purely by earliest timestamp across weak keys first.
        # proves: CLAIM(tier-precedence)
        runs = [
            # Limit-ended run with thread t1, session s1, project p1.
            _run(
                "a1",
                0,
                100,
                provider="claude",
                thread_id="t1",
                session_id="s1",
                project="p1",
                usage_limit_end=True,
            ),
            # Earlier unrelated run in same project p1 (different thread/session).
            _run("other_proj", 150, 200, provider="claude", thread_id="t9", session_id="s9", project="p1"),
            # Earlier run in same session s1 (different thread).
            _run("other_thread", 250, 300, provider="claude", thread_id="t2", session_id="s1", project="p1"),
            # Actual same-thread resume at t=600 -> gap = 600 - 100 = 500s.
            _run("a2", 600, 700, provider="claude", thread_id="t1", session_id="s1", project="p1"),
            # Standalone session-only limit run (no thread_id).
            _run("b1", 1000, 1100, provider="claude", session_id="s2", project="p1", usage_limit_end=True),
            # Earlier run in same project p1 (different session).
            _run("b_proj", 1200, 1250, provider="claude", session_id="s8", project="p1"),
            # Same-session resume at t=1700 -> gap = 1700 - 1100 = 600s.
            _run("b2", 1700, 1800, provider="claude", session_id="s2", project="p1"),
            # Standalone single-turn run (no thread_id, no later run in same session).
            _run("c1", 2000, 2100, provider="codex", session_id="cx1", project="p2", usage_limit_end=True),
            # Weak resume in same project+provider at t=5700 -> gap = 5700 - 2100 = 3600s.
            _run("c2", 5700, 6000, provider="codex", session_id="cx2", project="p2"),
        ]
        result = ug.pair_usage_gaps(runs)
        by_src = {p.limit_run_id: p for p in result.pairs}
        self.assertEqual((by_src["a1"].resume_run_id, by_src["a1"].match_tier, by_src["a1"].strength, by_src["a1"].gap_s), ("a2", "same-t3-thread", "strong", 500.0))
        self.assertEqual((by_src["b1"].resume_run_id, by_src["b1"].match_tier, by_src["b1"].strength, by_src["b1"].gap_s), ("b2", "same-session", "strong", 600.0))
        self.assertEqual((by_src["c1"].resume_run_id, by_src["c1"].match_tier, by_src["c1"].strength, by_src["c1"].gap_s), ("c2", "same-project-provider", "weak", 3600.0))

    def test_include_weak_false_leaves_project_only_matches_unresumed(self):
        # why: callers analyzing high-confidence pairs must be able to exclude weak matches.
        # catches: returning weak project+provider pairs when include_weak=False.
        runs = [
            _run("c1", 0, 100, provider="codex", session_id="cx1", project="p1", usage_limit_end=True),
            _run("c2", 3700, 4000, provider="codex", session_id="cx2", project="p1"),
        ]
        result = ug.pair_usage_gaps(runs, include_weak=False)
        self.assertEqual(result.pairs, ())
        self.assertEqual(result.unresumed_run_ids, ("c1",))

    def test_weak_match_enforces_one_to_one_claiming_for_parallel_workers(self):
        # why: two parallel workers hitting a usage limit simultaneously in one project
        #      cannot both claim a single subsequent run as their resume.
        # catches: double-counting one resumed run across multiple simultaneous limit ends.
        # proves: CLAIM(one-to-one-weak-claim)
        runs = [
            _run("w1", 0, 100, provider="codex", session_id="s1", project="p1", usage_limit_end=True),
            _run("w2", 1, 101, provider="codex", session_id="s2", project="p1", usage_limit_end=True),
            _run("r1", 12100, 12500, provider="codex", session_id="s3", project="p1"),
        ]
        result = ug.pair_usage_gaps(runs)
        self.assertEqual(len(result.pairs), 1)
        self.assertEqual(result.pairs[0].limit_run_id, "w1")
        self.assertEqual(result.pairs[0].resume_run_id, "r1")
        self.assertEqual(result.pairs[0].gap_s, 12000.0)
        self.assertEqual(result.unresumed_run_ids, ("w2",))

    def test_weak_match_requires_matching_provider(self):
        # why: switching from Codex to Claude in the same repo is a provider switch,
        #      not a Codex usage-window reset wait.
        # catches: matching across different providers on project alone.
        runs = [
            _run("cx1", 0, 100, provider="codex", session_id="s1", project="p1", usage_limit_end=True),
            _run("cl1", 200, 400, provider="claude", session_id="s2", project="p1"),
        ]
        result = ug.pair_usage_gaps(runs)
        self.assertEqual(result.pairs, ())
        self.assertEqual(result.unresumed_run_ids, ("cx1",))

    def test_owner_away_upper_bound_is_inclusive_at_exact_cap(self):
        # why: gaps strictly above max_gap_s are owner-away, not usage waits;
        #      at exactly max_gap_s (43200s) a gap remains a usage-wait, matching
        #      the >43200s convention.
        # catches: off-by-one (< vs <=), silently dropping owner-away gaps, or
        #          including owner-away gaps in usage_waits / quantiles.
        # proves: CLAIM(owner-away-cap)
        runs = [
            _run("a1", 0, 100, thread_id="t1", ordinal=1, usage_limit_end=True),
            _run("a2", 43300, 43400, thread_id="t1", ordinal=2),  # gap = 43200.0 (exact cap)
            _run("b1", 0, 100, thread_id="t2", ordinal=1, usage_limit_end=True),
            _run("b2", 43301, 43400, thread_id="t2", ordinal=2),  # gap = 43201.0 (owner-away)
        ]
        result = ug.pair_usage_gaps(runs, max_gap_s=43200.0)
        self.assertEqual([p.limit_run_id for p in result.usage_waits], ["a1"])
        self.assertEqual(result.usage_waits[0].gap_s, 43200.0)
        self.assertEqual(result.usage_waits[0].classification, "usage-wait")
        self.assertEqual([p.limit_run_id for p in result.owner_away], ["b1"])
        self.assertEqual(result.owner_away[0].gap_s, 43201.0)
        self.assertEqual(result.owner_away[0].classification, "owner-away")

    def test_rejected_retry_chain_collapses_into_one_outage_episode(self):
        # why: retrying a thread before reset triggers instant 0-progress limit failures;
        #      pairing adjacent attempts splits one 3000s wait into short fragments.
        # catches: emitting 3 fragments instead of 1 episode (gap = 3100 - 100 = 3000s)
        #          or losing the explicit continuation tier at the chain's tail.
        # proves: CLAIM(retry-chain-collapse)
        runs = [
            _run("r1", 0, 100, thread_id="t1", ordinal=1, status="failed", failure_class="usage_limit", made_progress=True),
            _run("r2", 700, 702, thread_id="t1", ordinal=2, status="failed", failure_class="usage_limit", made_progress=False),
            _run("r3", 1900, 1901, thread_id="t1", ordinal=3, status="failed", failure_class="usage_limit", made_progress=False),
            _run(
                "r4",
                3100,
                3500,
                thread_id="t1",
                ordinal=4,
                status="completed",
                usage_limit_continuation_of="r3",
                made_progress=True,
            ),
        ]
        collapsed = ug.pair_usage_gaps(runs, collapse_rejected_retries=True)
        self.assertEqual(len(collapsed.pairs), 1)
        pair = collapsed.pairs[0]
        self.assertEqual(pair.limit_run_id, "r1")
        self.assertEqual(pair.resume_run_id, "r4")
        self.assertEqual(pair.gap_s, 3000.0)  # 3100 - 100
        self.assertEqual(pair.last_attempt_gap_s, 1199.0)  # 3100 - 1901
        self.assertEqual(pair.rejected_retries, 2)
        self.assertEqual(pair.rejected_retry_ids, ("r2", "r3"))
        self.assertEqual(pair.match_tier, "explicit-continuation")

        # Without collapsing, all 3 adjacent pairs appear with exact arithmetic.
        adjacent = ug.pair_usage_gaps(runs, collapse_rejected_retries=False)
        self.assertEqual([p.gap_s for p in adjacent.pairs], [600.0, 1198.0, 1199.0])

    def test_resumed_run_that_makes_progress_before_hitting_new_limit_starts_second_episode(self):
        # why: if run 2 actually resumed work (made_progress=True) and later hit a new
        #      usage limit, it closes episode 1 and opens episode 2.
        # catches: collapsing a productive run just because it also ended on a usage limit.
        # proves: CLAIM(retry-chain-collapse)
        runs = [
            _run("r1", 0, 100, thread_id="t1", ordinal=1, usage_limit_end=True, made_progress=True),
            _run("r2", 1100, 2000, thread_id="t1", ordinal=2, usage_limit_end=True, made_progress=True),
            _run("r3", 4500, 5000, thread_id="t1", ordinal=3, made_progress=True),
        ]
        result = ug.pair_usage_gaps(runs)
        self.assertEqual(
            [(p.limit_run_id, p.resume_run_id, p.gap_s) for p in result.pairs],
            [("r1", "r2", 1000.0), ("r2", "r3", 2500.0)],
        )

    def test_input_order_does_not_affect_pairing_results(self):
        # why: callers may merge runs from multiple sources in arbitrary order.
        # catches: relying on input sequence order instead of sorting chronologically.
        # proves: CLAIM(deterministic-order)
        ordered = [
            _run("r1", 10, 100, thread_id="t1", ordinal=1, usage_limit_end=True),
            _run("r2", 700, 900, thread_id="t1", ordinal=2),
            _run("r3", 200, 300, thread_id="t2", ordinal=1, usage_limit_end=True),
            _run("r4", 1500, 1600, thread_id="t2", ordinal=2),
        ]
        shuffled = [ordered[3], ordered[1], ordered[2], ordered[0]]
        self.assertEqual(ug.pair_usage_gaps(ordered), ug.pair_usage_gaps(shuffled))

    def test_empty_and_invalid_inputs_behave_explicitly(self):
        # why: absent input returns empty results with None quantiles; invalid spans
        #      or non-positive max_gap_s fail loudly at the boundary.
        # catches: crashing on empty input, fabricating 0.0 quantiles, or silently
        #          accepting end < start or max_gap_s <= 0.
        empty = ug.pair_usage_gaps([])
        self.assertEqual(empty.pairs, ())
        self.assertEqual(empty.unresumed_run_ids, ())
        summary = ug.summarize_gaps(empty)
        self.assertEqual(summary.overall.count, 0)
        self.assertIsNone(summary.overall.p50_s)
        self.assertIsNone(summary.overall.p80_s)
        self.assertIsNone(summary.overall.p90_s)

        with self.assertRaises(ValueError):
            ug.pair_usage_gaps([], max_gap_s=0)
        with self.assertRaises(ValueError):
            ug.pair_usage_gaps([], max_gap_s=-10)
        with self.assertRaises(ValueError):
            ug.pair_usage_gaps([_run("bad", 200, 100)])


class SummaryAndNormalizationTest(unittest.TestCase):
    def test_summary_splits_by_provider_and_computes_exact_quantiles(self):
        # why: replay must report count and P50/P80/P90 overall and by provider
        #      (Claude vs Codex vs other), excluding owner-away gaps from wait quantiles.
        # catches: pooling providers, mis-normalizing claudeAgent/openai, or leaking
        #          owner-away values into P50/P80/P90.
        # Arithmetic (Hyndman-Fan type 7, matching agentpace.estimate.quantile):
        #   claude gaps = [100, 200, 300, 400, 500]
        #     pos(0.5) = 2.0 -> 300.0
        #     pos(0.8) = 3.2 -> 400 + 0.2*(500-400) = 420.0
        #     pos(0.9) = 3.6 -> 400 + 0.6*(500-400) = 460.0
        #   codex gaps = [1000, 2000]
        #     pos(0.5) = 0.5 -> 1500.0, pos(0.8) = 0.8 -> 1800.0, pos(0.9) = 0.9 -> 1900.0
        #   other gaps = [600] -> 600.0 / 600.0 / 600.0
        runs = []
        for idx, gap in enumerate((100, 200, 300, 400, 500), 1):
            runs.append(_run(f"cl{idx}a", idx * 10000, idx * 10000 + 10, provider="claudeAgent", thread_id=f"tc{idx}", usage_limit_end=True))
            runs.append(_run(f"cl{idx}b", idx * 10000 + 10 + gap, idx * 10000 + 20 + gap, provider="claudeAgent", thread_id=f"tc{idx}"))
        # Add one owner-away claude pair (50000s > 43200s) that must NOT shift quantiles.
        runs.append(_run("cl_away_a", 0, 10, provider="claude", thread_id="tc_away", usage_limit_end=True))
        runs.append(_run("cl_away_b", 50010, 50100, provider="claude", thread_id="tc_away"))

        for idx, gap in enumerate((1000, 2000), 1):
            runs.append(_run(f"cx{idx}a", idx * 10000, idx * 10000 + 10, provider="codex", session_id=f"sx{idx}", usage_limit_end=True))
            runs.append(_run(f"cx{idx}b", idx * 10000 + 10 + gap, idx * 10000 + 20 + gap, provider="codex", session_id=f"sx{idx}"))

        runs.append(_run("ot1a", 100, 110, provider="gemini", session_id="so1", usage_limit_end=True))
        runs.append(_run("ot1b", 710, 800, provider="gemini", session_id="so1"))

        paired = ug.pair_usage_gaps(runs, max_gap_s=43200.0)
        summary = ug.summarize_gaps(paired)
        self.assertEqual(summary.overall.count, 8)
        self.assertEqual(summary.owner_away_count, 1)
        self.assertEqual(summary.by_provider["claude"].count, 5)
        self.assertAlmostEqual(summary.by_provider["claude"].p50_s, 300.0)
        self.assertAlmostEqual(summary.by_provider["claude"].p80_s, 420.0)
        self.assertAlmostEqual(summary.by_provider["claude"].p90_s, 460.0)
        self.assertEqual(summary.by_provider["codex"].count, 2)
        self.assertAlmostEqual(summary.by_provider["codex"].p50_s, 1500.0)
        self.assertAlmostEqual(summary.by_provider["codex"].p80_s, 1800.0)
        self.assertAlmostEqual(summary.by_provider["codex"].p90_s, 1900.0)
        self.assertEqual(summary.by_provider["other"].count, 1)
        self.assertAlmostEqual(summary.by_provider["other"].p50_s, 600.0)


class LoadersTest(unittest.TestCase):
    def test_load_t3_runs_is_read_only_hashes_project_and_reads_limit_signals(self):
        # why: T3 database reads must never write/create the DB, must never emit raw
        #      workspace paths or prompts, and must surface usage_limit error items,
        #      system notices, and usageLimitContinuationOfRunId links.
        # catches: opening DB read-write, leaking workspace paths, or missing T3's
        #          structured usage_limit signals.
        # proves: CLAIM(t3-readonly-loader)
        with tempfile.TemporaryDirectory() as root:
            missing = Path(root) / "missing.sqlite"
            with self.assertRaises(sqlite3.OperationalError):
                ug.load_t3_runs(missing)
            self.assertFalse(missing.exists())

            db_path = Path(root) / "statev2.sqlite"
            con = sqlite3.connect(db_path)
            con.executescript(
                """
                CREATE TABLE orchestration_v2_projection_runs(
                    run_id TEXT, thread_id TEXT, ordinal INTEGER, provider TEXT,
                    status TEXT, requested_at TEXT, completed_at TEXT, payload_json TEXT
                );
                CREATE TABLE orchestration_v2_projection_threads(thread_id TEXT, project_id TEXT);
                CREATE TABLE projection_projects(project_id TEXT, workspace_root TEXT);
                CREATE TABLE orchestration_v2_projection_provider_threads(provider_thread_id TEXT, payload_json TEXT);
                CREATE TABLE orchestration_v2_projection_run_attempts(payload_json TEXT);
                CREATE TABLE orchestration_v2_projection_turn_items(
                    run_id TEXT, type TEXT, status TEXT, payload_json TEXT
                );
                """
            )
            con.execute("INSERT INTO projection_projects VALUES('p1', '/secret/owner/repo')")
            con.execute("INSERT INTO orchestration_v2_projection_threads VALUES('t1', 'p1')")
            con.execute(
                "INSERT INTO orchestration_v2_projection_provider_threads VALUES('pt1', ?)",
                (json.dumps({"nativeThreadRef": {"nativeId": "nat-1"}}),),
            )
            con.execute(
                "INSERT INTO orchestration_v2_projection_runs VALUES(?,?,?,?,?,?,?,?)",
                (
                    "run-1",
                    "t1",
                    1,
                    "claudeAgent",
                    "failed",
                    "2026-11-11T07:00:00Z",
                    "2026-11-11T07:10:00Z",
                    json.dumps(
                        {
                            "providerThreadId": "pt1",
                            "startedAt": "2026-11-11T07:00:00Z",
                            "completedAt": "2026-11-11T07:10:00Z",
                            "modelSelection": {"model": "claude-opus-4-6"},
                        }
                    ),
                ),
            )
            con.execute(
                "INSERT INTO orchestration_v2_projection_runs VALUES(?,?,?,?,?,?,?,?)",
                (
                    "run-2",
                    "t1",
                    2,
                    "claudeAgent",
                    "completed",
                    "2026-11-11T07:40:00Z",
                    "2026-11-11T07:45:00Z",
                    json.dumps(
                        {
                            "providerThreadId": "pt1",
                            "startedAt": "2026-11-11T07:40:00Z",
                            "completedAt": "2026-11-11T07:45:00Z",
                            "usageLimitContinuationOfRunId": "run-1",
                            "modelSelection": {"model": "claude-opus-4-6"},
                        }
                    ),
                ),
            )
            con.execute(
                "INSERT INTO orchestration_v2_projection_turn_items VALUES(?,?,?,?)",
                (
                    "run-1",
                    "error",
                    "failed",
                    json.dumps({"failure": {"class": "usage_limit", "code": "api_error_429"}}),
                ),
            )
            con.execute(
                "INSERT INTO orchestration_v2_projection_turn_items VALUES(?,?,?,?)",
                ("run-1", "reasoning", "completed", json.dumps({})),
            )
            con.commit()
            con.close()
            before = db_path.read_bytes()

            loaded = ug.load_t3_runs(db_path)
            self.assertEqual(db_path.read_bytes(), before)

        self.assertEqual(len(loaded), 2)
        self.assertNotIn("/secret/owner/repo", json.dumps([r.to_dict() for r in loaded]))
        paired = ug.pair_usage_gaps(loaded)
        self.assertEqual(len(paired.pairs), 1)
        self.assertEqual(paired.pairs[0].gap_s, 1800.0)  # 07:40:00 - 07:10:00
        self.assertEqual(paired.pairs[0].match_tier, "explicit-continuation")
        self.assertEqual(paired.pairs[0].provider, "claude")

    def test_load_claude_transcript_runs_ignores_unanswered_queued_user_records(self):
        # why: Claude transcripts sometimes flush a queued user/task_notification record
        #      at +0.0s after a usage limit error with zero assistant response; treating
        #      that as a resumed run fabricates a 0.0s wait.
        # catches: pairing a limit turn with an unanswered queued user record.
        # proves: CLAIM(claude-turn-progress)
        records = [
            {"timestamp": 100, "type": "user", "cwd": "/tmp/demo", "message": {"role": "user", "content": "do task"}},
            {"timestamp": 110, "type": "assistant", "message": {"model": "claude-sonnet-4-6", "content": [{"type": "text", "text": "working"}]}},
            {
                "timestamp": 200,
                "type": "assistant",
                "isApiErrorMessage": True,
                "message": {
                    "model": "<synthetic>",
                    "content": [{"type": "text", "text": "You've hit your monthly spend limit · your session limit resets 8am"}],
                },
            },
            # Unanswered queued user record at +0.0s — must NOT count as a resume!
            {"timestamp": 200, "type": "user", "cwd": "/tmp/demo", "message": {"role": "user", "content": "queued"}},
            # Real resumed turn at t=2000 with real assistant progress -> gap = 2000 - 200 = 1800s.
            {"timestamp": 2000, "type": "user", "cwd": "/tmp/demo", "message": {"role": "user", "content": "continue"}},
            {"timestamp": 2010, "type": "assistant", "message": {"model": "claude-sonnet-4-6", "content": [{"type": "text", "text": "done"}]}},
        ]
        with tempfile.TemporaryDirectory() as root:
            proj = Path(root) / "proj-a"
            proj.mkdir()
            (proj / "sess-1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            runs = ug.load_claude_transcript_runs(Path(root))
            excluded = ug.load_claude_transcript_runs(Path(root), exclude_sessions={"sess-1"})

        self.assertEqual(excluded, [])
        paired = ug.pair_usage_gaps(runs)
        self.assertEqual(len(paired.pairs), 1)
        self.assertEqual(paired.pairs[0].gap_s, 1800.0)
        self.assertEqual(paired.pairs[0].match_tier, "same-session")
        self.assertEqual(paired.pairs[0].provider, "claude")

    def test_load_codex_transcript_runs_extracts_limit_turns_and_skips_forks(self):
        # why: Codex logs "You’ve hit your usage limit..." on task_complete events;
        #      forked rollouts duplicate parent history and must be skipped.
        # catches: missing Codex task_complete limit messages or double-counting forks.
        # proves: CLAIM(codex-turn-extraction)
        main_records = [
            {"timestamp": 100, "type": "session_meta", "payload": {"id": "cx-1", "cwd": "/tmp/repo"}},
            {"timestamp": 100, "type": "event_msg", "payload": {"type": "task_started", "turn_id": "t1"}},
            {"timestamp": 150, "type": "response_item", "payload": {"type": "reasoning"}},
            {
                "timestamp": 200,
                "type": "event_msg",
                "payload": {
                    "type": "task_complete",
                    "turn_id": "t1",
                    "last_agent_message": "You’ve hit your usage limit. Upgrade to Pro or try again at 4:00 PM.",
                },
            },
            {"timestamp": 3800, "type": "event_msg", "payload": {"type": "task_started", "turn_id": "t2"}},
            {"timestamp": 3850, "type": "response_item", "payload": {"type": "reasoning"}},
            {"timestamp": 3900, "type": "event_msg", "payload": {"type": "task_complete", "turn_id": "t2", "last_agent_message": "finished"}},
        ]
        fork_records = [
            {"timestamp": 100, "type": "session_meta", "payload": {"id": "cx-fork", "cwd": "/tmp/repo", "forked_from_id": "cx-1"}},
            *main_records[1:],
        ]
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "rollout-2026-11-01-cx-1.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in main_records), encoding="utf-8"
            )
            (Path(root) / "rollout-2026-11-01-cx-fork.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in fork_records), encoding="utf-8"
            )
            runs = ug.load_codex_transcript_runs(Path(root))

        self.assertEqual(len(runs), 2)
        paired = ug.pair_usage_gaps(runs)
        self.assertEqual(len(paired.pairs), 1)
        self.assertEqual(paired.pairs[0].gap_s, 3600.0)  # 3800 - 200
        self.assertEqual(paired.pairs[0].match_tier, "same-session")
        self.assertEqual(paired.pairs[0].provider, "codex")

    def test_claude_queue_operations_do_not_stretch_limit_turn_end(self):
        # why: Claude writes `queue-operation` and `attachment` records hours later
        #      just before the next user prompt; letting non-execution records advance
        #      turn.end erases a multi-hour usage wait down to 0.0s.
        # catches: updating turn.end on every JSONL record type instead of execution
        #          records (assistant / tool_result).
        # proves: CLAIM(claude-turn-end-execution-only)
        records = [
            {"timestamp": 100, "type": "user", "cwd": "/tmp/demo", "message": {"role": "user", "content": "start"}},
            {
                "timestamp": 200,
                "type": "assistant",
                "isApiErrorMessage": True,
                "message": {
                    "model": "<synthetic>",
                    "content": [{"type": "text", "text": "You've hit your monthly spend limit · your session limit resets 8am"}],
                },
            },
            # Hours later, queue-operation + attachment arrive right before the resume prompt.
            {"timestamp": 10200, "type": "queue-operation"},
            {"timestamp": 10200, "type": "attachment"},
            {"timestamp": 10200, "type": "user", "cwd": "/tmp/demo", "message": {"role": "user", "content": "resume"}},
            {"timestamp": 10220, "type": "assistant", "message": {"model": "claude-opus-4-6", "content": [{"type": "text", "text": "ok"}]}},
        ]
        with tempfile.TemporaryDirectory() as root:
            proj = Path(root) / "proj-q"
            proj.mkdir()
            (proj / "sess-q.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            runs = ug.load_claude_transcript_runs(Path(root))
        paired = ug.pair_usage_gaps(runs)
        self.assertEqual(len(paired.pairs), 1)
        self.assertEqual(paired.pairs[0].gap_s, 10000.0)  # 10200 - 200, NOT 0.0

    def test_claude_synthetic_no_response_turn_does_not_count_as_resume(self):
        # why: Claude can emit a `<synthetic>` "No response requested." assistant stub
        #      on a queued notification right after a usage limit; with zero real model
        #      progress it did not resume the work.
        # catches: treating `<synthetic>` non-limit stubs as resumed turns.
        # proves: CLAIM(claude-turn-progress)
        records = [
            {"timestamp": 100, "type": "user", "cwd": "/tmp/demo", "message": {"role": "user", "content": "start"}},
            {
                "timestamp": 200,
                "type": "assistant",
                "isApiErrorMessage": True,
                "message": {
                    "model": "<synthetic>",
                    "content": [{"type": "text", "text": "You've hit your monthly spend limit · your session limit resets 8am"}],
                },
            },
            {"timestamp": 300, "type": "user", "cwd": "/tmp/demo", "message": {"role": "user", "content": "notif"}},
            {"timestamp": 301, "type": "assistant", "message": {"model": "<synthetic>", "content": [{"type": "text", "text": "No response requested."}]}},
            {"timestamp": 5200, "type": "user", "cwd": "/tmp/demo", "message": {"role": "user", "content": "resume"}},
            {"timestamp": 5220, "type": "assistant", "message": {"model": "claude-opus-4-6", "content": [{"type": "text", "text": "done"}]}},
        ]
        with tempfile.TemporaryDirectory() as root:
            proj = Path(root) / "proj-s"
            proj.mkdir()
            (proj / "sess-s.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            runs = ug.load_claude_transcript_runs(Path(root))
        paired = ug.pair_usage_gaps(runs)
        self.assertEqual(len(paired.pairs), 1)
        self.assertEqual(paired.pairs[0].gap_s, 5000.0)  # 5200 - 200, NOT 100.0

    def test_claude_compaction_duplicate_limit_turn_is_not_double_counted(self):
        # why: conversation compaction can replay an earlier user+limit turn at the
        #      same timestamp later in the JSONL; both copies must not claim the same resume.
        # catches: emitting duplicate usage-gap pairs for one compacted limit turn.
        # proves: CLAIM(claude-dedupe-compacted-turns)
        records = [
            {"timestamp": 100, "type": "user", "cwd": "/tmp/demo", "message": {"role": "user", "content": "start"}},
            {
                "timestamp": 101,
                "type": "assistant",
                "isApiErrorMessage": True,
                "message": {
                    "model": "<synthetic>",
                    "content": [{"type": "text", "text": "You've hit your monthly spend limit · your session limit resets 8am"}],
                },
            },
            {"timestamp": 3701, "type": "user", "cwd": "/tmp/demo", "message": {"role": "user", "content": "resume"}},
            {"timestamp": 3720, "type": "assistant", "message": {"model": "claude-opus-4-6", "content": [{"type": "text", "text": "done"}]}},
            # Compaction later in the file replays the earlier limit turn verbatim.
            {"timestamp": 4000, "type": "system", "subtype": "compact_boundary"},
            {"timestamp": 100, "type": "user", "cwd": "/tmp/demo", "message": {"role": "user", "content": "start"}},
            {
                "timestamp": 101,
                "type": "assistant",
                "isApiErrorMessage": True,
                "message": {
                    "model": "<synthetic>",
                    "content": [{"type": "text", "text": "You've hit your monthly spend limit · your session limit resets 8am"}],
                },
            },
        ]
        with tempfile.TemporaryDirectory() as root:
            proj = Path(root) / "proj-c"
            proj.mkdir()
            (proj / "sess-c.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            runs = ug.load_claude_transcript_runs(Path(root))
        paired = ug.pair_usage_gaps(runs)
        self.assertEqual(len(paired.pairs), 1)
        self.assertEqual(paired.pairs[0].gap_s, 3600.0)  # 3701 - 101, single pair


class CliUsageGapsTest(unittest.TestCase):
    def test_cli_usage_gaps_json_reports_by_provider_on_synthetic_dirs(self):
        # why: CLI `agentpace usage-gaps --json` must wire loaders, pairing, and summary
        #      end-to-end without touching the owner's real ~/.t3 or transcript dirs.
        # catches: missing subcommand wiring, broken flag forwarding (--max-gap-s /
        #          --strong-only), or failing when default T3 DB is absent.
        # proves: CLAIM(cli-usage-gaps)
        import contextlib
        import io
        from agentpace import cli

        claude_records = [
            {"timestamp": 100, "type": "user", "cwd": "/tmp/p", "message": {"role": "user", "content": "hi"}},
            {
                "timestamp": 200,
                "type": "assistant",
                "isApiErrorMessage": True,
                "message": {
                    "model": "<synthetic>",
                    "content": [{"type": "text", "text": "You've hit your monthly spend limit · your session limit resets 8am"}],
                },
            },
            {"timestamp": 1400, "type": "user", "cwd": "/tmp/p", "message": {"role": "user", "content": "go"}},
            {"timestamp": 1450, "type": "assistant", "message": {"model": "claude-opus-4-6", "content": [{"type": "text", "text": "ok"}]}},
        ]
        with tempfile.TemporaryDirectory() as root:
            claude_dir = Path(root) / "claude" / "proj"
            claude_dir.mkdir(parents=True)
            (claude_dir / "s1.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in claude_records), encoding="utf-8"
            )
            codex_dir = Path(root) / "codex"
            codex_dir.mkdir()
            missing_t3 = Path(root) / "no-t3.sqlite"

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli.main(
                    [
                        "usage-gaps",
                        "--no-t3",
                        "--claude-dir",
                        str(Path(root) / "claude"),
                        "--codex-dir",
                        str(codex_dir),
                        "--max-gap-s",
                        "3600",
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["summary"]["overall"]["count"], 1)
            self.assertEqual(payload["summary"]["by_provider"]["claude"]["p50_s"], 1200.0)
            self.assertEqual(payload["summary"]["by_provider"]["codex"]["count"], 0)
            self.assertEqual(payload["summary"]["by_provider"]["other"]["count"], 0)

            # Explicitly passing a non-existent --t3-db must fail with exit code 2.
            err_buf = io.StringIO()
            with contextlib.redirect_stderr(err_buf):
                rc_err = cli.main(
                    [
                        "usage-gaps",
                        "--t3-db",
                        str(missing_t3),
                        "--claude-dir",
                        str(Path(root) / "claude"),
                        "--codex-dir",
                        str(codex_dir),
                    ]
                )
            self.assertEqual(rc_err, 2)


if __name__ == "__main__":
    unittest.main()
