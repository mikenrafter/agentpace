"""The estimator's own estimate and the run id travel with a dispatch.

`estimate-run --vigil-meta` prints the flags an orchestrator appends to `vigil hold add`; `learn`
links a hold carrying `meta.run` to that run by id and scores `meta.p50`/`meta.p80` as the
estimate, keeping a hand-set `--expect-s` as a separate `ceiling`; `calibration` reports the
ceiling apart from P80 coverage. SYNTHETIC fixtures only (tests/learn_fixtures.py).

Mated parts: `local_runs.cmd_estimate_run` writes the meta keys, `learn.read_vigil` reads them.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import shlex
import shutil
import unittest
from unittest import mock

from agentpace import __version__, calibration, cli, learn
from agentpace import local_runs as lr
from tests import learn_fixtures as fx
from tests import test_learn

HOUR = 3600.0
DAY = 86_400.0
NOW = fx.T0 + 10 * HOUR


def vigil_flags(argv: list[str], data) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["estimate-run", "--method", "reference", "--data", str(data), *argv])
    return code, out.getvalue(), err.getvalue()


def meta_of(tokens: list[str]) -> dict[str, str]:
    """The `--meta k=v` pairs of a flag line, as vigil stores them."""
    return dict(tokens[i + 1].split("=", 1) for i, token in enumerate(tokens) if token == "--meta")


class FlagCase(test_learn.LearnCase):
    def history(self, model, kind, actuals):
        path = self.root / "rows.jsonl"
        lr.write_rows(path, [lr.make_row(start=i * 10_000, end=i * 10_000 + a, model=model, kind=kind, session=str(i)) for i, a in enumerate(actuals)])
        return path


class VigilMetaFlags(FlagCase):

    def test_flags_are_exactly_what_vigil_hold_add_takes(self):
        # why: the orchestrator pastes this line after `vigil hold add`; every flag must parse and
        #      carry the number learn scores later. Twenty 600 s runs: P50 = P80 = 600 by hand.
        # catches: a different flag name, a missing --meta, or the P80 not equal to --expect-s.
        data = self.history("opus", "implementation", [600] * 20)
        code, out, _ = vigil_flags(["--model", "opus", "--kind", "implementation", "--vigil-meta"], data)
        self.assertEqual(code, 0)
        self.assertEqual(
            shlex.split(out),
            ["--expect-s", "600", "--meta", "p50=600", "--meta", "p80=600", "--meta", f"est=agentpace-{__version__}", "--meta", "model=opus", "--meta", "kind=implementation"],
        )
        self.assertEqual(out.count("\n"), 1, "one line, so $(...) substitution stays one shell word list")

    def test_p50_and_p80_are_the_estimators_own_quantiles(self):
        # why: p50 and p80 must differ when the history does; a P80 printed twice would score
        #      every run as a P50 hit. Runs of 100..2000 s in steps of 100: the P80 sits above the P50.
        # catches: --vigil-meta printing one number for both, or a different P80 than --expect-s.
        data = self.history("sol", "review", list(range(100, 2100, 100)))
        _, flags, _ = vigil_flags(["--model", "sol", "--kind", "review", "--vigil-meta"], data)
        _, expect_s, _ = vigil_flags(["--model", "sol", "--kind", "review", "--expect-s"], data)
        _, as_json, _ = vigil_flags(["--model", "sol", "--kind", "review", "--json"], data)
        result = json.loads(as_json)
        tokens = shlex.split(flags)
        meta = meta_of(tokens)
        self.assertEqual(tokens[1], expect_s.strip())
        self.assertEqual((int(meta["p50"]), int(meta["p80"])), (math.ceil(result["p50"] - 1e-9), math.ceil(result["p80"] - 1e-9)))
        self.assertLess(int(meta["p50"]), int(meta["p80"]))

    def test_model_and_kind_are_the_normalized_names_learn_stores(self):
        # why: learn joins hold meta to history by the family name; a raw `gemini-4-neon` or
        #      `impl` would split Neon's implementation runs into groups with no history.
        # catches: printing --model and --kind as typed.
        data = self.history("neon", "implementation", [300] * 12)
        _, out, _ = vigil_flags(["--model", "gemini-4-neon", "--kind", "impl", "--vigil-meta"], data)
        self.assertIn("--meta model=neon --meta kind=implementation", out)

    def test_it_refuses_to_mix_with_the_other_output_forms(self):
        # why: --json and --expect-s print their own shape; a line of flags inside a JSON
        #      document (or after a bare integer) is unusable in $(...).
        # catches: silently printing one form and dropping the other.
        data = self.history("opus", "fix", [60] * 12)
        for other in ("--json", "--expect-s"):
            code, out, err = vigil_flags(["--model", "opus", "--kind", "fix", "--vigil-meta", other], data)
            self.assertEqual((code, out), (2, ""), other)
            self.assertIn("--vigil-meta", err)

    def test_a_flag_line_scores_back_through_learn(self):
        # why: the producer and the reader are mated parts; the flags the CLI prints, put on a
        #      hold, must be scored by learn as that run's P50/P80 with source agentpace.
        # catches: a key one side spells differently (p50 vs p_50, est vs estimator).
        data = self.history("opus", "implementation", [600] * 20)
        _, out, _ = vigil_flags(["--model", "opus", "--kind", "implementation", "--vigil-meta"], data)
        tokens = shlex.split(out)
        meta = meta_of(tokens)
        fx.add_t3_run(self.db, "R1", start=fx.T0, end=fx.T0 + 700, thread_id="thread:delegated-task:child", model="claude-opus-5-5")
        self.ledger.write_text(fx.vigil_line("W", "opus-impl", start=fx.T0 + 5, end=fx.T0 + 800, expect_s=int(tokens[1]), meta={**meta, "run": "child"}))
        self.learn()
        (row,) = learn.collapse(self.store())
        self.assertEqual((row["source"], row["estimate"]["p50"], row["estimate"]["p80"], row["estimate_source"]), ("t3-database", 600.0, 600.0, "agentpace"))
        self.assertEqual(row["estimator"], f"agentpace-{__version__}")


class IdLinkCase(test_learn.LearnCase):
    THREAD = "thread:delegated-task:command%3Amcp%3Aabc%3Adelegate-task%3Atask-1"
    OTHER = "thread:delegated-task:command%3Amcp%3Aabc%3Adelegate-task%3Atask-2"

    def hold(self, label, run, *, start=fx.T0 + 5, end=fx.T0 + 1500, expect_s=1200, **meta):
        extra = {"model": "opus", "kind": "implementation", "p50": "500", "p80": "1000", "est": "agentpace-0.13.1", **meta}
        if run is not None:
            extra["run"] = run
        return fx.vigil_line("W", label, start=start, end=end, expect_s=expect_s, meta=extra).replace('"session_id": "s-orchestrator"', '"session_id": "orch-1"')

    def runs(self):
        fx.add_t3_run(self.db, "R1", start=fx.T0, end=fx.T0 + 900, thread_id=self.THREAD, model="claude-opus-5-5")
        fx.add_t3_run(self.db, "R2", start=fx.T0 + 1, end=fx.T0 + 800, thread_id=self.OTHER, model="claude-opus-5-5")

    def collapsed(self):
        self.learn()
        return learn.collapse(self.store())

    def scored(self, rows):
        return {r["run_id"]: r for r in rows if r["source"] == "t3-database"}


class LinkByRunId(IdLinkCase):

    def test_a_hold_with_meta_run_scores_the_run_it_names_with_no_transcript(self):
        # why: the orchestrator transcript is the only link today and `$(agentpace ...)` in the
        #      hold command hides --expect-s from it; the id the orchestrator recorded must link
        #      the hold to its run alone. R1 and R2 both overlap the hold; only the id picks R1.
        # catches: leaving the hold unlinked (no-transcript) or choosing a run by time.
        self.runs()
        self.ledger.write_text(self.hold("opus-impl", self.THREAD))
        rows = self.collapsed()
        runs = self.scored(rows)
        self.assertEqual((runs["R1"]["estimate"]["p50"], runs["R1"]["estimate"]["p80"], runs["R1"]["estimate_source"]), (500.0, 1000.0, "agentpace"))
        self.assertEqual((runs["R2"]["estimate"]["p80"], runs["R2"]["estimate_source"]), (None, None))
        self.assertEqual(runs["R1"]["link_source"], "meta-run")
        self.assertFalse([r for r in rows if r["source"] == "vigil-ledger"], "the hold is the run, not a second row")

    def test_the_task_id_without_the_thread_prefix_names_the_same_thread(self):
        # why: delegate_task returns a task id and a child thread id; either may be recorded.
        #      The thread id is `thread:delegated-task:` plus the url-encoded task id.
        # catches: matching only the full thread id string.
        self.runs()
        self.ledger.write_text(self.hold("opus-impl", "command:mcp:abc:delegate-task:task-1"))
        self.assertEqual(self.scored(self.collapsed())["R1"]["estimate"]["p80"], 1000.0)

    def test_the_id_beats_the_transcript_link(self):
        # why: "before any label/time matching": the transcript here says the hold watched R2
        #      (its dispatch result names R2); the recorded id says R1. The recorded id wins.
        # catches: trying the transcript first and using meta.run only as a fallback.
        self.runs()
        fx.write_orchestrator_transcript(self.claude, "orch-1", fx.delegate(fx.T0, "c1", "R2", self.OTHER) + [fx.bash_hold(fx.T0 + 4, "opus-impl", 1200)])
        self.ledger.write_text(self.hold("opus-impl", self.THREAD))
        runs = self.scored(self.collapsed())
        self.assertEqual((runs["R1"]["estimate"]["p80"], runs["R2"]["estimate"]["p80"]), (1000.0, None))

    def test_an_unknown_id_is_not_rescued_by_the_transcript(self):
        # why: an id that matches nothing must not fall back to a guess (the guess is what
        #      mislinked holds before); it is reported as a failed link.
        # catches: falling back to label/time matching when meta.run is present.
        self.runs()
        fx.write_orchestrator_transcript(self.claude, "orch-1", fx.delegate(fx.T0, "c1", "R2", self.OTHER) + [fx.bash_hold(fx.T0 + 4, "opus-impl", 1200)])
        self.ledger.write_text(self.hold("opus-impl", "thread:delegated-task:no-such-thread"))
        rows = self.collapsed()
        (hold,) = [r for r in rows if r["source"] == "vigil-ledger"]
        self.assertEqual(hold["link_status"], "link-failed")
        self.assertTrue(all(r["estimate"]["p80"] is None for r in self.scored(rows).values()))

    def test_a_t3_run_id_names_exactly_one_run(self):
        # why: a hold may name the run (not its thread); a thread with two runs is then no
        #      longer ambiguous.
        # catches: resolving only thread ids.
        fx.add_t3_run(self.db, "R1", start=fx.T0, end=fx.T0 + 900, thread_id=self.THREAD, model="claude-opus-5-5")
        fx.add_t3_run(self.db, "R2", start=fx.T0 + 20, end=fx.T0 + 880, thread_id=self.THREAD, ordinal=2, model="claude-opus-5-5")
        self.ledger.write_text(self.hold("opus-impl", "R2"))
        runs = self.scored(self.collapsed())
        self.assertEqual((runs["R1"]["estimate"]["p80"], runs["R2"]["estimate"]["p80"]), (None, 1000.0))

    def test_a_thread_with_two_runs_in_the_hold_is_multi_run(self):
        # why: a thread id with several runs inside the hold is not one run's estimate; same rule
        #      as a transcript link ("exactly one run").
        # catches: scoring the first or last run of a continued thread.
        fx.add_t3_run(self.db, "R1", start=fx.T0, end=fx.T0 + 900, thread_id=self.THREAD, model="claude-opus-5-5")
        fx.add_t3_run(self.db, "R2", start=fx.T0 + 1000, end=fx.T0 + 1400, thread_id=self.THREAD, ordinal=2, model="claude-opus-5-5")
        self.ledger.write_text(self.hold("opus-impl", self.THREAD))
        rows = self.collapsed()
        (hold,) = [r for r in rows if r["source"] == "vigil-ledger"]
        self.assertEqual((hold["link_status"], hold["linked_runs"]), ("multi-run", 2))
        self.assertTrue(all(r["estimate"]["p80"] is None for r in self.scored(rows).values()))

    def test_an_earlier_run_of_the_thread_outside_the_hold_does_not_make_it_multi_run(self):
        # why: a continued thread keeps its old runs; only runs from this dispatch on are the hold's.
        # catches: counting every run of the thread.
        fx.add_t3_run(self.db, "R0", start=fx.T0 - 5 * HOUR, end=fx.T0 - 4 * HOUR, thread_id=self.THREAD, model="claude-opus-5-5")
        fx.add_t3_run(self.db, "R1", start=fx.T0, end=fx.T0 + 900, thread_id=self.THREAD, ordinal=2, model="claude-opus-5-5")
        self.ledger.write_text(self.hold("opus-impl", self.THREAD))
        runs = self.scored(self.collapsed())
        self.assertEqual((runs["R0"]["estimate"]["p80"], runs["R1"]["estimate"]["p80"]), (None, 1000.0))

    def test_a_background_agent_id_links_its_subagent(self):
        # why: a Claude Agent dispatch returns an agentId; the transcript is `agent-<id>.jsonl`.
        #      Either spelling names it.
        # catches: resolving T3 ids only.
        fx.write_claude_subagent(self.claude, "agent-abc123", start=fx.T0 + 2, end=fx.T0 + 1802, session="orch-1", model="claude-opus-5-5")
        for spelling in ("abc123", "agent-abc123"):
            self.ledger.write_text(self.hold("opus-impl", spelling, end=fx.T0 + 4000))
            (row,) = [r for r in self.collapsed() if r["source"] != "vigil-ledger"]
            self.assertEqual((row["source"], row["actual_s"], row["estimate"]["p80"]), ("claude-subagent", 1800.0, 1000.0), spelling)

    def test_a_codex_session_id_links_its_turn(self):
        # why: a Codex dispatch has no T3 run; its session id is the rollout's `session_meta.id`.
        # catches: ignoring Codex rows when resolving the id.
        fx.write_codex_rollout(self.codex, "x", session_id="codex-sess-1", start=fx.T0 + 2, end=fx.T0 + 602)
        self.ledger.write_text(self.hold("sol-impl", "codex-sess-1", model="sol", end=fx.T0 + 900))
        (row,) = [r for r in self.collapsed() if r["source"] != "vigil-ledger"]
        self.assertEqual((row["source"], row["actual_s"], row["estimate"]["p50"]), ("codex-transcript", 600.0, 500.0))

    def test_a_hold_registered_after_the_run_ended_is_still_not_scored(self):
        # why: the ended-first guard (the answer was known) applies to an exact link too. A run
        #      id names that one run; a thread id names whatever ran on it, and a run that was
        #      over before the hold opened is not one the hold watched (a broken link, not a guess).
        # catches: exempting id links from the guards that protect the score.
        self.runs()
        for recorded, status in (("R1", "target-ended-first"), (self.THREAD, "link-failed")):
            self.ledger.write_text(self.hold("opus-impl", recorded, start=fx.T0 + 2000, end=fx.T0 + 2100))
            rows = self.collapsed()
            (hold,) = [r for r in rows if r["source"] == "vigil-ledger"]
            self.assertEqual(hold["link_status"], status, recorded)
            self.assertIsNone(self.scored(rows)["R1"]["estimate"]["p80"], recorded)

    def test_a_model_that_disagrees_with_the_run_is_not_scored(self):
        # why: meta.model says sol and the run is Opus: one of the two ids is wrong, and a wrong
        #      pair would score a Sol estimate against an Opus actual.
        # catches: skipping the model check for id links.
        self.runs()
        self.ledger.write_text(self.hold("sol-impl", self.THREAD, model="sol"))
        (hold,) = [r for r in self.collapsed() if r["source"] == "vigil-ledger"]
        self.assertEqual(hold["link_status"], "model-mismatch")

    def test_two_holds_naming_one_run_score_neither(self):
        # why: a run has one estimate; two holds claiming it leave no way to say which was made.
        # catches: the later hold overwriting the earlier one.
        self.runs()
        self.ledger.write_text(self.hold("opus-a", self.THREAD) + self.hold("opus-b", self.THREAD))
        rows = self.collapsed()
        self.assertEqual({r["link_status"] for r in rows if r["source"] == "vigil-ledger"}, {"shared-target"})
        self.assertIsNone(self.scored(rows)["R1"]["estimate"]["p80"])

    def test_a_run_ingested_after_its_hold_is_scored_at_the_next_read(self):
        # why: the hold ends before learn has seen the run (a T3 run is read once it finishes);
        #      the link is resolved when the store is read, so a late run is not lost.
        # catches: deciding the link once, at the hold's ingest.
        self.ledger.write_text(self.hold("opus-impl", self.THREAD))
        (hold,) = learn.collapse(self.collapsed())
        self.assertEqual(hold["link_status"], "link-failed")
        fx.add_t3_run(self.db, "R1", start=fx.T0, end=fx.T0 + 900, thread_id=self.THREAD, model="claude-opus-5-5")
        self.assertEqual(self.scored(self.collapsed())["R1"]["estimate"]["p80"], 1000.0)

    def test_a_cancelled_run_is_reported_as_cancelled_not_failed(self):
        # why: a delegation that was stopped has no run to score; calling it a broken link hides
        #      the difference between a lost id and a cancellation.
        # catches: id links skipping the cancelled-target probe.
        fx.add_t3_run(self.db, "R1", start=fx.T0, end=fx.T0 + 50, thread_id=self.THREAD, status="cancelled", model="claude-opus-5-5")
        self.ledger.write_text(self.hold("opus-impl", self.THREAD))
        (hold,) = [r for r in self.collapsed() if r["source"] == "vigil-ledger"]
        self.assertEqual(hold["link_status"], "target-cancelled")


class EstimateSource(test_learn.LearnCase):
    def hold(self, expect_s, **meta):
        return fx.vigil_line("W", "opus-impl", start=fx.T0, end=fx.T0 + 900, expect_s=expect_s, meta={"model": "opus", "kind": "implementation", **meta})

    def row(self):
        self.learn()
        (row,) = self.store()
        return row

    def test_meta_quantiles_are_the_estimate(self):
        # why: p50/p80 in meta are the estimator's output at dispatch; scoring them is the point.
        # catches: scoring expect_s instead, or dropping the P50.
        row = self.row_for(self.hold(2400, p50="400", p80="900", est="agentpace-0.13.1"))
        self.assertEqual((row["estimate"]["p50"], row["estimate"]["p80"], row["estimate_source"], row["estimator"]), (400.0, 900.0, "agentpace", "agentpace-0.13.1"))
        self.assertIsNone(row["ceiling_s"], "the 2400 s alarm is not an estimate once the estimator's own P80 is there")

    def row_for(self, text):
        self.ledger.write_text(text)
        return self.row()

    def test_a_hand_set_expect_s_alone_is_a_ceiling_not_a_p80(self):
        # why: most holds carry a hand-set alarm that the orchestrator picked as a ceiling; scoring
        #      it as a P80 reads its 95%+ hit rate as a calibrated estimate.
        # catches: expect_s still landing in estimate["p80"].
        row = self.row_for(self.hold(1200))
        self.assertEqual((row["ceiling_s"], row["estimate_source"]), (1200.0, "ceiling"))
        self.assertEqual(row["estimate"], {"p50": None, "p80": None, "p90": None})

    def test_invalid_meta_numbers_leave_the_expect_s_a_ceiling(self):
        # why: meta is opaque text; a hand-typed `p80=soon` must not become 0, NaN or a crash,
        #      and must not hide the alarm the hold still carries.
        # catches: float() on unvalidated text, or accepting zero, negative and non-finite values.
        for bad in ("soon", "-5", "0", "nan", "inf", ""):
            row = self.row_for(self.hold(1200, p50=bad, p80=bad))
            self.assertEqual((row["estimate_source"], row["ceiling_s"], row["estimate"]["p80"]), ("ceiling", 1200.0, None), repr(bad))

    def test_one_valid_quantile_is_still_the_estimate(self):
        # why: a hold with only p80 (or only p50) was still estimated by the estimator.
        # catches: requiring both keys.
        row = self.row_for(self.hold(None, p80="900"))
        self.assertEqual((row["estimate"]["p50"], row["estimate"]["p80"], row["estimate_source"]), (None, 900.0, "agentpace"))

    def test_a_plan_forecast_merges_with_a_ceiling_without_becoming_a_p80(self):
        # why: the plan task's P50 and a hand-set ceiling are two different things for one run;
        #      merged, the P50 is the estimate and the ceiling stays a ceiling.
        # catches: the merge writing the ceiling into estimate["p80"].
        ledger_id = learn.vigil_execution_key("W", "opus-impl", fx.T0)
        self.ledger.write_text(self.hold(1200, task="plan-a/t1"))
        fx.write_plan(
            self.plans,
            "plan-a",
            [{"id": "t1", "type": "agent", "model": "opus", "kind": "implementation", "status": "done", "actual_s": 900, "started_at": fx.T0, "ended_at": fx.T0 + 900, "actual_source": "vigil-ledger", "forecast_s": 800}],
            events=[{"action": "ledger-done", "task": "t1", "at": fx.T0 + 900, "ledger_id": ledger_id}],
        )
        self.learn()
        (merged,) = learn.collapse(self.store())
        self.assertEqual((merged["estimate"]["p50"], merged["estimate"]["p80"], merged["ceiling_s"]), (800.0, None, 1200.0))
        self.assertEqual(merged["estimate_source"], "ceiling+plan-forecast")

    def test_a_ceiling_moves_onto_the_run_it_watched(self):
        # why: the ceiling is judged against the run's actual, not the hold's lifetime, so it
        #      follows the link like an estimate does.
        # catches: dropping ceiling_s when the hold is folded into its run.
        fx.add_t3_run(self.db, "R1", start=fx.T0, end=fx.T0 + 900, thread_id="thread:delegated-task:c", model="claude-opus-5-5")
        self.ledger.write_text(self.hold(1200, run="c"))
        self.learn()
        (run,) = learn.collapse(self.store())
        self.assertEqual((run["source"], run["ceiling_s"], run["estimate"]["p80"], run["estimate_source"]), ("t3-database", 1200.0, None, "ceiling"))


class ObservationCase(unittest.TestCase):
    def obs(self, actual, *, p80=None, ceiling=None, age_days=1, tag="x"):
        end = NOW - age_days * DAY
        return learn.observation(
            source="t3-database",
            key=("t", tag, actual, p80, ceiling),
            start=end - actual,
            end=end,
            model_raw="opus",
            model="opus",
            kind="implementation",
            kind_source="vigil-meta",
            label_source="t3-database",
            estimate={"p80": p80},
            estimate_source="agentpace" if p80 else None,
            ceiling_s=ceiling,
        )


class CeilingReport(ObservationCase):
    def test_ceilings_are_counted_apart_from_p80_coverage(self):
        # why: a ceiling is a hand-set alarm, not a quantile; in P80 coverage it looks like a
        #      calibrated 80% estimate. Three agentpace runs (one over its P80) and two ceiling
        #      runs (both under): P80 coverage must be 2/3, the ceilings 2/2, reported separately.
        # catches: ceilings feeding coverage, loss or n_estimated.
        rows = [
            self.obs(100, p80=120, tag="a"),
            self.obs(110, p80=120, tag="b"),
            self.obs(500, p80=120, tag="c"),
            self.obs(300, ceiling=3600, tag="d"),
            self.obs(400, ceiling=3600, tag="e"),
        ]
        report = calibration.calibrate(rows, now=NOW, window_s=7 * DAY)
        (group,) = [g for g in report["groups"] if g["kind"] == "implementation"]
        this = group["this"]
        self.assertEqual((this["n"], this["n_estimated"], this["coverage"]["p80"]), (5, 3, {"hits": 2, "n": 3, "rate": 2 / 3}))
        self.assertEqual(this["ceiling"], {"hits": 2, "n": 2, "rate": 1.0})
        self.assertEqual((report["estimates"]["estimated"], report["estimates"]["ceiling"]), (3, 2))

    def test_a_ceiling_overrun_is_counted_as_a_miss_of_the_ceiling(self):
        # why: the ceiling's hit rate is the one number it supports (how often the alarm was
        #      generous enough); an overrun must read as a miss, not be dropped.
        # catches: counting only the runs under the ceiling.
        rows = [self.obs(100, ceiling=300, tag="a"), self.obs(900, ceiling=300, tag="b")]
        report = calibration.calibrate(rows, now=NOW, window_s=7 * DAY)
        self.assertEqual(report["groups"][0]["this"]["ceiling"], {"hits": 1, "n": 2, "rate": 0.5})

    def test_no_ceiling_reads_as_absent_not_zero(self):
        # why: honest absence (module invariant): a group without ceilings has n 0 and rate None.
        # catches: a 0 rate that reads as every ceiling being missed.
        report = calibration.calibrate([self.obs(100, p80=120)], now=NOW, window_s=7 * DAY)
        self.assertEqual(report["groups"][0]["this"]["ceiling"], {"hits": 0, "n": 0, "rate": None})

    def test_the_markdown_names_the_ceiling_section_and_says_it_is_not_p80(self):
        # why: the weekly reader sees the markdown; the two kinds of number must not look alike.
        # catches: a report that lists ceilings only in the JSON, or in the P80 column.
        rows = [self.obs(100, p80=120, tag="a"), self.obs(300, ceiling=3600, tag="b")]
        text = calibration.render_markdown(calibration.calibrate(rows, now=NOW, window_s=7 * DAY))
        self.assertIn("## Hand-set ceilings", text)
        self.assertIn("not a P80", text)

    def test_a_group_with_only_ceilings_is_not_hidden(self):
        # why: the markdown table omits groups with no estimate; a model whose every hold is a
        #      ceiling still has a ceiling table row, so the reader sees how often it was met.
        # catches: filtering the ceiling section by n_estimated.
        text = calibration.render_markdown(calibration.calibrate([self.obs(300, ceiling=3600)], now=NOW, window_s=7 * DAY))
        self.assertIn("| opus | implementation |", text.split("## Hand-set ceilings", 1)[1])


class SkillText(unittest.TestCase):
    def test_the_manual_and_cli_doc_show_the_command_the_cli_accepts(self):
        # why: orchestrators learn the dispatch recipe from the skill; a recipe naming a flag the
        #      CLI does not have (or dropping `run`) puts holds back to label/time linking.
        # catches: the skill and docs/cli.md drifting from --vigil-meta and --meta run=.
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        for page in (root / "skills" / "agentpace" / "SKILL.md", root / "docs" / "cli.md"):
            text = page.read_text(encoding="utf-8")
            self.assertIn("estimate-run --model", text, page.name)
            self.assertIn("--vigil-meta", text, page.name)
            self.assertIn("--meta run=", text, page.name)
        parser_flags = {a for action in cli.build_parser()._subparsers._group_actions[0].choices["estimate-run"]._actions for a in action.option_strings}
        self.assertIn("--vigil-meta", parser_flags)


class EnvIsolated(unittest.TestCase):
    def test_estimate_run_does_not_need_a_learned_store(self):
        # why: --vigil-meta runs on every dispatch; without --learned it must not touch the
        #      state directory (a fresh machine has none).
        # catches: reading learn state unconditionally.
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "rows.jsonl"
            lr.write_rows(data, [lr.make_row(start=i * 1000, end=i * 1000 + 60, model="opus", kind="fix", session=str(i)) for i in range(12)])
            with mock.patch.dict(os.environ, {"AGENTPACE_STATE": str(Path(directory) / "none")}):
                code, out, _ = vigil_flags(["--model", "opus", "--kind", "fix", "--vigil-meta"], data)
            self.assertEqual(code, 0)
            self.assertTrue(out.startswith("--expect-s 60 "))
            self.assertFalse((Path(directory) / "none").exists())

# Synthetic ids in the exact shapes T3 returns (checked against a live delegate_task and
# t3_thread_send result): the task id is `node:delegated-task:` plus the url-encoded command
# name, the child thread `thread:delegated-task:` plus the same, the run id the thread id
# url-encoded once more, and a follow-up run carries its own task id in `payload.delegatedTaskId`.
ORCH = "0a0a0a0a-1b1b-4c2c-8d3d-4e4e4e4e4e4e"
TASK_UUID = "5f5f5f5f-6a6a-4b7b-9c8c-7d7d7d7d7d7d"
SEND_UUID = "8e8e8e8e-9f9f-4a0a-8b1b-2c2c2c2c2c2c"
COMMAND = f"command:mcp:{ORCH}:delegate-task:{TASK_UUID}"
ENCODED = COMMAND.replace(":", "%3A")
CHILD_THREAD = f"thread:delegated-task:{ENCODED}"
TASK_ID = f"node:delegated-task:{ENCODED}"
TURN_ITEM = f"turn-item:delegated-task:{ENCODED}"
RUN_1 = "run:thread:" + CHILD_THREAD.replace("%", "%25").replace(":", "%3A") + ":ordinal:1"
RUN_2 = RUN_1.replace(":ordinal:1", ":ordinal:2")
FOLLOW_UP = "node:delegated-task:" + f"command:mcp:{ORCH}:thread-send:{SEND_UUID}".replace(":", "%3A")
DECOY_THREAD = CHILD_THREAD.replace(TASK_UUID, "1d1d1d1d-2e2e-4f3f-8a4a-5b5b5b5b5b5b")


class T3IdForms(IdLinkCase):
    """Every id T3 hands an orchestrator for a delegated run resolves to that run."""

    def first_run(self, **kw):
        fx.add_t3_run(self.db, RUN_1, start=fx.T0, end=fx.T0 + 900, thread_id=CHILD_THREAD, model="claude-opus-5-5", **kw)
        fx.add_t3_run(self.db, "DECOY", start=fx.T0 + 1, end=fx.T0 + 800, thread_id=DECOY_THREAD, model="claude-opus-5-5")

    def follow_up(self, **kw):
        fx.add_t3_run(self.db, RUN_2, start=fx.T0 + 2000, end=fx.T0 + 2600, thread_id=CHILD_THREAD, ordinal=2, delegated_task_id=FOLLOW_UP, model="claude-opus-5-5", **kw)

    def test_every_spelling_of_the_first_dispatch_names_its_run(self):
        # why: delegate_task returns taskId (`node:`), childThreadId (`thread:`) and childRunId;
        #      the parent's background list shows `turn-item:`; an orchestrator may also decode
        #      the id or drop the prefix. The skills say record childThreadId, and a taskId works.
        # catches: matching only `thread:delegated-task:` (every taskId-linked hold link-failed).
        self.first_run()
        for recorded in (CHILD_THREAD, TASK_ID, TURN_ITEM, RUN_1, COMMAND, ENCODED, "thread:delegated-task:" + COMMAND, "node:delegated-task:" + COMMAND):
            self.ledger.write_text(self.hold("opus-impl", recorded))
            runs = self.scored(self.collapsed())
            self.assertEqual((runs[RUN_1]["estimate"]["p80"], runs["DECOY"]["estimate"]["p80"]), (1000.0, None), recorded)

    def test_a_follow_up_task_id_names_the_follow_up_run(self):
        # why: t3_thread_send starts run 2 on the same child thread and returns its own taskId,
        #      which T3 stores on that run as payload.delegatedTaskId. A hold recorded with it
        #      is run 2's estimate, even when the hold also spans run 1.
        # catches: not reading delegatedTaskId, or resolving a follow-up id to the whole thread.
        self.first_run()
        self.follow_up()
        self.ledger.write_text(self.hold("opus-fix", FOLLOW_UP, start=fx.T0 + 5, end=fx.T0 + 2700))
        runs = self.scored(self.collapsed())
        self.assertEqual((runs[RUN_1]["estimate"]["p80"], runs[RUN_2]["estimate"]["p80"]), (None, 1000.0))

    def test_the_first_task_id_does_not_name_the_follow_up(self):
        # why: the delegate_task taskId names the dispatch that created the thread, not runs a
        #      later t3_thread_send started; a hold spanning both is still run 1's.
        # catches: treating a taskId as its thread (multi-run here).
        self.first_run()
        self.follow_up()
        self.ledger.write_text(self.hold("opus-impl", TASK_ID, start=fx.T0 + 5, end=fx.T0 + 2700))
        runs = self.scored(self.collapsed())
        self.assertEqual((runs[RUN_1]["estimate"]["p80"], runs[RUN_2]["estimate"]["p80"]), (1000.0, None))

    def test_the_child_thread_id_picks_the_follow_up_by_the_hold_window(self):
        # why: the preferred value is childThreadId, the same for every run on the thread; the
        #      hold's window picks the run it watched.
        # catches: a thread id always resolving to the first run, or to multi-run.
        self.first_run()
        self.follow_up()
        self.ledger.write_text(self.hold("opus-a", CHILD_THREAD, start=fx.T0 + 5, end=fx.T0 + 950) + self.hold("opus-b", CHILD_THREAD, start=fx.T0 + 2005, end=fx.T0 + 2650, p80="2000"))
        runs = self.scored(self.collapsed())
        self.assertEqual((runs[RUN_1]["estimate"]["p80"], runs[RUN_2]["estimate"]["p80"]), (1000.0, 2000.0))

    def test_a_cancelled_task_is_reported_as_cancelled_for_every_id_form(self):
        # why: a stopped delegation has nothing to score; a taskId or follow-up id must find its
        #      cancelled run in T3 as the thread id does, or the cancellation reads as a lost id.
        # catches: the cancelled probe matching only thread ids.
        fx.add_t3_run(self.db, RUN_1, start=fx.T0, end=fx.T0 + 50, thread_id=CHILD_THREAD, status="cancelled", model="claude-opus-5-5")
        for recorded in (TASK_ID, CHILD_THREAD, COMMAND, RUN_1):
            self.ledger.write_text(self.hold("opus-impl", recorded))
            (hold,) = [r for r in self.collapsed() if r["source"] == "vigil-ledger"]
            self.assertEqual(hold["link_status"], "target-cancelled", recorded)

    def test_a_cancelled_follow_up_is_cancelled_though_run_1_completed(self):
        # why: run 1 finished; the follow-up was stopped. Its hold names the follow-up and must
        #      say cancelled, not claim run 1 or call the link broken.
        # catches: the cancelled probe requiring every run of the thread to be cancelled.
        self.first_run()
        self.follow_up(status="cancelled")
        for recorded in (FOLLOW_UP, CHILD_THREAD):
            self.ledger.write_text(self.hold("opus-fix", recorded, start=fx.T0 + 1995, end=fx.T0 + 2100))
            rows = self.collapsed()
            (hold,) = [r for r in rows if r["source"] == "vigil-ledger"]
            self.assertEqual(hold["link_status"], "target-cancelled", recorded)
            self.assertIsNone(self.scored(rows)[RUN_1]["estimate"]["p80"], recorded)


class HoldWindow(IdLinkCase):
    """A hold recorded with a thread id claims only a run that overlaps it."""

    def test_a_failed_run_does_not_hand_its_hold_to_the_next_run(self):
        # why: run 1 failed (not stored); an hour later run 2 succeeded on the same thread with
        #      its own hold. Hold 1 names one stored run, run 2, which it never watched.
        # catches: a single named run returned without an overlap check (hold 1 stealing run 2,
        #          or both holds becoming shared-target).
        fx.add_t3_run(self.db, "R1", start=fx.T0, end=fx.T0 + 300, thread_id=self.THREAD, status="failed", model="claude-opus-5-5")
        fx.add_t3_run(self.db, "R2", start=fx.T0 + HOUR, end=fx.T0 + HOUR + 600, thread_id=self.THREAD, ordinal=2, model="claude-opus-5-5")
        self.ledger.write_text(self.hold("opus-a", self.THREAD, start=fx.T0 + 5, end=fx.T0 + 600) + self.hold("opus-b", self.THREAD, start=fx.T0 + HOUR + 5, end=fx.T0 + HOUR + 700, p80="2000"))
        rows = self.collapsed()
        (hold,) = [r for r in rows if r["source"] == "vigil-ledger"]
        self.assertEqual((hold["pointers"]["vigil_label"], hold["link_status"]), ("opus-a", "link-failed"))
        self.assertEqual(self.scored(rows)["R2"]["estimate"]["p80"], 2000.0)

    def test_a_run_that_finished_before_the_hold_opened_is_not_counted(self):
        # why: run 1 (T0..T0+180) ended before hold 2 opened at T0+305 for run 2; it started
        #      within the dispatch window, but a run that is over is not one the hold watched.
        # catches: `inside` without `end > hold.start` (hold 2 falsely multi-run).
        fx.add_t3_run(self.db, "R1", start=fx.T0, end=fx.T0 + 180, thread_id=self.THREAD, model="claude-opus-5-5")
        fx.add_t3_run(self.db, "R2", start=fx.T0 + 300, end=fx.T0 + 900, thread_id=self.THREAD, ordinal=2, model="claude-opus-5-5")
        self.ledger.write_text(self.hold("opus-b", self.THREAD, start=fx.T0 + 305, end=fx.T0 + 950))
        runs = self.scored(self.collapsed())
        self.assertEqual((runs["R1"]["estimate"]["p80"], runs["R2"]["estimate"]["p80"]), (None, 1000.0))

    def test_a_run_that_starts_after_the_hold_closed_is_not_the_holds(self):
        # why: the one named run starts after the hold was done; nothing the hold watched.
        # catches: the single-run shortcut skipping the window.
        fx.add_t3_run(self.db, "R1", start=fx.T0 + 2 * HOUR, end=fx.T0 + 2 * HOUR + 600, thread_id=self.THREAD, model="claude-opus-5-5")
        self.ledger.write_text(self.hold("opus-a", self.THREAD, start=fx.T0 + 5, end=fx.T0 + 600))
        rows = self.collapsed()
        (hold,) = [r for r in rows if r["source"] == "vigil-ledger"]
        self.assertEqual(hold["link_status"], "link-failed")
        self.assertIsNone(self.scored(rows)["R1"]["estimate"]["p80"])


class RecordedIdBeatsLabelRole(IdLinkCase):
    def test_an_owner_or_model_free_label_does_not_hide_a_recorded_id(self):
        # why: the label rule guesses check-ins (`owner`) and machine jobs (no model word); an
        #      orchestrator that recorded the run's id said what the hold watched.
        # catches: hold_role running before meta.run (the hold marked not-an-agent-run).
        self.runs()
        for label, meta in (("opus-owner-decision-demo", {}), ("fix-parser", {"model": ""})):
            shutil.rmtree(self.state, ignore_errors=True)  # a fresh store: two holds on R1 would be shared-target
            self.ledger.write_text(self.hold(label, self.THREAD, **meta))
            rows = self.collapsed()
            self.assertEqual(self.scored(rows)["R1"]["estimate"]["p80"], 1000.0, label)
            self.assertFalse([r for r in rows if r["source"] == "vigil-ledger"], label)


class CancelledSubagent(IdLinkCase):
    def test_a_cancelled_background_agent_named_by_id_is_target_cancelled(self):
        # why: an Agent stopped before it ended a turn leaves a transcript and no stored run; the
        #      recorded agentId finds that transcript, so the hold says cancelled.
        # catches: the cancelled probe reading subagent transcripts only for transcript links.
        fx.write_claude_subagent(self.claude, "agent-abc123", start=fx.T0 + 2, end=fx.T0 + 300, session="orch-1", turns=[fx.claude_turn(fx.T0 + 2, fx.T0 + 300, ending="none")])
        for spelling in ("abc123", "agent-abc123"):
            self.ledger.write_text(self.hold("opus-impl", spelling))
            (hold,) = [r for r in self.collapsed() if r["source"] == "vigil-ledger"]
            self.assertEqual(hold["link_status"], "target-cancelled", spelling)


class HoldWithNothingToScore(IdLinkCase):
    def test_a_recorded_id_without_estimate_or_ceiling_is_not_scored(self):
        # why: `$(agentpace ... --vigil-meta)` can fail and leave a hold with only `--meta run=`;
        #      it found its run but carries nothing to score, so it is no scored hold and no estimate.
        # catches: counting it in holds.scored and estimates.linked_by_id.
        self.runs()
        self.ledger.write_text(fx.vigil_line("W", "opus-impl", start=fx.T0 + 5, end=fx.T0 + 1500, meta={"run": self.THREAD, "model": "opus"}))
        self.learn()
        report = calibration.calibrate(self.store(), now=NOW, window_s=7 * DAY)
        self.assertEqual((report["holds"].get("scored"), report["estimates"]["linked_by_id"], report["estimates"]["estimated"]), (0, 0, 0))
        self.assertEqual(report["holds"].get("linked-no-estimate"), 1)


class CeilingCount(ObservationCase):
    def test_a_ceiling_beside_an_estimate_is_not_counted_as_ceiling_only(self):
        # why: "N more only a hand-set ceiling" adds to the estimated count; a run with both a
        #      plan P50 and a ceiling is already in the estimated count.
        # catches: "1 of 1 had a dispatch estimate and 1 more only a hand-set ceiling".
        row = self.obs(300, p80=400, ceiling=3600)
        report = calibration.calibrate([row], now=NOW, window_s=7 * DAY)
        self.assertEqual((report["estimates"]["estimated"], report["estimates"]["ceiling"]), (1, 0))
        self.assertIn("1 of 1 scored runs this window had a dispatch estimate and 0 more only a hand-set ceiling", calibration.render_markdown(report))


class FlagValues(FlagCase):
    def test_a_value_with_whitespace_is_refused(self):
        # why: the line is expanded unquoted in `$(...)`; quotes inside it are literal, so
        #      `kind=code review` splits into a stray positional argument for vigil.
        # catches: printing a value the shell splits.
        data = self.history("opus", "code review", [60] * 12)
        code, out, err = vigil_flags(["--model", "opus", "--kind", "code review", "--vigil-meta"], data)
        self.assertEqual((code, out), (2, ""))
        self.assertIn("kind", err)


class SourceTags(test_learn.LearnCase):
    def test_a_merged_source_is_named_only_when_it_gave_a_quantile(self):
        # why: estimate_source says whose numbers are scored; a plan forecast whose P50 lost to
        #      the hold's own P50 contributed nothing.
        # catches: `agentpace+plan-forecast` with only the agentpace quantiles.
        ledger_id = learn.vigil_execution_key("W", "opus-impl", fx.T0)
        self.ledger.write_text(fx.vigil_line("W", "opus-impl", start=fx.T0, end=fx.T0 + 900, meta={"model": "opus", "kind": "implementation", "p50": "400", "p80": "900", "task": "plan-a/t1"}))
        fx.write_plan(
            self.plans,
            "plan-a",
            [{"id": "t1", "type": "agent", "model": "opus", "kind": "implementation", "status": "done", "actual_s": 900, "started_at": fx.T0, "ended_at": fx.T0 + 900, "actual_source": "vigil-ledger", "forecast_s": 800}],
            events=[{"action": "ledger-done", "task": "t1", "at": fx.T0 + 900, "ledger_id": ledger_id}],
        )
        self.learn()
        (merged,) = learn.collapse(self.store())
        self.assertEqual((merged["estimate"]["p50"], merged["estimate_source"]), (400.0, "agentpace"))


class LateHoldAndStoppedRunCase(IdLinkCase):
    """A late hold, a stopped run beside a thread id, and the dispatch rule."""

    def first_run(self, **kw):
        fx.add_t3_run(self.db, RUN_1, start=fx.T0, end=fx.T0 + 900, thread_id=CHILD_THREAD, model="claude-opus-5-5", **kw)

    def follow_up(self, **kw):
        fx.add_t3_run(self.db, RUN_2, start=fx.T0 + 2000, end=fx.T0 + 2600, thread_id=CHILD_THREAD, ordinal=2, delegated_task_id=FOLLOW_UP, model="claude-opus-5-5", **kw)

    def holds(self, rows):
        return {r["pointers"]["vigil_label"]: r["link_status"] for r in rows if r["source"] == "vigil-ledger"}

    def test_a_late_hold_on_an_ended_run_does_not_void_the_valid_hold(self):
        # why: the orchestrator reuses the creating taskId for a later hold (the skills say it
        #      works); run 1 ended before hold B opened, so B is no claimant of run 1.
        # catches: both holds becoming shared-target, so run 1 is never scored.
        self.first_run()
        self.follow_up()
        self.ledger.write_text(self.hold("opus-a", TASK_ID, start=fx.T0 + 5, end=fx.T0 + 950) + self.hold("opus-b", TASK_ID, start=fx.T0 + 2005, end=fx.T0 + 2650, p80="2000"))
        rows = self.collapsed()
        self.assertEqual(self.holds(rows), {"opus-b": "target-ended-first"})
        runs = self.scored(rows)
        self.assertEqual((runs[RUN_1]["estimate"]["p80"], runs[RUN_2]["estimate"]["p80"]), (1000.0, None))

    def test_a_stopped_run_beside_a_thread_hold_makes_it_multi_run(self):
        # why: run 1 stopped (cancelled, failed or interrupted; none is stored) and run 2 was
        #      started on the same thread; a hold named by thread that spans both cannot say which
        #      run its estimate was for. The same shape scored run 2 with run 1's estimate.
        # catches: scoring the one stored run, because the stopped one is invisible to the store.
        for status in ("cancelled", "failed", "interrupted"):
            shutil.rmtree(self.state, ignore_errors=True)
            self.db.unlink()
            fx.make_t3_db(self.db)
            fx.add_t3_run(self.db, RUN_1, start=fx.T0, end=fx.T0 + 50, thread_id=CHILD_THREAD, status=status, model="claude-opus-5-5")
            fx.add_t3_run(self.db, RUN_2, start=fx.T0 + 100, end=fx.T0 + 700, thread_id=CHILD_THREAD, ordinal=2, model="claude-opus-5-5")
            self.ledger.write_text(self.hold("opus-a", CHILD_THREAD, start=fx.T0 + 5, end=fx.T0 + 750))
            rows = self.collapsed()
            self.assertEqual(self.holds(rows), {"opus-a": "multi-run"}, status)
            self.assertIsNone(self.scored(rows)[RUN_2]["estimate"]["p80"], status)

    def test_a_stopped_run_that_ended_before_the_hold_opened_changes_nothing(self):
        # why: run 1 stopped at T0+50; the hold opened at T0+100 for run 2, so it never watched run 1.
        # catches: counting every stopped run on the thread (multi-run), not just those the hold overlaps.
        fx.add_t3_run(self.db, RUN_1, start=fx.T0, end=fx.T0 + 50, thread_id=CHILD_THREAD, status="failed", model="claude-opus-5-5")
        fx.add_t3_run(self.db, RUN_2, start=fx.T0 + 100, end=fx.T0 + 700, thread_id=CHILD_THREAD, ordinal=2, model="claude-opus-5-5")
        self.ledger.write_text(self.hold("opus-b", CHILD_THREAD, start=fx.T0 + 105, end=fx.T0 + 750))
        rows = self.collapsed()
        self.assertEqual(self.holds(rows), {})
        self.assertEqual(self.scored(rows)[RUN_2]["estimate"]["p80"], 1000.0)

    def test_an_exact_task_id_is_not_made_multi_run_by_a_stopped_run(self):
        # why: only a thread id names every run on the thread; a follow-up taskId names that run
        #      alone, so run 1 stopping earlier does not change the follow-up's hold.
        # catches: applying the stopped-run rule to ids that are exact.
        fx.add_t3_run(self.db, RUN_1, start=fx.T0, end=fx.T0 + 50, thread_id=CHILD_THREAD, status="failed", model="claude-opus-5-5")
        self.follow_up()
        self.ledger.write_text(self.hold("opus-b", FOLLOW_UP, start=fx.T0 + 5, end=fx.T0 + 2700))
        rows = self.collapsed()
        self.assertEqual(self.holds(rows), {})
        self.assertEqual(self.scored(rows)[RUN_2]["estimate"]["p80"], 1000.0)

    def test_an_unrelated_old_run_does_not_change_a_thread_holds_outcome(self):
        # why: the 15-minute dispatch rule used to apply only when the id named two stored runs,
        #      so a run that ended two days ago flipped a hold on a 3-hour run from scored to link-failed.
        # catches: the rule depending on how many runs the thread has.
        outcomes = []
        for old in (False, True):
            shutil.rmtree(self.state, ignore_errors=True)
            self.db.unlink()
            fx.make_t3_db(self.db)
            if old:
                fx.add_t3_run(self.db, "OLD", start=fx.T0 - 49 * HOUR, end=fx.T0 - 48 * HOUR, thread_id=CHILD_THREAD, model="claude-opus-5-5")
            fx.add_t3_run(self.db, RUN_2, start=fx.T0, end=fx.T0 + 3 * HOUR, thread_id=CHILD_THREAD, ordinal=2, model="claude-opus-5-5")
            self.ledger.write_text(self.hold("opus-a", CHILD_THREAD, start=fx.T0 + 2 * HOUR, end=fx.T0 + 3 * HOUR))
            outcomes.append(self.holds(self.collapsed()))
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0], {"opus-a": "link-failed"})

    def test_a_task_id_does_not_name_a_later_run_without_a_follow_up_id(self):
        # why: the creating taskId names run 1; a later run T3 started without its own task id
        #      (the owner typing in the child thread) is not that dispatch.
        # catches: giving every run on a delegated thread the creating task's alias.
        self.first_run()
        fx.add_t3_run(self.db, RUN_2, start=fx.T0 + 1000, end=fx.T0 + 1600, thread_id=CHILD_THREAD, ordinal=2, model="claude-opus-5-5")
        self.ledger.write_text(self.hold("opus-a", TASK_ID, start=fx.T0 + 5, end=fx.T0 + 1700))
        runs = self.scored(self.collapsed())
        self.assertEqual((runs[RUN_1]["estimate"]["p80"], runs[RUN_2]["estimate"]["p80"]), (1000.0, None))


if __name__ == "__main__":
    unittest.main()
