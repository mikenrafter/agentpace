"""agentpace calibration: evidence for the weekly digest. SYNTHETIC observations only.

Expected values are hand-computed in each test's comment, not derived from the code under test.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agentpace import calibration, cli, learn

DAY = 86_400.0
NOW = 1_800_000_000.0


def obs(model, kind, *, age_days, actual, p50=None, p80=None, p90=None, category=None, source="t3-database", pointers=None, parent_session=None, **extra):
    """One observation finished `age_days` before NOW."""
    end = NOW - age_days * DAY
    estimate_source = None if p50 is None and p80 is None and p90 is None else "test-estimate"
    return learn.observation(
        source=source,
        key=(source, model, kind, age_days, actual, str(pointers), str(extra)),
        start=end - actual,
        end=end,
        model_raw=model,
        model=model,
        kind=kind,
        kind_source="vigil-meta",
        label_source=extra.pop("label_source", "vigil-hold" if source == "vigil-ledger" else source),
        estimate={"p50": p50, "p80": p80, "p90": p90},
        estimate_source=estimate_source,
        usage_limit=category == "usage-wait",
        role=category if category in learn.ROLE_CATEGORIES else None,
        pointers=pointers or {},
        **extra,
    ) | ({"parent_session": parent_session} if parent_session else {})


def group(report, model, kind):
    (found,) = [g for g in report["groups"] if (g["model"], g["kind"]) == (model, kind)]
    return found


class Stats(unittest.TestCase):
    def test_pinball_and_coverage_for_a_p80_only_estimate(self):
        # why: vigil --expect-s is one P80 number; loss and coverage must be scored on it alone.
        # catches: scoring a missing P50/P90 as zero loss or as a hit.
        # arithmetic, p80=100, q=.8: actual 50 -> err -50 -> (q-1)*err = .2*50 = 10; actual 100 -> 0;
        #   actual 150 -> err 50 -> .8*50 = 40; actual 300 -> err 200 -> 160. mean = 210/4 = 52.5.
        #   hits (actual <= 100): 2 of 4.
        rows = [obs("opus", "implementation", age_days=1, actual=a, p80=100) for a in (50, 100, 150, 300)]
        stats = group(calibration.calibrate(rows, now=NOW, window_s=7 * DAY), "opus", "implementation")["this"]
        self.assertEqual((stats["n"], stats["n_estimated"], stats["mean_pinball_s"]), (4, 4, 52.5))
        self.assertEqual(stats["coverage"]["p80"], {"hits": 2, "n": 4, "rate": 0.5})
        self.assertEqual(stats["coverage"]["p50"], {"hits": 0, "n": 0, "rate": None})

    def test_three_quantile_loss_is_the_mean_over_the_three_points(self):
        # why: an estimate carrying P50/P80/P90 is scored on all three, averaged.
        # catches: scoring only P80, or summing instead of averaging.
        # arithmetic, actual 100, p50=80,p80=120,p90=200: p50 err 20 -> .5*20 = 10; p80 err -20 -> .2*20 = 4;
        #   p90 err -100 -> .1*100 = 10; mean = 24/3 = 8.0. coverage: p50 miss, p80 hit, p90 hit.
        rows = [obs("sol", "review", age_days=1, actual=100, p50=80, p80=120, p90=200)]
        stats = group(calibration.calibrate(rows, now=NOW, window_s=7 * DAY), "sol", "review")["this"]
        self.assertAlmostEqual(stats["mean_pinball_s"], 8.0, places=9)  # float error only
        self.assertEqual([stats["coverage"][q]["hits"] for q in ("p50", "p80", "p90")], [0, 1, 1])

    def test_window_and_previous_weeks_are_separate_populations(self):
        # why: this week's error is judged against earlier weeks, so the two must not mix.
        # catches: scoring everything in one bucket, or a boundary off by one (end == now - window).
        # arithmetic: ends 1d, 6.9d old are in; 7d (== boundary, belongs to previous) and 20d are previous.
        rows = [obs("opus", "fix", age_days=d, actual=100, p80=100) for d in (1, 6.9, 7, 20)]
        found = group(calibration.calibrate(rows, now=NOW, window_s=7 * DAY), "opus", "fix")
        self.assertEqual((found["this"]["n"], found["previous"]["n"]), (2, 2))

    def test_runs_without_any_estimate_report_none_not_zero_loss(self):
        # why: honest absence; a run with no dispatch estimate has no error, not a perfect one.
        # catches: mean_pinball 0.0 for an unestimated group.
        rows = [obs("neon", "review", age_days=1, actual=100)]
        stats = group(calibration.calibrate(rows, now=NOW, window_s=7 * DAY), "neon", "review")["this"]
        self.assertEqual((stats["n"], stats["n_estimated"]), (1, 0))
        self.assertIsNone(stats["mean_pinball_s"])

    def test_usage_wait_runs_are_counted_but_not_scored(self):
        # why: an evidenced usage-limit wait is not a duration error; the estimator's labels exclude it.
        # catches: scoring censored usage-wait runs, or hiding them from the category counts.
        rows = [obs("opus", "build", age_days=1, actual=100, p80=100), obs("opus", "build", age_days=1, actual=9999, p80=100, category="usage-wait")]
        report = calibration.calibrate(rows, now=NOW, window_s=7 * DAY)
        self.assertEqual(group(report, "opus", "build")["this"]["n_estimated"], 1)
        self.assertEqual(report["categories"]["usage-wait"], 1)
        self.assertEqual(report["categories"]["ordinary"], 1)

    def test_orchestrator_turns_are_counted_apart_and_never_scored(self):
        # why: main-thread turns include owner waits; scoring or counting them as agent runs
        #      dominated group n and the kind_other count.
        # catches: grouping every T3 row regardless of category.
        rows = [obs("opus", "implementation", age_days=1, actual=900, p80=1000, source="t3-database")]
        rows += [obs("opus", "implementation", age_days=1, actual=20, p80=10, source="t3-database", category="orchestrator-turn") for _ in range(3)]
        report = calibration.calibrate(rows, now=NOW, window_s=7 * DAY)
        self.assertEqual(group(report, "opus", "implementation")["this"]["n"], 1)
        self.assertEqual(report["categories"]["orchestrator-turn"], 3)
        self.assertEqual(len(report["misses"]), 1)
        self.assertEqual(report["labels"]["kind_other"] + report["estimates"]["scored"], 1)

    def test_nested_subagents_are_counted_apart_not_as_a_second_sample(self):
        # why: a child run inside a scored parent run is already in the parent's actual.
        # catches: grouping the child next to its parent.
        parent = obs("opus", "implementation", age_days=1, actual=900, p80=1000, source="t3-database", native_session="s1")
        child = obs("sonnet", "review", age_days=1, actual=300, source="claude-subagent", parent_session="s1")
        child.update(start=parent["start"] + 10, end=parent["start"] + 310)
        report = calibration.calibrate([parent, child], now=NOW, window_s=7 * DAY)
        self.assertEqual([(g["model"], g["kind"]) for g in report["groups"]], [("opus", "*"), ("opus", "implementation")])
        self.assertEqual(report["observations"]["nested_subagents"], 1)

    def test_execution_count_is_the_collapsed_store_not_the_last_group(self):
        # why: a loop variable once shadowed the collapsed rows, so `executions` reported the size
        #      of the last thin group.
        # catches: reusing the collapsed-rows name inside the per-group loop.
        rows = [obs("opus", "fix", age_days=1, actual=60), obs("sol", "review", age_days=1, actual=60), obs("sol", "review", age_days=2, actual=60)]
        report = calibration.calibrate(rows, now=NOW, window_s=7 * DAY)
        self.assertEqual((report["observations"]["executions"], report["observations"]["agent_runs"]), (3, 3))

    def test_model_rollup_row_spans_its_kinds(self):
        # why: a new model with two kinds of thin history is judged as a whole too.
        # catches: groups only by (model, kind).
        rows = [obs("neon", "review", age_days=1, actual=100, p80=100), obs("neon", "fix", age_days=1, actual=200, p80=100)]
        report = calibration.calibrate(rows, now=NOW, window_s=7 * DAY)
        self.assertEqual(group(report, "neon", "*")["this"]["n"], 2)


class DriftFlags(unittest.TestCase):
    def test_p80_undercoverage_fires_with_enough_runs_and_not_before(self):
        # why: drift flags are evidence for a reader; a flag on 4 runs would be noise, and
        #      silence on 6 missed runs would hide the drift.
        # catches: no minimum n, or a threshold that never fires.
        # arithmetic: 6 runs, actual 200 vs p80 100 -> 0 of 6 covered (< .6); 4 runs -> too few.
        many = [obs("opus", "implementation", age_days=1, actual=200, p80=100, pointers={"n": i}) for i in range(6)]
        few = [obs("sol", "implementation", age_days=1, actual=200, p80=100, pointers={"n": i}) for i in range(4)]
        report = calibration.calibrate(many + few, now=NOW, window_s=7 * DAY)
        self.assertIn("p80-undercoverage", group(report, "opus", "implementation")["flags"])
        self.assertNotIn("p80-undercoverage", group(report, "sol", "implementation")["flags"])

    def test_loss_up_compares_relative_loss_with_earlier_weeks(self):
        # why: a model whose scaled error doubled is drifting even if its coverage looks fine.
        # catches: comparing absolute seconds across a change in run length, or ignoring history.
        # arithmetic: previous: actual 100, p80 100 -> loss 0; this: actual 200, p80 100 ->
        #   .8*100 = 80, relative 80/200 = .4 vs previous 0 -> up. Need >= 5 in each window.
        previous = [obs("opus", "fix", age_days=10 + i, actual=100, p80=100, pointers={"i": i}) for i in range(5)]
        this = [obs("opus", "fix", age_days=1, actual=200, p80=100, pointers={"i": i}) for i in range(5)]
        report = calibration.calibrate(previous + this, now=NOW, window_s=7 * DAY)
        self.assertIn("loss-up", group(report, "opus", "fix")["flags"])
        steady = calibration.calibrate(previous + [obs("opus", "fix", age_days=1, actual=100, p80=100, pointers={"i": i}) for i in range(5)], now=NOW, window_s=7 * DAY)
        self.assertNotIn("loss-up", group(steady, "opus", "fix")["flags"])


class NewModels(unittest.TestCase):
    def test_new_model_and_thin_kind_are_listed_with_their_history_counts(self):
        # why: Gemini 4 Neon has no history; the digest must say so instead of trusting a mean.
        # catches: listing only never-seen models, or listing every old thin kind (noise).
        old = [obs("opus", "implementation", age_days=30 + i, actual=100, source="t3-database", pointers={"i": i}) for i in range(12)]
        thin_old = [obs("opus", "research", age_days=40, actual=100, source="t3-database", pointers={"i": i}) for i in range(2)]
        new = [obs("neon", "review", age_days=1 + i, actual=100, source="t3-database", pointers={"i": i}) for i in range(3)]
        this_opus = [obs("opus", "research", age_days=2, actual=100, source="t3-database")]
        report = calibration.calibrate(old + thin_old + new + this_opus, now=NOW, window_s=7 * DAY)
        listed = {(n["model"], n["kind"]): n for n in report["new"]}
        self.assertEqual(listed[("neon", "*")]["reason"], "first-seen-in-window")
        self.assertEqual(listed[("neon", "*")]["runs"], 3)
        self.assertEqual(listed[("opus", "research")]["reason"], "thin-history")
        self.assertEqual(listed[("opus", "research")]["runs"], 3)
        self.assertNotIn(("opus", "implementation"), listed)  # 12 runs, none this week
        self.assertNotIn(("opus", "*"), listed)

    def test_history_counts_run_level_sources_once_not_hold_ledger_copies(self):
        # why: a vigil item and its T3 run are one execution; history volume counts runs.
        # catches: ledger rows inflating the history of a model that has only 2 real runs.
        runs = [obs("neon", "review", age_days=1, actual=100, source="t3-database", pointers={"i": i}) for i in range(2)]
        ledger = [obs("neon", "review", age_days=1, actual=100, p80=90, source="vigil-ledger", pointers={"i": i}) for i in range(20)]
        report = calibration.calibrate(runs + ledger, now=NOW, window_s=7 * DAY)
        listed = {(n["model"], n["kind"]): n for n in report["new"]}
        self.assertEqual(listed[("neon", "review")]["runs"], 2)


class Misses(unittest.TestCase):
    def test_largest_misses_are_ranked_and_carry_pointers_but_no_prompt_text(self):
        # why: a reader opens the top misses; each needs a way to find the run, and the report
        #      is pointers-only (brief: "Pointers, not prompt text").
        # catches: ranking by actual duration, dropping pointers, or leaking a brief field.
        # arithmetic (p80 only, q=.8): A actual 1000 vs 100 -> .8*900 = 720; B actual 130 vs 100 -> 24;
        #   C actual 20 vs 100 -> .2*80 = 16. Order A, B, C.
        a = obs("sol", "implementation", age_days=1, actual=1000, p80=100, pointers={"thread_id": "thread:A", "transcript": "/t/a.jsonl"}, brief="CANARY_BRIEF")
        b = obs("sol", "implementation", age_days=1, actual=130, p80=100, pointers={"vigil_hold": "h", "vigil_label": "lbl-B"})
        c = obs("sol", "implementation", age_days=1, actual=20, p80=100, pointers={"plan": "/p/plan.json", "task": "t9"})
        report = calibration.calibrate([c, b, a], now=NOW, window_s=7 * DAY)
        self.assertEqual([m["pointers"].get("thread_id") or m["pointers"].get("vigil_label") or m["pointers"].get("task") for m in report["misses"]], ["thread:A", "lbl-B", "t9"])
        for found, expected in zip([m["mean_pinball_s"] for m in report["misses"]], [720.0, 24.0, 16.0], strict=True):
            self.assertAlmostEqual(found, expected, places=9)  # float error only
        self.assertEqual(report["misses"][0]["pointers"]["transcript"], "/t/a.jsonl")
        self.assertEqual([m["direction"] for m in report["misses"]], ["over-run", "over-run", "finished-early"])
        self.assertNotIn("CANARY", json.dumps(report))

    def test_misses_list_is_limited_and_only_this_window(self):
        # why: the digest reads the top few, from this week only.
        # catches: an unbounded list, or last month's misses in this week's report.
        rows = [obs("sol", "fix", age_days=1, actual=100 + 50 * i, p80=100, pointers={"i": i}) for i in range(15)]
        rows.append(obs("sol", "fix", age_days=30, actual=99999, p80=100, pointers={"i": "old"}))
        report = calibration.calibrate(rows, now=NOW, window_s=7 * DAY, top=10)
        self.assertEqual(len(report["misses"]), 10)
        self.assertNotIn("old", [m["pointers"]["i"] for m in report["misses"]])


class Collapse(unittest.TestCase):
    def test_a_hold_and_its_plan_task_are_one_unscored_hold_without_a_run_link(self):
        # why: the vigil item and its plan task are one execution; scoring both double-weights
        #      it, and without a link to its run neither actual is the run's (the hold lifetime
        #      is the orchestrator's wait). The scored path is tests/test_learn.py VigilLinks.
        # catches: calibrate reading the raw store, or scoring an unlinked hold.
        vigil = obs("sol", "implementation", age_days=1, actual=900, p80=1200, source="vigil-ledger", pointers={"vigil_hold": "h"}, execution_key="E1")
        vigil["link_status"] = "no-hold-record"
        plan = obs("sol", "implementation", age_days=1, actual=900, p50=800, source="plan", label_source="vigil-hold", pointers={"plan": "/p"}, execution_key="E1")
        report = calibration.calibrate([vigil, plan], now=NOW, window_s=7 * DAY)
        self.assertEqual((report["groups"], report["holds"]), ([], {"no-hold-record": 1, "scored": 0}))
        self.assertEqual(report["observations"]["executions"], 1)


class Render(unittest.TestCase):
    def report(self):
        rows = [obs("opus", "implementation", age_days=1, actual=300, p80=100, pointers={"thread_id": "thread:X"}) for _ in range(6)]
        rows += [obs("neon", "review", age_days=1, actual=50, p80=100, source="t3-database")]
        return calibration.calibrate(rows, now=NOW, window_s=7 * DAY)

    def test_markdown_names_sections_flags_and_pointers(self):
        # why: the markdown is what the digest writer reads; it must carry the flags, the thin
        #      models and a pointer per miss, and say what the coverage targets are.
        # catches: a report that prints numbers without the evidence a reader needs to open a run.
        text = calibration.render_markdown(self.report())
        for needle in ("# agentpace calibration", "## Drift flags", "## New or thin", "## Largest misses", "p80-undercoverage", "thread:X", "neon", "target 50/80/90"):
            self.assertIn(needle, text)

    def test_group_table_lists_only_groups_with_an_estimate_and_counts_the_rest(self):
        # why: most runs carry no dispatch estimate; a table of dashes buries the groups that can
        #      be judged. The omitted ones are counted, and still in the JSON.
        # catches: listing every group, or dropping the unestimated ones without saying so.
        rows = [obs("opus", "implementation", age_days=1, actual=100, p80=100), obs("neon", "review", age_days=1, actual=50, source="t3-database")]
        report = calibration.calibrate(rows, now=NOW, window_s=7 * DAY)
        text = calibration.render_markdown(report)
        table = text.split("## Calibration by model and kind")[1].split("## Drift flags")[0]
        self.assertIn("| opus | implementation |", table)
        self.assertNotIn("| neon | review |", table)
        self.assertIn("2 groups with runs but no dispatch estimate are omitted", table)
        self.assertIn("1 of 2 scored runs this window had a dispatch estimate", text)
        self.assertIn(("neon", "review"), [(g["model"], g["kind"]) for g in report["groups"]])

    def test_markdown_of_an_empty_store_says_so(self):
        # why: a first run has no data; the report must say that, not print empty tables or crash.
        # catches: division by zero, or a table of Nones.
        text = calibration.render_markdown(calibration.calibrate([], now=NOW, window_s=7 * DAY))
        self.assertIn("No observations", text)


class Cli(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.state = Path(folder.name) / "state"
        learn.append_observations(self.state, [obs("opus", "fix", age_days=1, actual=300, p80=100, pointers={"run_id": "r1"})])

    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {"AGENTPACE_STATE": str(self.state)}), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["calibration", "--at", str(NOW), *args])
        return code, out.getvalue(), err.getvalue()

    def test_json_output_and_since_parsing(self):
        # why: the CLI is `calibration --since 7d --json`; 7d is a window of 604800 s.
        # catches: --since ignored, or 7d parsed as 7 seconds.
        code, out, _ = self.run_cli("--since", "7d", "--json")
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["window"]["since_s"], 604_800)
        self.assertEqual(group(report, "opus", "fix")["this"]["n"], 1)

    def test_since_units_and_rejection(self):
        # why: hours and weeks are accepted; garbage is a usage error, never a silent default.
        # catches: falling back to 7d on a bad value.
        self.assertEqual(calibration.parse_since("12h"), 43_200)
        self.assertEqual(calibration.parse_since("2w"), 1_209_600)
        code, _, err = self.run_cli("--since", "soon")
        self.assertEqual(code, 2)
        self.assertIn("--since", err)

    def test_md_is_the_default_and_out_writes_a_file(self):
        # why: the digest file is written by --out; md must be what lands in it.
        # catches: --out writing JSON, or printing instead of writing.
        target = self.state.parent / "digest" / "agentpace-calibration.md"
        code, out, _ = self.run_cli("--since", "7d", "--out", str(target))
        self.assertEqual(code, 0)
        self.assertTrue(target.read_text().startswith("# agentpace calibration"))
        self.assertNotIn("# agentpace calibration", out)


    def test_top_must_be_a_positive_count(self):
        # why: `--top -1` sliced misses[:-1] and exited 0 with one miss silently dropped.
        # catches: passing --top to a slice unchecked.
        for bad in ("0", "-1"):
            code, _, err = self.run_cli("--top", bad)
            self.assertEqual(code, 2, bad)
            self.assertIn("--top", err)

    def test_at_must_be_a_finite_time(self):
        # why: `--at nan` exited 0 with a NaN window in JSON and crashed the Markdown renderer.
        # catches: accepting any float() as the report time.
        for bad in ("nan", "inf", "-inf"):
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.dict(os.environ, {"AGENTPACE_STATE": str(self.state)}), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = cli.main(["calibration", f"--at={bad}", "--json"])
            self.assertEqual(code, 2, bad)
            self.assertIn("--at", err.getvalue())

    def test_out_file_is_owner_only(self):
        # why: the digest holds thread ids, transcript paths and vigil labels; written with the
        #      default umask it was world-readable (0644), unlike the store.
        # catches: Path.write_text for a private report, or keeping an existing file's 0644.
        target = self.state.parent / "digest.md"
        target.write_text("old")
        target.chmod(0o644)
        old = os.umask(0o022)
        try:
            code, _, _ = self.run_cli("--out", str(target))
        finally:
            os.umask(old)
        self.assertEqual((code, target.stat().st_mode & 0o777), (0, 0o600))
        self.assertTrue(target.read_text().startswith("# agentpace calibration"))


if __name__ == "__main__":
    unittest.main()
