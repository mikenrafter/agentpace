"""The eta page's schedule: the timeline rail, the faster card, and the numbers on each card.

Every figure here comes from `deliverable.timeline`/`speedups` through the eta snapshot; these tests
read the page the way a browser's tree builder sees it and compare it with the snapshot and with
hand-derived numbers.
"""
import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

from agentpace import cli
from agentpace import local_runs as lr
from tests.test_state_html import Page


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def percent(style, name):
    return float(re.search(rf"{name}:([\d.]+)%", style).group(1))


def colour(style, pattern):
    return re.search(pattern, style).group(1)


class EtaRail(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.missing = str(self.root / "missing")
        # twelve sonnet/docs runs of exactly 1200 s -> each docs task is 1200 s.
        self.data = self.root / "runs.jsonl"
        self.data.write_text("".join(json.dumps(lr.make_row(
            start=1000 * i, end=1000 * i + 1200, model="sonnet", kind="docs", project=str(self.root), session=f"s{i}")) + "\n"
            for i in range(1, 13)))
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"forecast_method": "ridge"}))

    def plan(self, tasks, name="ship", **top):
        path = self.root / f"{name}.json"
        path.write_text(json.dumps({"cwd": str(self.root), "deliverable": "demo", **top, "tasks": tasks}))
        return path

    def eta(self, plan, *extra):
        return run("eta", str(plan), "--now", "100", "--config", str(self.config), "--data", str(self.data),
                   "--timings", self.missing, "--ledger", self.missing, *extra)

    def demo(self):
        # setup (done, 600 s), a 1200 s and b 1200 s (independent docs agents), x a 1800 s machine step.
        # median schedule at cap 1: a 0-1200, b 1200-2400 (waits for the agent), x 0-1800; makespan 2400 s = 40 min.
        # one more agent would start b at 0 and finish everything at 1800 (x last): saves 600 s -> commit 30 min.
        # the catalogue can cut x to 600 s: alone it saves nothing at cap 1; with cap 2 the plan ends at 1200 s.
        tasks = [
            {"id": "setup", "type": "machine", "seconds": 600, "basis": "measured", "status": "done",
             "started_at": 1, "ended_at": 601, "actual_s": 600},
            {"id": "a", "model": "sonnet", "kind": "docs"},
            {"id": "b", "model": "sonnet", "kind": "docs"},
            {"id": "x", "type": "machine", "seconds": 1800, "basis": "measured"},
        ]
        tradeoffs = [{"id": "x-subset", "task": "x", "seconds": 600, "basis": "measured subset", "risk": "less coverage",
                      "savings_s": 1200, "cost": {"class": "none"}}]
        return self.plan(tasks, tradeoffs=tradeoffs)

    def page(self, plan=None, *extra):
        target = self.root / "eta.html"
        code, out, err = self.eta(plan or self.demo(), "--json", "--html", str(target), *extra)
        self.assertEqual(code, 0, err)
        return Page(target.read_text()), json.loads(out)

    def test_one_bar_per_open_task_and_none_for_finished_ones(self):
        # why: the rail is the schedule of the work still to do; setup is done and takes no time.
        # catches: a bar for the finished task, or a task with no bar.
        page, _ = self.page()
        self.assertEqual(sorted(page.bars), ["a", "b", "x"])
        self.assertEqual({k for k, v in page.cards.items() if not v["finished"]}, {"a", "b", "x"})

    def test_each_bar_is_the_colour_of_its_cards_left_border(self):
        # why: colour is the only link between the thin rail and the cards.
        # catches: colours assigned separately for bars and cards (so they drift), or bars all one colour.
        page, _ = self.page()
        card_colour = {k: colour(page.cards[k]["style"], r"--c:var\(--s-c(\d)\)") for k in page.bars}
        bar_colour = {k: colour(v, r"background:var\(--s-c(\d)\)") for k, v in page.bars.items()}
        self.assertEqual(card_colour, bar_colour)
        self.assertEqual(len(set(bar_colour.values())), 3, "three open tasks, three colours")

    def test_colours_cycle_through_the_six_chart_variables(self):
        # why: the owner's theme has six chart colours; the seventh task reuses the first.
        # catches: running past --s-c6, or reusing a colour inside the first six.
        tasks = [{"id": f"t{i}", "type": "machine", "seconds": 60, "basis": "stated"} for i in range(1, 8)]
        page, _ = self.page(self.plan(tasks, name="seven"))
        found = [colour(page.bars[f"t{i}"], r"background:var\(--s-c(\d)\)") for i in range(1, 8)]
        self.assertEqual(found, ["1", "2", "3", "4", "5", "6", "1"])
        self.assertEqual(colour(page.cards["t7"]["style"], r"--c:var\(--s-c(\d)\)"), "1")

    def test_bar_geometry_is_the_scheduler_s_start_and_end(self):
        # arithmetic: makespan 2400 s. a 0-1200 -> top 0%, height 50%. b 1200-2400 -> top 50%, height 50%.
        #   x 0-1800 -> top 0%, height 75%.
        # catches: scaling by the commit or the P80 instead of the median schedule's makespan, or bars that all start at the top.
        page, _ = self.page()
        got = {k: (percent(v, "top"), percent(v, "height")) for k, v in page.bars.items()}
        self.assertEqual(got, {"a": (0, 50), "b": (50, 50), "x": (0, 75)})

    def test_a_task_that_would_start_earlier_with_one_more_agent_has_a_ghost_bar(self):
        # arithmetic: b waits 1200 s for the agent; with a second agent it starts at 0 -> a ghost at top 0%, the
        #   same 50% long. a and x start at 0 already: no ghost.
        # catches: a ghost for every task, none at all, or a ghost at the bar's own start.
        page, _ = self.page()
        self.assertEqual(sorted(page.ghosts), ["b"])
        self.assertEqual((percent(page.ghosts["b"], "top"), percent(page.ghosts["b"], "height")), (0, 50))
        self.assertEqual(colour(page.ghosts["b"], r"color:var\(--s-c(\d)\)"), colour(page.bars["b"], r"background:var\(--s-c(\d)\)"))

    def test_overlapping_bars_sit_in_different_lanes(self):
        # why: a, b (with its ghost, which begins at 0) and x overlap in time; three bars on one line would hide each other.
        # catches: every bar at the same left offset.
        page, _ = self.page()
        lefts = {k: colour(v, r"left:([\d.]+rem)") for k, v in page.bars.items()}
        self.assertEqual(len(set(lefts.values())), 3, lefts)

    def test_rail_ticks_are_the_quarter_points_of_the_schedule(self):
        # arithmetic: makespan 2400 s: 0, 600 (10 min), 1200 (20 min), 1800 (30 min), 2400 (40 min) at 0, 25, 50, 75, 100 %.
        # catches: ticks at fixed pixel steps, or labelled in raw seconds.
        page, snapshot = self.page()
        self.assertEqual([text for _, text in page.ticks], ["0 s", "10 min", "20 min", "30 min", "40 min"])
        self.assertEqual([percent(style, "top") for style, _ in page.ticks], [0, 25, 50, 75, 100])
        self.assertEqual(snapshot["schedule"]["ticks_s"], [0, 600, 1200, 1800, 2400])

    def test_cards_say_when_each_task_runs_and_why_it_waits_or_is_critical(self):
        # arithmetic: a 0 s -> 20 min, b 20 min -> 40 min, x 0 s -> 30 min. Critical path = the longest dependency chain = x alone.
        #   b waits 20 min for the agent (queue, not dependency); one more agent starts it at 0 s.
        # catches: formatting the end as a duration, calling the queued task critical, no wait badge.
        page, _ = self.page()
        text = {k: " ".join(v["text"].split()) for k, v in page.cards.items()}
        self.assertIn("Runs 0 s → 20 min", text["a"])
        self.assertIn("Runs 20 min → 40 min", text["b"])
        self.assertIn("Runs 0 s → 30 min", text["x"])
        self.assertIn("waits for a free agent", text["b"])
        self.assertNotIn("waits for a free agent", text["a"])
        self.assertNotIn("waits for a free agent", text["x"], "a machine step never waits for an agent")
        self.assertIn("One more agent would start this at 0 s", text["b"])
        self.assertIn("critical path", text["x"])
        self.assertNotIn("critical path", text["a"] + text["b"])

    def test_faster_card_lists_computed_savings_with_the_new_commit(self):
        # arithmetic (see demo): cap 2 saves 600 s -> commit 1800 s; the x cut alone saves 0; both save 1200 s -> commit 1200 s.
        #   Durations print as 10 min, 30 min; 20 min, 20 min.
        # catches: listing the cut's declared 1200 s, summing the two, or leaving the card off.
        page, snapshot = self.page()
        card = " ".join(page.roles["speedups"].split())
        self.assertRegex(card, r"Agent cap 2.*10 min\s*→\s*30 min")
        self.assertRegex(card, r"All of these.*20 min\s*→\s*20 min")
        self.assertIn("Computed by the scheduler", card)
        self.assertIn("No effect on the commit", card)
        self.assertIn("x-subset", card)
        self.assertEqual([e["saves_s"] for e in snapshot["speedups"] if e["id"] in ("agent-cap", "combined")], [600, 1200])

    def test_faster_card_says_so_when_there_is_nothing(self):
        # why: an honest empty card is one line, not a padded list.
        # catches: omitting the card (the owner cannot tell "none" from "not computed") or listing zero-saving rows as savings.
        page, snapshot = self.page(self.plan([{"id": "only", "model": "sonnet", "kind": "docs"}], name="single"))
        card = " ".join(page.roles["speedups"].split())
        self.assertIn("No computed speed-up", card)
        self.assertNotIn("→", card)

    def test_embedded_snapshot_still_equals_json_with_the_new_keys(self):
        # why: CLAIM(html-json-parity) holds for the additions: schedule, speedups, per-task p80 and notes.
        # catches: embedding the snapshot before the schedule was added.
        page, printed = self.page()
        self.assertEqual(page.snapshot, printed)
        for key in ("schedule", "speedups"):
            self.assertIn(key, printed)
        self.assertTrue(all("p80" in t and "notes" in t for t in printed["tasks"]))

    def test_cards_show_the_per_task_p80(self):
        # arithmetic: constant history -> a's P80 is 1200 s = 20 min; x is exactly 1800 s = 30 min.
        # catches: P80 only in the JSON.
        page, _ = self.page()
        self.assertRegex(" ".join(page.cards["a"]["text"].split()), r"P80\s+20 min")
        self.assertRegex(" ".join(page.cards["x"]["text"].split()), r"P80\s+30 min")


if __name__ == "__main__":
    unittest.main()
