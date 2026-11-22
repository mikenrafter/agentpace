"""The scheduler as data: each task's start and end, the critical path, and the computed speed-ups.

Hand-derived numbers sit in each test's comment. The scheduler is the only source of every time and
saving the eta page shows, so these are the tests that matter most for it.
"""
import tempfile
import unittest
from pathlib import Path

from agentpace import deliverable as dl
from agentpace import local_runs as lr


def agent(task_id, *after):
    return {"id": task_id, "model": "sonnet", "kind": "docs", "depends_on": list(after)}


def machine(task_id, *after):
    return {"id": task_id, "type": "machine", "seconds": 1, "basis": "fixture", "depends_on": list(after)}


def spans(tasks, durations, cap):
    detail = dl.timeline(tasks, durations, cap)
    return {k: (v["start_s"], v["end_s"], v["wait_s"]) for k, v in detail["tasks"].items()}, detail["makespan_s"]


class Timeline(unittest.TestCase):
    CHAIN = [agent("a"), agent("b", "a"), agent("c", "b")]
    CHAIN_S = {"a": 10, "b": 20, "c": 30}
    # a 10 -> b 20 -> c 30: one path, so a second agent changes nothing.
    # starts 0, 10, 30; ends 10, 30, 60; nobody waits (each is ready the moment its agent is free).

    def test_chain_is_the_same_at_cap_1_and_cap_2(self):
        # why: dependencies, not the cap, order a chain; a second agent must not move it.
        # catches: starting a task before its dependency ends, or letting spare agents shorten a chain.
        for cap in (1, 2):
            found, makespan = spans(self.CHAIN, self.CHAIN_S, cap)
            self.assertEqual(found, {"a": (0, 10, 0), "b": (10, 30, 0), "c": (30, 60, 0)}, cap)
            self.assertEqual(makespan, 60)

    DIAMOND = [agent("a"), agent("b", "a"), agent("c", "a"), agent("d", "b", "c")]
    DIAMOND_S = {"a": 10, "b": 20, "c": 30, "d": 5}

    def test_diamond_at_cap_1_queues_the_second_branch(self):
        # arithmetic: a 0-10. b is ready at 10 and takes the agent: 10-30. c is ready at 10 but the
        #   agent is busy until 30, so c waits 20 and runs 30-60. d is ready at 60: 60-65.
        # catches: ignoring the cap (c at 10-40, makespan 45), or reporting no wait for c.
        found, makespan = spans(self.DIAMOND, self.DIAMOND_S, 1)
        self.assertEqual(found, {"a": (0, 10, 0), "b": (10, 30, 0), "c": (30, 60, 20), "d": (60, 65, 0)})
        self.assertEqual(makespan, 65)

    def test_diamond_at_cap_2_runs_the_branches_together(self):
        # arithmetic: a 0-10; b and c both ready at 10 and two agents are free: b 10-30, c 10-40;
        #   d is ready at max(30, 40) = 40 and an agent is free (b's, free since 30): 40-45.
        # catches: serialising the branches at cap 2, or d starting at b's end (30).
        found, makespan = spans(self.DIAMOND, self.DIAMOND_S, 2)
        self.assertEqual(found, {"a": (0, 10, 0), "b": (10, 30, 0), "c": (10, 40, 0), "d": (40, 45, 0)})
        self.assertEqual(makespan, 45)

    INDEPENDENT = [agent("x"), agent("y"), agent("z")]
    INDEPENDENT_S = {"x": 10, "y": 20, "z": 30}

    def test_independent_tasks_at_cap_1_and_cap_2(self):
        # arithmetic, cap 1: x 0-10, y 10-30 (waits 10), z 30-60 (waits 30).
        #   cap 2: x on agent 1 0-10, y on agent 2 0-20, z takes the first free agent (x's, at 10): 10-40, waits 10.
        # catches: a wait measured from time zero instead of from when the task was ready.
        self.assertEqual(spans(self.INDEPENDENT, self.INDEPENDENT_S, 1),
                         ({"x": (0, 10, 0), "y": (10, 30, 10), "z": (30, 60, 30)}, 60))
        self.assertEqual(spans(self.INDEPENDENT, self.INDEPENDENT_S, 2),
                         ({"x": (0, 10, 0), "y": (0, 20, 0), "z": (10, 40, 10)}, 40))

    def test_machine_steps_run_beside_agent_steps_and_never_wait(self):
        # arithmetic, cap 1: agents a (60) and b (60) share one agent: a 0-60, b 60-120 (waits 60).
        #   machine m (100 s) has no agent and no dependency: 0-100, wait 0. Makespan max(120, 100) = 120.
        #   machine n (100 s) after a starts at a's end: 60-160; makespan 160.
        # catches: giving machine steps an agent slot (m would run 120-220), or a wait for them.
        tasks = [agent("a"), agent("b"), machine("m"), machine("n", "a")]
        found, makespan = spans(tasks, {"a": 60, "b": 60, "m": 100, "n": 100}, 1)
        self.assertEqual(found, {"a": (0, 60, 0), "b": (60, 120, 60), "m": (0, 100, 0), "n": (60, 160, 0)})
        self.assertEqual(makespan, 160)

    def test_zero_length_tasks_take_no_agent(self):
        # why: a finished task has duration 0; it must not queue the next one behind it.
        # catches: a zero-duration agent task occupying a slot.
        found, makespan = spans([agent("done"), agent("next", "done")], {"done": 0, "next": 50}, 1)
        self.assertEqual(found, {"done": (0, 0, 0), "next": (0, 50, 0)})
        self.assertEqual(makespan, 50)

    def test_schedule_is_the_makespan_of_the_timeline(self):
        # why: `schedule` is what commit, P50 and P80 are built from; its numbers must not move.
        # catches: the refactor changing the makespan (hand values from the tests above).
        self.assertEqual(dl.schedule(self.DIAMOND, self.DIAMOND_S, 1), 65)
        self.assertEqual(dl.schedule(self.DIAMOND, self.DIAMOND_S, 2), 45)
        self.assertEqual(dl.schedule(self.INDEPENDENT, self.INDEPENDENT_S, 2), 40)


class CriticalPath(unittest.TestCase):
    def test_critical_path_is_the_longest_dependency_chain(self):
        # arithmetic: diamond a10 -> {b20, c30} -> d5: a-b-d = 35, a-c-d = 45. Chain: a, c, d.
        # catches: following the first dependency listed (b), or picking by task count.
        self.assertEqual(dl.critical_path(Timeline.DIAMOND, Timeline.DIAMOND_S), ["a", "c", "d"])

    def test_critical_path_does_not_depend_on_the_cap(self):
        # why: it is the chain that bounds the plan however many agents there are; a task that only
        #      queues for an agent is shown by its wait, not by this list.
        # catches: marking the queued task c critical at cap 1 by folding the cap in.
        self.assertEqual(dl.critical_path(Timeline.INDEPENDENT, Timeline.INDEPENDENT_S), ["z"])
        self.assertEqual(dl.critical_path(Timeline.CHAIN, Timeline.CHAIN_S), ["a", "b", "c"])

    def test_machine_step_on_the_chain_counts(self):
        # arithmetic: agent a 60 -> machine n 100 = 160 beats the independent agent b 60.
        tasks = [agent("a"), agent("b"), machine("n", "a")]
        self.assertEqual(dl.critical_path(tasks, {"a": 60, "b": 60, "n": 100}), ["a", "n"])


class SpeedUps(unittest.TestCase):
    """Computed what-ifs, through `simulate` with measured-looking history so the agent tasks are exact."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        # twelve sonnet/docs runs of exactly 1200 s: every docs task is forecast at 1200 s (cut, median and P80).
        self.rows = [lr.make_row(start=1000 * i, end=1000 * i + 1200, model="sonnet", kind="docs",
                                 project=str(self.root), session=f"s{i}") for i in range(1, 13)]
        self.config = {"forecast_method": "ridge"}

    def plan(self, **extra):
        # a and b: independent 1200 s agent tasks. x: a 1800 s machine step, off the agents.
        # cap 1: a 0-1200, b 1200-2400, x 0-1800 -> commit 2400 s.
        # cap 2: a 0-1200, b 0-1200, x 0-1800 -> 1800 s: saves 600.
        # catalogue: x could be cut to 600 s. At cap 1 that saves nothing (b still ends at 2400);
        #   with cap 2 and the cut: max(1200, 600) = 1200 -> saves 2400 - 1200 = 1200, more than the sum (600 + 0).
        base = {"cwd": str(self.root), "deliverable": "demo", "forecast_method": "ridge", "tasks": [
            agent("a"), agent("b"), {"id": "x", "type": "machine", "seconds": 1800, "basis": "measured"}],
            "tradeoffs": [{"id": "x-subset", "task": "x", "seconds": 600, "basis": "measured subset", "risk": "less coverage",
                           "savings_s": 1200, "cost": {"class": "none"}}]}
        base.update(extra)
        return base

    def result(self, plan=None, **kwargs):
        return dl.simulate(plan or self.plan(), self.rows, self.config, draws=100, now=2_000_000, **kwargs)

    def test_each_task_carries_its_start_end_and_wait_from_the_median_schedule(self):
        # arithmetic: medians a 1200, b 1200, x 1800 at cap 1: a 0-1200, b 1200-2400 (waits 1200), x 0-1800.
        #   Machine step x never waits. One more agent would start b at 0.
        # catches: the HTML layer having to compute these, or a schedule built from commit cuts instead of medians.
        schedule = self.result()["schedule"]
        self.assertEqual(schedule["makespan_s"], 2400)
        self.assertEqual(schedule["agent_cap"], 1)
        self.assertEqual(schedule["critical_path"], ["x"])
        self.assertEqual(schedule["ticks_s"], [0, 600, 1200, 1800, 2400])
        got = {k: (v["start_s"], v["end_s"], v["wait_s"], v["start_with_extra_agent_s"]) for k, v in schedule["tasks"].items()}
        self.assertEqual(got, {"a": (0, 1200, 0, 0), "b": (1200, 2400, 1200, 0), "x": (0, 1800, 0, 0)})

    def test_what_ifs_are_scheduler_makespan_differences(self):
        # why: every saving on the page is a makespan difference, never a sum of task savings.
        # arithmetic: see plan(): cap + 1 saves 2400 - 1800 = 600 (new commit 1800); cutting x alone saves 0;
        #   both: 2400 - 1200 = 1200 (new commit 1200), which is not 600 + 0.
        # catches: summing standalone savings for the combination, or reporting the cut's declared 1200 s as its ETA saving.
        result = self.result()
        self.assertEqual(result["commit"], 2400)
        by_id = {e["id"]: e for e in result["speedups"]}
        self.assertEqual(set(by_id), {"agent-cap", "x-subset", "combined"})
        self.assertEqual((by_id["agent-cap"]["saves_s"], by_id["agent-cap"]["commit_s"]), (600, 1800))
        self.assertEqual((by_id["x-subset"]["saves_s"], by_id["x-subset"]["commit_s"]), (0, 2400))
        self.assertEqual((by_id["combined"]["saves_s"], by_id["combined"]["commit_s"]), (1200, 1200))
        for entry in by_id.values():
            self.assertTrue(entry["computed"], entry)

    def test_what_ifs_equal_the_scheduler_called_directly(self):
        # why: the brief's oracle: the same dl.schedule the commit uses, on the same cuts, reproduces each saving.
        # catches: a private second scheduler inside the what-if code.
        result = self.result()
        tasks = self.plan()["tasks"]
        cuts = {"a": 1200, "b": 1200, "x": 1800}
        base = dl.schedule(tasks, cuts, 1)
        self.assertEqual(base, result["commit"])
        by_id = {e["id"]: e for e in result["speedups"]}
        self.assertEqual(by_id["agent-cap"]["saves_s"], base - dl.schedule(tasks, cuts, 2))
        self.assertEqual(by_id["x-subset"]["saves_s"], base - dl.schedule(tasks, dict(cuts, x=600), 1))
        self.assertEqual(by_id["combined"]["saves_s"], base - dl.schedule(tasks, dict(cuts, x=600), 2))

    def test_an_applied_shortcut_is_not_offered_again_and_a_kept_one_is_not_offered(self):
        # why: the card lists what is still available; an applied cut is already in the commit, and a gate
        #      the owner asked to keep is not a suggestion.
        # catches: listing the applied entry as a saving, or suggesting a cut that --keep protects.
        applied = self.result(self.plan(apply=["x-subset"]))
        self.assertNotIn("x-subset", {e["id"] for e in applied["speedups"]})
        protected = self.plan()
        protected["tradeoffs"][0]["protects"] = ["full-suite"]
        kept = self.result(protected, keep=["full-suite"])
        self.assertNotIn("x-subset", {e["id"] for e in kept["speedups"]})

    def test_no_what_if_when_nothing_could_be_faster(self):
        # why: a single agent task has no parallelism and no catalogue: the list is empty, not padded.
        # catches: inventing an entry (a zero-saving cap row listed as a speed-up).
        # arithmetic: one 1200 s task, cap 2 changes nothing -> 0 saved -> the agent-cap what-if is computed but saves 0.
        single = {"cwd": str(self.root), "forecast_method": "ridge", "tasks": [agent("only")]}
        result = self.result(single)
        self.assertEqual([e for e in result["speedups"] if e["saves_s"] > 0], [])

    def test_per_task_p80_comes_from_the_simulation(self):
        # why: eta --json gains each task's P80 beside its P50; the page shows it.
        # arithmetic: constant 1200 s history -> every draw of a is 1200 -> P80 1200; machine x is exactly 1800.
        # catches: copying the median into the P80 slot, or leaving it out of the snapshot.
        tasks = {t["id"]: t for t in self.result()["tasks"]}
        self.assertEqual((tasks["a"]["p50"], tasks["a"]["p80"]), (1200, 1200))
        self.assertEqual((tasks["x"]["p50"], tasks["x"]["p80"]), (1800, 1800))

    def test_p80_differs_from_the_median_when_the_history_spreads(self):
        # why: the constant-history test cannot tell P80 from P50; a spread can.
        # arithmetic: five measured timings 100..500 s, equal weight. weighted_quantile steps at cumulative weight
        #   q * 5: P50 needs 2.5 -> third value 300; P80 needs 4.0 -> fourth value 400; the balanced commit cut (P40) needs 2.0 -> 200.
        # catches: filling the P80 slot from the median draws.
        project = lr.identity(str(self.root.resolve()))
        timings = [{"project": project, "what": "build", "seconds": float(s)} for s in (100, 200, 300, 400, 500)]
        plan = {"cwd": str(self.root), "tasks": [{"id": "build", "type": "machine", "what": "build"}]}
        (task,) = dl.simulate(plan, [], {}, draws=100, now=2_000_000, timings=timings)["tasks"]
        self.assertEqual((task["commit"], task["p50"], task["p80"]), (200, 300, 400))

    def test_existing_json_keys_keep_their_meaning(self):
        # why: only additions this round.
        # catches: changing commit/p50/p80 while adding the schedule.
        result = self.result()
        self.assertEqual((result["commit"], result["p50"], result["p80"]), (2400, 2400, 2400))
        self.assertEqual(result["agent_cap"], 1)


if __name__ == "__main__":
    unittest.main()
