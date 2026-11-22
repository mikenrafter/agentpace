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


class BindingPredecessor(unittest.TestCase):
    def test_each_task_records_what_set_its_start(self):
        # why: the critical path is the chain of binding predecessors; `bound_by` is the data it is read from.
        # arithmetic, diamond at cap 1 (see Timeline): a 0-10 starts at 0, nothing bound it. b is ready at 10 and the
        #   agent is free at 10: a dependency and a slot tie, the dependency wins -> a. c is ready at 10 but the agent
        #   is held by b until 30 -> b (the slot). d is ready at 60 (c's end) and the agent frees at 60: tie -> c.
        #   At cap 2: c takes the second agent at 10, bound by its dependency a; d is ready at 40 (c) and an agent is
        #   free since 30, so only c holds it back -> c.
        # catches: ignoring the slot (c bound by a), preferring the slot on a tie, or never recording it for a start at zero.
        bound = lambda cap: {k: v["bound_by"] for k, v in dl.timeline(Timeline.DIAMOND, Timeline.DIAMOND_S, cap)["tasks"].items()}
        self.assertEqual(bound(1), {"a": None, "b": "a", "c": "b", "d": "c"})
        self.assertEqual(bound(2), {"a": None, "b": "a", "c": "a", "d": "c"})

    def test_a_dependency_wins_a_tie_with_the_agent_slot(self):
        # arithmetic, cap 2: p and q (10 s each) take the two agents at 0 and end at 10. r depends on q: it is ready at 10 and
        #   an agent is free at 10 too, the one p held (the first free slot). Both end at 10; the dependency q is the binding
        #   predecessor, so the path is q, r (not p, r: p ended with nothing waiting for it).
        # catches: preferring the slot holder on a tie (path p, r), which would blame an unrelated task.
        tasks = [agent("p"), agent("q"), agent("r", "q")]
        durations = {"p": 10, "q": 10, "r": 10}
        self.assertEqual(dl.timeline(tasks, durations, 2)["tasks"]["r"]["bound_by"], "q")
        self.assertEqual(dl.critical_path(tasks, durations, 2), ["q", "r"])


class CriticalPath(unittest.TestCase):
    """The chain of binding predecessors back from the task that ends last: what really delays the commit."""

    def test_diamond_at_cap_2_is_the_longest_dependency_chain(self):
        # arithmetic: diamond a10 -> {b20, c30} -> d5 at cap 2: c ends at 40, b at 30; d waits for c: 40-45.
        #   d bound by c, c bound by a: a, c, d.
        # catches: following the first dependency listed (b), or picking by task count.
        self.assertEqual(dl.critical_path(Timeline.DIAMOND, Timeline.DIAMOND_S, 2), ["a", "c", "d"])

    def test_diamond_at_cap_1_runs_through_the_task_that_held_the_agent(self):
        # proves: CLAIM(critical-path)
        # arithmetic: cap 1: a 0-10, b 10-30, c 30-60 (the agent was b's until 30), d 60-65. d <- c <- b <- a.
        #   Cutting b really does move the end, so b is on the path although c does not depend on it.
        # catches: the pure dependency chain a, c, d at cap 1 (the owner-visible contradiction in review finding 1).
        self.assertEqual(dl.critical_path(Timeline.DIAMOND, Timeline.DIAMOND_S, 1), ["a", "b", "c", "d"])

    def test_independent_agent_tasks_queue_so_all_are_critical_at_cap_1_but_not_at_cap_2(self):
        # arithmetic: cap 1: x 0-10, y 10-30, z 30-60; z <- y <- x. cap 2: x 0-10 and y 0-20 in parallel, z takes x's
        #   agent at 10 and ends at 40, the plan's last end: z <- x. y ends at 20 and delays nothing.
        # catches: the old answer ["z"] at cap 1 (y and x ignored), or y kept on the path at cap 2.
        self.assertEqual(dl.critical_path(Timeline.INDEPENDENT, Timeline.INDEPENDENT_S, 1), ["x", "y", "z"])
        self.assertEqual(dl.critical_path(Timeline.INDEPENDENT, Timeline.INDEPENDENT_S, 2), ["x", "z"])

    def test_a_chain_is_the_path_at_any_cap(self):
        # arithmetic: a10 -> b20 -> c30 ends at 60 at cap 1 and 2; each task is ready the moment its agent is free.
        # catches: a path that depends on the cap for a plan with no parallelism.
        for cap in (1, 2):
            self.assertEqual(dl.critical_path(Timeline.CHAIN, Timeline.CHAIN_S, cap), ["a", "b", "c"], cap)

    def test_a_machine_step_with_slack_is_off_the_path(self):
        # arithmetic, cap 1: agents a and b are 1200 s each, machine x is 1800 s with no dependency. a 0-1200,
        #   b 1200-2400, x 0-1800. The plan ends at 2400, set by b queueing behind a; x has 600 s of slack.
        # catches: x on the path (the old longest-chain answer ["x"]) while its cut saves nothing.
        tasks = [agent("a"), agent("b"), machine("x")]
        self.assertEqual(dl.critical_path(tasks, {"a": 1200, "b": 1200, "x": 1800}, 1), ["a", "b"])

    def test_machine_step_on_the_chain_counts(self):
        # arithmetic: agent a 60 -> machine n 100 ends at 160; agent b (60 s) runs 60-120 beside it. n <- a.
        tasks = [agent("a"), agent("b"), machine("n", "a")]
        self.assertEqual(dl.critical_path(tasks, {"a": 60, "b": 60, "n": 100}, 1), ["a", "n"])

    def test_a_finished_task_at_the_head_of_the_chain_is_left_out(self):
        # why: done tasks have duration 0; they bound the next start at 0 and are not work left to do.
        # arithmetic: done 0 s -> next 50 s: next 0-50, bound by done. Path without zero-length tasks: next.
        # catches: dropping the `duration > 0` filter (the path would read ["done", "next"]).
        tasks = [agent("done"), agent("next", "done")]
        self.assertEqual(dl.critical_path(tasks, {"done": 0, "next": 50}, 1), ["next"])


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

    def probe_plan(self, tradeoffs):
        # cap 1: agents a, b, c at 1200 s each queue to 3600; machine x is 3000 s beside them.
        #   cap 2: a, b 0-1200, c 1200-2400; x 3000 -> 3000 (saves 600). cap 3: all three 0-1200; x 3000 -> 3000 (saves 600).
        return {"cwd": str(self.root), "deliverable": "demo", "forecast_method": "ridge", "tasks": [
            agent("a"), agent("b"), agent("c"), {"id": "x", "type": "machine", "seconds": 3000, "basis": "measured"}],
            "tradeoffs": tradeoffs}

    def cap_entry(self, entry_id, agent_cap):
        return {"id": entry_id, "mechanism": "extra-agents", "agent_cap": agent_cap, "cost": {"class": "high"}, "basis": "stated", "risk": "spend"}

    def x_cut(self, entry_id="xcut", cost="none", seconds=600):
        return {"id": entry_id, "task": "x", "seconds": seconds, "basis": "measured subset", "risk": "less coverage", "savings_s": 3000 - seconds, "cost": {"class": cost}}

    def result(self, plan=None, **kwargs):
        return dl.simulate(plan or self.plan(), self.rows, self.config, draws=100, now=2_000_000, **kwargs)

    def test_each_task_carries_its_start_end_and_wait_from_the_median_schedule(self):
        # arithmetic: medians a 1200, b 1200, x 1800 at cap 1: a 0-1200, b 1200-2400 (waits 1200), x 0-1800.
        #   Machine step x never waits. One more agent would start b at 0.
        # catches: the HTML layer having to compute these, or a schedule built from commit cuts instead of medians.
        schedule = self.result()["schedule"]
        self.assertEqual(schedule["makespan_s"], 2400)
        self.assertEqual(schedule["agent_cap"], 1)
        # critical path at cap 1: b ends last (2400), held back by a: a, b. Machine step x has 600 s of slack.
        self.assertEqual(schedule["critical_path"], ["a", "b"])
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

    def test_combined_is_computed_at_the_highest_cap_among_its_members(self):
        # why: review finding 3: "All of these" listed a cap-3 entry but was computed at cap 2.
        # arithmetic (probe_plan): base 3600. Cap 2 saves 3600 - 3000 = 600, cap 3 saves 600 (x ends last at 3000), the x
        #   cut alone saves 0 (the agents still queue to 3600). All three: cap 3 and x at 600 s -> max(1200, 600) = 1200:
        #   saves 2400, commit 1200. Computed at cap 2 it would be max(2400, 600) = 2400 (saves 1200).
        # catches: building the combination at cap + 1 whatever its members need.
        result = self.result(self.probe_plan([self.cap_entry("three", 3), self.x_cut()]))
        by_id = {e["id"]: e for e in result["speedups"]}
        self.assertEqual(set(by_id), {"agent-cap", "three", "xcut", "combined"})
        self.assertEqual((by_id["agent-cap"]["saves_s"], by_id["three"]["saves_s"], by_id["xcut"]["saves_s"]), (600, 600, 0))
        combined = by_id["combined"]
        self.assertEqual((combined["saves_s"], combined["commit_s"], combined["agent_cap"]), (2400, 1200, 3))
        self.assertEqual(combined["members"], ["agent-cap", "three", "xcut"])
        self.assertTrue(combined["needs_owner"], "it includes a high-cost cap entry")

    def test_two_cuts_on_one_task_combine_to_the_shorter(self):
        # why: "All of these" applies every cut; where two entries cut the same task the shorter one wins, not the later one.
        # arithmetic (probe_plan): cap 3 entry; x cut to 1500 (entry `deep`) and to 2000 (entry `shallow`, listed after it).
        #   All of them: cap 3 -> a, b, c in 0-1200; x at min(1500, 2000) = 1500 -> commit 1500, saves 3600 - 1500 = 2100.
        #   Taking the later entry (2000) would give commit 2000.
        # catches: the later cut overwriting the earlier, shorter one in the combination.
        plan = self.probe_plan([self.cap_entry("three", 3), self.x_cut("deep", seconds=1500), self.x_cut("shallow", seconds=2000)])
        combined = {e["id"]: e for e in self.result(plan)["speedups"]}["combined"]
        self.assertEqual((combined["commit_s"], combined["saves_s"]), (1500, 2100))

    def test_a_cap_row_with_no_catalogue_entry_needs_the_owner(self):
        # why: review finding 4: simulate refuses to raise the cap without an authorized extra-agents entry, so the
        #      default row must say needs_owner and carry the high cost class such an entry has to declare.
        # catches: the row claiming needs_owner False and cost none (an agent reads the JSON, not the prose).
        row = {e["id"]: e for e in self.result(self.probe_plan([]))["speedups"]}["agent-cap"]
        self.assertEqual((row["needs_owner"], row["cost"], row["catalogue_id"]), (True, {"class": "high"}, None))
        self.assertEqual((row["agent_cap"], row["saves_s"]), (2, 600))

    def test_a_catalogue_entry_for_the_next_cap_replaces_the_default_row_and_a_higher_one_adds_a_row(self):
        # arithmetic (probe_plan): cap 2 saves 600; the cap 3 entry saves 600 as well, x ends last at 3000. Two cap
        #   entries are rows 'agent-cap' (named by the entry `two`) and `three`; neither alone beats 600, so no 'combined'.
        # catches: ignoring extra-agents entries (the default row keeps catalogue_id None, `three` is missing), or
        #   adding the cap + 1 entry as a second row beside the default one.
        result = self.result(self.probe_plan([self.cap_entry("two", 2), self.cap_entry("three", 3)]))
        by_id = {e["id"]: e for e in result["speedups"]}
        self.assertEqual(set(by_id), {"agent-cap", "three"})
        self.assertEqual((by_id["agent-cap"]["catalogue_id"], by_id["agent-cap"]["cost"], by_id["agent-cap"]["needs_owner"]), ("two", {"class": "high"}, True))
        self.assertEqual((by_id["three"]["agent_cap"], by_id["three"]["cost"], by_id["three"]["needs_owner"], by_id["three"]["saves_s"]), (3, {"class": "high"}, True, 600))

    def test_a_cut_needs_the_owner_only_when_its_cost_is_high(self):
        # arithmetic (probe_plan): the x cut to 600 s saves 0 at cap 1 (a, b, c still queue to 3600); rows with no saving stay in the JSON.
        # catches: needs_owner False for every cut, or True for every cut.
        costly = {e["id"]: e for e in self.result(self.probe_plan([self.x_cut(cost="high")]))["speedups"]}["xcut"]
        free = {e["id"]: e for e in self.result(self.probe_plan([self.x_cut(cost="none")]))["speedups"]}["xcut"]
        self.assertEqual((costly["needs_owner"], costly["cost"]), (True, {"class": "high"}))
        self.assertEqual((free["needs_owner"], free["cost"]), (False, {"class": "none"}))

    def test_no_defer_removes_the_cut_on_that_task_from_the_what_ifs(self):
        # why: --no-defer x says x must not be shortened; the faster card must not suggest it.
        # arithmetic: plan(): without the flag the rows are agent-cap, x-subset, combined (see above); with x protected only agent-cap is left, and one row makes no combination.
        # catches: ignoring --no-defer in the what-ifs (x-subset and combined stay).
        self.assertEqual({e["id"] for e in self.result(no_defer=["x"])["speedups"]}, {"agent-cap"})

    def test_every_what_if_names_the_durations_it_used(self):
        # why: review nit 9: the rail uses medians and the savings use commit cuts; the data says which.
        # catches: a row without the basis (an agent cannot tell the two clocks apart).
        rows = self.result()["speedups"]
        self.assertEqual({e["duration_basis"] for e in rows}, {"commit cuts"})
        self.assertEqual(len(rows), 3)

    def test_existing_json_keys_keep_their_meaning(self):
        # why: only additions this round.
        # catches: changing commit/p50/p80 while adding the schedule.
        result = self.result()
        self.assertEqual((result["commit"], result["p50"], result["p80"]), (2400, 2400, 2400))
        self.assertEqual(result["agent_cap"], 1)


if __name__ == "__main__":
    unittest.main()
