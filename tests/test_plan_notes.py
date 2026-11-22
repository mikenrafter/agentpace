"""`agentpace plan note`: an agent's claims about a task, kept apart from the scheduler's numbers.

Cross-cutting (CLI, plan store, eta snapshot and page), so it lives beside the --html suite's helpers.
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from agentpace import cli
from agentpace import local_runs as lr
from tests.test_state_html import HOSTILE, Page


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class PlanNotes(unittest.TestCase):
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
        source = self.root / "source.json"
        # a 1200 s and b 1200 s (independent agents) and x, a 1800 s machine step: at cap 1 the commit is 2400 s;
        # one more agent saves 600 s (x then ends last at 1800).
        source.write_text(json.dumps({"id": "ship", "cwd": str(self.root), "deliverable": "demo", "tasks": [
            {"id": "a", "model": "sonnet", "kind": "docs"}, {"id": "b", "model": "sonnet", "kind": "docs"},
            {"id": "x", "type": "machine", "seconds": 1800, "basis": "measured"}]}))
        self.plan = self.root / "ship.state.json"
        code, _, err = run("plan", "init", str(self.plan), "--from", str(source), "--at", "1970-01-01T00:00:10+00:00")
        self.assertEqual(code, 0, err)

    def note(self, task, *extra):
        return run("plan", "note", str(self.plan), task, *extra, "--at", "1970-01-01T00:00:20+00:00")

    def eta(self, *extra):
        return run("eta", str(self.plan), "--now", "100", "--config", str(self.config), "--data", str(self.data),
                   "--timings", self.missing, "--ledger", self.missing, *extra)

    def notes_of(self, snapshot, task):
        (found,) = [t for t in snapshot["tasks"] if t["id"] == task]
        return found["notes"]

    def test_note_round_trips_into_eta_json_and_the_page(self):
        # why: the agent attaches a note once; the page and `eta --json` both show it under its task, with the
        #      same fields (kind, text, claim) and the time it was written.
        # catches: storing the note but not carrying it into the eta result, or into the card.
        code, out, err = self.note("a", "--kind", "parallel", "--text", "docs can start now", "--saves-s", "600")
        self.assertEqual(code, 0, err)
        page = self.root / "eta.html"
        code, out, err = self.eta("--json", "--html", str(page))
        self.assertEqual(code, 0, err)
        notes = self.notes_of(json.loads(out), "a")
        self.assertEqual(len(notes), 1)
        self.assertEqual({k: notes[0][k] for k in ("kind", "text", "saves_s", "at")},
                         {"kind": "parallel", "text": "docs can start now", "saves_s": 600.0, "at": 20.0})
        parsed = Page(page.read_text())
        self.assertEqual(self.notes_of(parsed.snapshot, "a"), notes)
        card = " ".join(parsed.cards["a"]["text"].split())
        self.assertIn("docs can start now", card)
        self.assertEqual(self.notes_of(json.loads(out), "b"), [], "other tasks carry an empty list, not the note")

    def test_a_claim_is_backed_only_by_what_the_scheduler_finds_for_that_task(self):
        # why: notes are the agent's claims. A cut is checked by shortening that task; a parallel claim by one more agent
        #      moving *this task's* start earlier. Plan-wide gain does not back a note on a task that cannot use the agent.
        # arithmetic (setUp plan): cap 1 commit 2400 (a 0-1200, b 1200-2400, x 0-1800). Cap 2: b starts at 0 and the plan
        #   ends at 1800: saves 600, and only b starts earlier (a was at 0, x is a machine step).
        #   b parallel 600: b starts earlier and the plan saves 600 -> backed. b parallel 5000: found 600 -> claim.
        #   a parallel 600: a starts at 0 either way -> not backed, found 0, although the plan-wide gain is 600.
        #   x parallel 600: a machine step takes no agent -> not backed, nothing to find (None), and it says why.
        #   cut b by 300: a 1200 + b 900 = 2100 > x 1800 -> commit 2100, saves 300 -> backed. cut x by 300 -> 1500: commit
        #   stays 2400 (b still ends last), saves 0 -> claim.
        # catches: comparing every parallel note with the plan-wide gain (the machine note and a's note come back backed),
        #   trusting --saves-s, or hiding the scheduler's own figure.
        self.note("b", "--kind", "parallel", "--text", "backed", "--saves-s", "600")
        self.note("b", "--kind", "parallel", "--text", "too big", "--saves-s", "5000")
        self.note("b", "--kind", "cut", "--text", "shorter", "--saves-s", "300")
        self.note("a", "--kind", "parallel", "--text", "already first", "--saves-s", "600")
        self.note("x", "--kind", "cut", "--text", "skip", "--saves-s", "300")
        self.note("x", "--kind", "parallel", "--text", "more agents", "--saves-s", "600")
        _, out, _ = self.eta("--json")
        snapshot = json.loads(out)
        parallel_ok, parallel_big, cut_b = self.notes_of(snapshot, "b")
        self.assertEqual((parallel_ok["backed"], parallel_ok["scheduler_saves_s"]), (True, 600))
        self.assertEqual((parallel_big["backed"], parallel_big["scheduler_saves_s"]), (False, 600))
        self.assertEqual((cut_b["backed"], cut_b["scheduler_saves_s"]), (True, 300))
        (parallel_a,) = self.notes_of(snapshot, "a")
        self.assertEqual((parallel_a["backed"], parallel_a["scheduler_saves_s"]), (False, 0))
        self.assertIn("start", parallel_a["why_not_backed"])
        cut_x, parallel_x = self.notes_of(snapshot, "x")
        self.assertEqual((cut_x["backed"], cut_x["scheduler_saves_s"], cut_x["why_not_backed"]), (False, 0, None))
        self.assertEqual((parallel_x["backed"], parallel_x["scheduler_saves_s"]), (False, None))
        self.assertIn("machine", parallel_x["why_not_backed"])

    def test_page_marks_claims_apart_from_computed_numbers(self):
        # why: the owner must see which figures the scheduler produced and which an agent asserted. The scheduler can
        #      only say "if the task saves N, the ETA moves by M": the page says "ETA effect checked", never
        #      "confirmed", and the task saving stays marked as the agent's claim even when the ETA effect holds.
        # catches: a note's saving rendered like a computed one, or the word "confirmed" on a claim.
        self.note("x", "--kind", "cut", "--text", "skip it", "--saves-s", "300")
        self.note("b", "--kind", "parallel", "--text", "start now", "--saves-s", "600")
        self.note("x", "--kind", "parallel", "--text", "more agents", "--saves-s", "600")
        page = self.root / "eta.html"
        self.eta("--html", str(page))
        cards = Page(page.read_text()).cards
        unbacked = " ".join(cards["x"]["text"].split())
        self.assertIn("claimed", unbacked)
        self.assertNotIn("ETA effect checked", unbacked)
        self.assertIn("ETA would move 0 s", unbacked)
        self.assertRegex(unbacked, r"machine step.*no agent")
        backed = " ".join(cards["b"]["text"].split())
        self.assertIn("ETA effect checked", backed)
        self.assertRegex(backed, r"claimed.*saves 10 min")
        self.assertNotIn("confirmed", " ".join(c["text"] for c in cards.values()))

    def test_claim_and_scheduler_that_round_to_the_same_text_show_seconds(self):
        # why: review nit 8: "claimed saves 10 min; scheduler finds 10 min" reads as agreement when it is not.
        # arithmetic: cut b by 610: a 1200 + b 590 = 1790 < x 1800 -> commit 1800, saves 600. Minutes: 610 s -> "10 min" and
        #   600 s -> "10 min", so both print in seconds. A cut of 700 saves 600 as well but prints "12 min" and
        #   "10 min": already different, no seconds needed.
        # catches: always printing seconds, or never.
        self.note("b", "--kind", "cut", "--text", "close call", "--saves-s", "610")
        self.note("b", "--kind", "cut", "--text", "far call", "--saves-s", "700")
        _, out, _ = self.eta("--json")
        close, far = self.notes_of(json.loads(out), "b")
        self.assertEqual((close["backed"], close["scheduler_saves_s"], far["scheduler_saves_s"]), (False, 600, 600))
        page = self.root / "eta.html"
        self.eta("--html", str(page))
        card = " ".join(Page(page.read_text()).cards["b"]["text"].split())
        self.assertIn("610 s", card)
        self.assertIn("600 s", card)
        self.assertIn("12 min", card)
        self.assertNotIn("700 s", card)

    def test_note_kind_without_a_claim_is_just_text(self):
        # catches: requiring --saves-s, or inventing a scheduler figure for a plain note.
        self.note("b", "--kind", "note", "--text", "waiting on review")
        _, out, _ = self.eta("--json")
        (note,) = self.notes_of(json.loads(out), "b")
        self.assertEqual((note["kind"], note["text"]), ("note", "waiting on review"))
        self.assertIsNone(note["saves_s"])
        self.assertIsNone(note["scheduler_saves_s"])
        self.assertFalse(note["backed"])

    def test_hostile_note_text_is_escaped_and_round_trips(self):
        # why: notes are agent-written text going into HTML and into a script block.
        # catches: an unescaped note ending the data script or injecting an element.
        self.note("a", "--kind", "note", "--text", HOSTILE)
        page = self.root / "eta.html"
        _, out, _ = self.eta("--json", "--html", str(page))
        text = page.read_text()
        parsed = Page(text)
        self.assertNotIn("img", parsed.tags)
        self.assertEqual(parsed.tags.count("script"), 1)
        self.assertIn("&lt;/script&gt;&lt;img src=x onerror=alert(1)&gt;", text)
        self.assertEqual(self.notes_of(parsed.snapshot, "a")[0]["text"], HOSTILE)
        self.assertEqual(self.notes_of(json.loads(out), "a")[0]["text"], HOSTILE)

    def test_refused_notes_leave_the_plan_untouched(self):
        # why: a bad call must not half-write. Each refusal exits 2 with a one-line reason.
        # catches: an unknown task appending to nothing silently, a claim on a plain note, a negative or
        #   non-finite saving, empty text, a note on finished work.
        before = self.plan.read_text()
        for args, reason in (
            (("nope", "--kind", "note", "--text", "t"), "unknown plan task"),
            (("a", "--kind", "note", "--text", "t", "--saves-s", "5"), "--saves-s"),
            (("a", "--kind", "cut", "--text", "t", "--saves-s", "-5"), "--saves-s"),
            (("a", "--kind", "cut", "--text", "t", "--saves-s", "nan"), "--saves-s"),
            (("a", "--kind", "cut", "--text", "  "), "text"),
        ):
            code, _, err = self.note(*args)
            self.assertEqual(code, 2, (args, err))
            self.assertIn(reason, err)
        self.assertEqual(self.plan.read_text(), before)
        run("plan", "start", str(self.plan), "x", "--at", "1970-01-01T00:00:30+00:00")
        run("plan", "done", str(self.plan), "x", "--at", "1970-01-01T00:00:40+00:00", "--took", "10")
        code, _, err = self.note("x", "--kind", "note", "--text", "late")
        self.assertEqual(code, 2)
        self.assertIn("open task", err)
        notes_in_plan = [t.get("agent_notes") for t in json.loads(self.plan.read_text())["tasks"]]
        self.assertEqual(notes_in_plan, [None, None, None])

    def test_notes_are_kept_in_the_order_written(self):
        # catches: replacing the list, or sorting it.
        for text in ("one", "two", "three"):
            self.note("a", "--kind", "note", "--text", text)
        _, out, _ = self.eta("--json")
        self.assertEqual([n["text"] for n in self.notes_of(json.loads(out), "a")], ["one", "two", "three"])

    def test_plan_note_refuses_a_hand_written_plan_and_leaves_it_alone(self):
        # why: review finding 5: a plan file with no `events` is a legacy ad-hoc plan; `eta` reads state from its
        #      `.state.json` companion, so notes written into it would be invisible, and the file was silently reformatted.
        # catches: rewriting the file (indent, agent_notes) or leaving a `.lock` beside it, or exiting 0.
        folder = self.root / "legacy"
        folder.mkdir()
        legacy = folder / "p.json"
        legacy.write_text(json.dumps({"id": "p", "cwd": str(self.root), "tasks": [{"id": "a", "model": "sonnet", "kind": "docs"}]}))
        before, entries = legacy.read_bytes(), sorted(path.name for path in folder.iterdir())
        code, _, err = run("plan", "note", str(legacy), "a", "--kind", "note", "--text", "hello")
        self.assertEqual(code, 2, err)
        self.assertIn("plan init", err)
        self.assertEqual(legacy.read_bytes(), before)
        self.assertEqual(sorted(path.name for path in folder.iterdir()), entries, "no .lock or temporary file appears")

    MALFORMED = (
        ({"text": "no kind"}, "kind"),
        ({"kind": "cut", "text": "string claim", "saves_s": "600"}, "saves_s"),
        ({"kind": "cut", "text": "negative", "saves_s": -5}, "saves_s"),
        ({"kind": "cut", "text": "flag", "saves_s": True}, "saves_s"),
        ({"kind": "note", "text": "claim on a plain note", "saves_s": 5}, "saves_s"),
        ({"kind": "cut", "saves_s": 5}, "text"),
        ({"kind": "cut", "text": "  "}, "text"),
        ({"kind": "huh", "text": "unknown kind"}, "kind"),
        ("just a string", "object"),
    )

    def test_storing_a_task_with_malformed_notes_is_refused_and_a_good_one_is_kept(self):
        # why: review finding 6: `plan add --task-json` and `plan scope --add-json` took any agent_notes, and the next `eta`
        #      crashed on them. Validate once, where they are stored, with the same rules as `plan note`.
        # catches: storing a note the reader cannot use; a refusal that half-writes the plan; a validator that rejects good notes.
        before = self.plan.read_text()
        for note, reason in self.MALFORMED:
            task = json.dumps({"id": "y", "model": "sonnet", "kind": "docs", "agent_notes": [note]})
            code, _, err = run("plan", "add", str(self.plan), "y", "--task-json", task)
            self.assertEqual(code, 2, (note, err))
            self.assertIn(reason, err)
            code, _, err = run("plan", "scope", str(self.plan), "--add-json", task, "--reason", "r")
            self.assertEqual(code, 2, (note, err))
        not_a_list = json.dumps({"id": "y", "model": "sonnet", "kind": "docs", "agent_notes": "oops"})
        self.assertEqual(run("plan", "add", str(self.plan), "y", "--task-json", not_a_list)[0], 2)
        self.assertEqual(self.plan.read_text(), before)
        good = json.dumps({"id": "y", "model": "sonnet", "kind": "docs", "agent_notes": [{"kind": "cut", "text": "fine", "saves_s": 300, "at": 5}, {"kind": "note", "text": "plain", "saves_s": None}]})
        code, _, err = run("plan", "add", str(self.plan), "y", "--task-json", good)
        self.assertEqual(code, 0, err)
        _, out, _ = self.eta("--json")
        self.assertEqual([n["text"] for n in self.notes_of(json.loads(out), "y")], ["fine", "plain"])

    def test_hand_edited_malformed_notes_are_skipped_with_a_visible_warning(self):
        # why: review finding 6: plans are JSON files agents edit by hand; a bad note must not take `eta` down (it was a
        #      KeyError traceback, exit 1). The good notes still show; each skipped one is named on the card and in the JSON.
        # arithmetic: b holds four entries: [valid plain note, no kind, string saves_s, a bare string]; entries 2-4 are skipped.
        #   a holds `agent_notes: "oops"`, not a list: skipped as a whole.
        # catches: an uncaught KeyError/TypeError, skipping silently, or dropping the valid note along with the bad ones.
        state = json.loads(self.plan.read_text())
        tasks = {t["id"]: t for t in state["tasks"]}
        tasks["b"]["agent_notes"] = [{"kind": "note", "text": "keep me", "saves_s": None, "at": 1}] + [m[0] for m in self.MALFORMED[:2]] + ["a bare string"]
        tasks["a"]["agent_notes"] = "oops"
        self.plan.write_text(json.dumps(state))
        page = self.root / "eta.html"
        code, out, err = self.eta("--json", "--html", str(page))
        self.assertEqual(code, 0, err)
        snapshot = json.loads(out)
        by_id = {t["id"]: t for t in snapshot["tasks"]}
        self.assertEqual([n["text"] for n in by_id["b"]["notes"]], ["keep me"])
        self.assertEqual(len(by_id["b"]["note_warnings"]), 3)
        for number in (2, 3, 4):
            self.assertTrue(any(f"agent note {number} skipped" in w for w in by_id["b"]["note_warnings"]), (number, by_id["b"]["note_warnings"]))
        self.assertEqual((by_id["a"]["notes"], len(by_id["a"]["note_warnings"])), ([], 1))
        self.assertIn("agent_notes", by_id["a"]["note_warnings"][0])
        self.assertEqual(by_id["x"]["note_warnings"], [], "a task with no problem carries an empty list")
        parsed = Page(page.read_text())
        self.assertEqual(parsed.snapshot, snapshot)
        card = " ".join(parsed.cards["b"]["text"].split())
        for warning in by_id["b"]["note_warnings"]:
            self.assertIn(warning, card)
        self.assertIn("keep me", card)
        self.assertIn(by_id["a"]["note_warnings"][0], " ".join(parsed.cards["a"]["text"].split()))


if __name__ == "__main__":
    unittest.main()
