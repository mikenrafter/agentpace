"""`--html`: agentpace state as one self-contained page that also carries the `--json` snapshot.

Cross-cutting: it drives the real CLI for `eta` and `calibration`, so it lives apart from the
per-module suites. Hand-derived numbers sit in each test's comment. Pages are read through the
stdlib HTML parser, never compared as strings.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import stat
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest import mock

from agentpace import cli, learn
from tests.test_calibration import NOW, obs

HOSTILE = "</script><img src=x onerror=alert(1)>"
COMMENT_TRAP = "<!--<script>"
VOID = {"meta", "br", "img", "input", "hr", "link"}


class Page(HTMLParser):
    """The page as a browser's tree builder sees it: tags, the data block, text per card."""

    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.scripts: list[tuple[dict, str]] = []
        self.cards: dict[str, dict] = {}
        self.roles: dict[str, str] = {}
        self._stack: list[tuple[str, dict]] = []
        self._script: list[str] | None = None
        self.feed(text)
        self.close()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.tags.append(tag)
        if tag in VOID:
            return
        self._stack.append((tag, attributes))
        if tag == "script":
            self._script = []
            self.scripts.append((attributes, ""))
        if tag == "article" and "data-task" in attributes:
            inside_details = any(name == "details" for name, _ in self._stack)
            self.cards[attributes["data-task"]] = {"text": "", "finished": inside_details}
        if "data-role" in attributes:
            self.roles.setdefault(attributes["data-role"], "")

    def handle_endtag(self, tag):
        while self._stack:
            name, attributes = self._stack.pop()
            if name == "script" and self._script is not None:
                self.scripts[-1] = (self.scripts[-1][0], "".join(self._script))
                self._script = None
            if name == tag:
                break

    def handle_data(self, data):
        if self._script is not None:
            self._script.append(data)
            return
        for name, attributes in self._stack:
            if name == "article" and "data-task" in attributes:
                self.cards[attributes["data-task"]]["text"] += data + " "
            if "data-role" in attributes:
                self.roles[attributes["data-role"]] += data + " "

    @property
    def snapshot(self):
        (attributes, body), = [s for s in self.scripts if s[0].get("type") == "application/json"]
        return json.loads(body)


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class EtaHtml(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.missing = str(self.root / "missing")

    def plan(self, tasks, name="plan", **top) -> Path:
        directory = self.root / name
        directory.mkdir()
        path = directory / "ship.json"
        path.write_text(json.dumps({"deliverable": "pilot-apk", "tasks": tasks, **top}))
        return path

    def eta(self, path, *extra):
        return run("eta", str(path), "--now", "100", "--config", self.missing, "--data", self.missing,
                   "--timings", self.missing, "--ledger", self.missing, *extra)

    def basic_tasks(self):
        # compile 3000 s then test 600 s after it: commit = 3000 + 600 = 3600 s -> "60 min"; setup is done.
        return [
            {"id": "setup", "type": "machine", "seconds": 1200, "basis": "measured", "status": "done",
             "started_at": 1, "ended_at": 1201, "actual_s": 1200},
            {"id": "compile", "type": "machine", "seconds": 3000, "basis": "measured", "depends_on": ["setup"]},
            {"id": "test", "type": "wait", "seconds": 600, "basis": "stated", "depends_on": ["compile"]},
        ]

    def test_embedded_snapshot_is_what_json_prints(self):
        # why: the agent reads the page's JSON instead of calling `--json`; they must be one object.
        # catches: embedding a different dict (before issued_id/plan_file are added, or a subset).
        # proves: CLAIM(html-json-parity)
        out_file = self.root / "eta.html"
        code, out, _ = self.eta(self.plan(self.basic_tasks()), "--json", "--html", str(out_file))
        self.assertEqual(code, 0)
        printed = json.loads(out)
        self.assertEqual(len(printed["tasks"]), 3)
        self.assertIn("issued_id", printed)
        self.assertEqual(Page(out_file.read_text()).snapshot, printed)

    def test_hostile_text_stays_text_and_json_round_trips(self):
        # why: plan text is untrusted: a task named `</script><img ...>` must neither end the data
        #      block early nor become a tag, and the JSON must still return the original string.
        # catches: escaping only the card text, or embedding json.dumps without the `</` escape.
        tasks = self.basic_tasks()
        tasks[1].update(id=HOSTILE, name=HOSTILE, note=HOSTILE, basis=HOSTILE)
        tasks[2]["depends_on"] = [HOSTILE]
        tasks[2].update(id=COMMENT_TRAP, note=COMMENT_TRAP)
        out_file = self.root / "eta.html"
        code, out, _ = self.eta(self.plan(tasks, deliverable=HOSTILE), "--json", "--html", str(out_file))
        self.assertEqual(code, 0)
        text = out_file.read_text()
        page = Page(text)
        self.assertNotIn("img", page.tags)
        self.assertEqual(page.tags.count("script"), 1)
        self.assertIn(HOSTILE, [task["id"] for task in page.snapshot["tasks"]])
        self.assertEqual(page.snapshot, json.loads(out))
        self.assertIn("&lt;/script&gt;&lt;img src=x onerror=alert(1)&gt;", text)
        # a `<!--` before `<script` would put a browser's script data in the double-escaped state.
        block = page.scripts[0][1]
        self.assertNotIn("</", block)
        self.assertNotIn("<!--", block)
        self.assertEqual(page.cards[HOSTILE]["finished"], False)
        self.assertIn(HOSTILE, page.cards[HOSTILE]["text"])

    def test_page_needs_nothing_from_the_network(self):
        # why: it must open from file:// and in T3's preview offline.
        # catches: a stylesheet link, an @import, a remote script or image.
        out_file = self.root / "eta.html"
        self.eta(self.plan(self.basic_tasks()), "--html", str(out_file))
        text = out_file.read_text().lower()
        self.assertGreater(len(text), 500)
        for forbidden in ("<link", 'src="http', "src='http", "@import", "url(http", "//cdn"):
            self.assertNotIn(forbidden, text)
        page = Page(out_file.read_text())
        self.assertEqual([attrs.get("src") for attrs, _ in page.scripts], [None])

    def test_html_writes_one_file_and_leaves_stdout_alone(self):
        # why: `--html` is an addition; scripts that read the text output must not notice it.
        # catches: printing a "wrote ..." line, writing beside the plan, or changing the text.
        plain = self.plan(self.basic_tasks(), name="plain")
        withhtml = self.plan(self.basic_tasks(), name="withhtml")
        out_dir = self.root / "out"
        out_dir.mkdir()
        _, plain_out, _ = self.eta(plain)
        before = sorted(p.name for p in withhtml.parent.iterdir())
        _, html_out, _ = self.eta(withhtml, "--html", str(out_dir / "eta.html"))
        self.assertEqual(html_out, plain_out)
        self.assertGreater(len(plain_out), 100)
        self.assertEqual([p.name for p in out_dir.iterdir()], ["eta.html"])
        # eta itself records the issued forecast next to the plan; --html adds nothing there.
        self.assertEqual(sorted(p.name for p in withhtml.parent.iterdir()),
                         sorted(set(before) | {"ship.state.json", "ship.state.lock"}))
        self.assertEqual(sorted(p.name for p in plain.parent.iterdir()),
                         sorted(p.name for p in withhtml.parent.iterdir()))

    def test_first_screen_carries_commit_median_and_slip(self):
        # why: the page is read in five seconds: commit, median and slip lead.
        # arithmetic: commit = 3000 + 600 = 3600 s -> render.duration "60 min"; machine/wait seconds
        #   are exact so P50 = P80 = 3600 s as well.
        # catches: putting task cards first, or dropping the slip (P80) figure.
        out_file = self.root / "eta.html"
        self.eta(self.plan(self.basic_tasks()), "--html", str(out_file))
        summary = " ".join(Page(out_file.read_text()).roles["summary"].split())
        self.assertRegex(summary, r"Commit\s+60 min")
        self.assertRegex(summary, r"Median\s+60 min")
        self.assertRegex(summary, r"Slip \(P80\)\s+60 min")
        self.assertIn("pilot-apk", summary)

    def test_one_card_per_open_task_and_finished_tasks_are_collapsed(self):
        # why: open work is the page; done work is context below it.
        # catches: a card for done tasks outside <details>, or missing cards for open tasks.
        out_file = self.root / "eta.html"
        self.eta(self.plan(self.basic_tasks()), "--html", str(out_file))
        cards = Page(out_file.read_text()).cards
        self.assertEqual({k: v["finished"] for k, v in cards.items()},
                         {"setup": True, "compile": False, "test": False})
        text = " ".join(cards["compile"]["text"].split())
        for fragment in ("todo", "machine", "aggressive", "P30", "50 min"):
            self.assertIn(fragment, text)
        done_text = " ".join(cards["setup"]["text"].split())
        self.assertIn("20 min", done_text)
        # a finished task has no commit cut or median left to show; "0 s" there would read as a forecast.
        self.assertNotIn("Commit cut", done_text)

    def test_agent_task_cards_name_model_and_kind(self):
        # why: for an agent task the person wants to know which model and what kind of work.
        # catches: reading model and kind from the forecast summary, which does not carry them.
        from agentpace import local_runs as lr

        out_file = self.root / "eta.html"
        directory = self.root / "agent"
        directory.mkdir()
        plan = directory / "ship.json"
        plan.write_text(json.dumps({"cwd": str(directory), "tasks": [
            {"id": "write", "model": "sonnet", "kind": "implementation"}]}))
        # twelve finished reference runs (the forecaster needs ten) of 600 s each; the forecast method is pinned so the home
        # directory's model policy cannot change it.
        data = directory / "runs.jsonl"
        data.write_text("".join(json.dumps(lr.make_row(start=1000 * i, end=1000 * i + 600, model="sonnet", kind="implementation",
                                                       project=str(directory), session=f"s{i}")) + "\n" for i in range(1, 13)))
        config = directory / "config.json"
        config.write_text(json.dumps({"forecast_method": "ridge"}))
        code, _, err = run("eta", str(plan), "--now", "100000", "--config", str(config), "--data", str(data),
                           "--timings", self.missing, "--ledger", self.missing, "--html", str(out_file))
        self.assertEqual(code, 0, err)
        text = " ".join(Page(out_file.read_text()).cards["write"]["text"].split())
        self.assertIn("sonnet", text)
        self.assertIn("implementation", text)

    def test_deferred_and_kept_tasks_are_labelled(self):
        # why: a shortcut that shaved the ETA must show on the task it shortened, and a kept gate
        #      must say it was kept.
        # catches: listing trade-offs only in JSON, or marking a kept gate as shortened.
        tasks = [{"id": "gate", "type": "machine", "seconds": 100, "basis": "measured"}]
        extras = dict(tradeoffs=[{"id": "targeted", "task": "gate", "seconds": 20, "basis": "measured subset",
                                  "risk": "less coverage", "protects": ["full-suite"], "savings_s": 80}],
                      apply=["targeted"])
        shortened, kept = self.root / "a.html", self.root / "b.html"
        self.eta(self.plan(tasks, name="a", **extras), "--html", str(shortened))
        self.eta(self.plan(tasks, name="b", **extras), "--keep", "full-suite", "--html", str(kept))
        shortened_text = " ".join(Page(shortened.read_text()).cards["gate"]["text"].split())
        kept_text = " ".join(Page(kept.read_text()).cards["gate"]["text"].split())
        self.assertIn("Shortcut applied", shortened_text)
        self.assertIn("saves 80 s", shortened_text)
        self.assertNotIn("Shortcut applied", kept_text)
        self.assertIn("Kept in full", kept_text)

    def test_unreliable_plan_says_so_above_the_cards(self):
        # why: research tasks are unbounded; the text output opens with that warning and the page must too.
        # catches: dropping the warning because it is only a print() in the text path.
        from agentpace import eta_html

        result = {"reliable": False, "commit": 60.0, "p50": 60.0, "p80": 90.0, "tasks": [], "tradeoffs": [],
                  "available_spend": [], "deadline_met": None, "unbounded_tasks": ["dig"], "elapsed_s": 0,
                  "actual_work_s": 0, "what_would_change_it": ["scope changes"]}
        page = Page(eta_html.render(result, {"deliverable": "x", "tasks": []}, keep=(), no_defer=()))
        self.assertIn("Low-confidence", page.roles["summary"])


class SharedHelper(unittest.TestCase):
    def test_json_block_escapes_what_ends_or_confuses_a_script(self):
        # why: embedding must survive `</script>` and `<!--<script>` and still decode to the input.
        # catches: copying site.embed's escape and forgetting the comment trap, or dumping raw.
        from agentpace import site

        data = {"a": HOSTILE, "b": COMMENT_TRAP, "c": "é "}
        block = site.json_block(data)
        self.assertNotIn("</", block)
        self.assertNotIn("<!--", block)
        self.assertEqual(json.loads(block), data)

    def test_site_embed_still_uses_the_same_block(self):
        # why: the leaderboard page and the state pages must share one escape.
        # catches: leaving a second copy of the escape inside site.embed.
        from agentpace import site

        shell = f"{site.START}old{site.END}"
        self.assertEqual(site.embed(shell, {"k": "</x>"}), f"{site.START}{site.json_block({'k': '</x>'})}{site.END}")

    def test_page_is_a_complete_document_without_body_background_or_side_padding(self):
        # why: T3's frame supplies the background and gutters; the host theme variables lead.
        # catches: setting body/html background or padding, or dropping the dark fallback.
        from agentpace import state_html

        text = state_html.page("T", ["<article class='card'>x</article>"], {"k": 1})
        self.assertTrue(text.lower().startswith("<!doctype html>"))
        self.assertIn('name="viewport"', text)
        css = "".join(re.findall(r"<style>(.*?)</style>", text, re.S))
        for variable in ("--card", "--border", "--foreground", "--muted-foreground", "--chart-1", "--radius", "--font-sans"):
            self.assertIn(f"var({variable},", css)
        self.assertIn("prefers-color-scheme: dark", css)
        for selector in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
            names, body = selector
            if re.search(r"(^|,)\s*(html|body)\s*(,|$)", names.strip()):
                self.assertNotRegex(body, r"background|padding")
        self.assertEqual(Page(text).snapshot, {"k": 1})
        self.assertEqual(Page(text).tags.count("article"), 1)


class CalibrationHtml(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.state = self.root / "state"
        self.rows = [
            # opus/fix: P80 100 against 300 s actuals: five misses, P80 coverage 0/5 -> flag.
            *[obs("opus", "fix", age_days=1 + i / 10, actual=300 + i, p80=100, pointers={"thread": f"t{i}"}) for i in range(5)],
            # a one-run model: first seen in the window -> listed as new.
            obs("haiku", "docs", age_days=1, actual=40, p80=60, pointers={"thread": HOSTILE}),
        ]
        learn.append_observations(self.state, self.rows)

    def cal(self, *extra):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {"AGENTPACE_STATE": str(self.state)}), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["calibration", "--at", str(NOW), *extra])
        return code, out.getvalue(), err.getvalue()

    def test_embedded_report_is_what_json_prints(self):
        # why: same contract as eta: the page's JSON is the report behind the markdown.
        # catches: embedding a trimmed summary instead of the full report.
        target = self.root / "cal.html"
        code, out, _ = self.cal("--json", "--html", str(target))
        self.assertEqual(code, 0)
        self.assertEqual(Page(target.read_text()).snapshot, json.loads(out))
        self.assertEqual(len(json.loads(out)["misses"]), 6)

    def test_summary_cards_show_coverage_misses_flags_and_thin_models(self):
        # why: the first read is coverage, the worst misses with where to open them, drift, thin history.
        # arithmetic: opus/fix 5 runs, all above P80 100 -> P80 coverage 0/5; flag p80-undercoverage
        #   (needs >= 5 scored, rate < 0.6). haiku/docs has 1 run -> first-seen-in-window.
        # catches: omitting the flag section, or a miss with no pointer.
        target = self.root / "cal.html"
        self.cal("--html", str(target))
        page = Page(target.read_text())
        for role in ("coverage", "misses", "flags", "new"):
            self.assertIn(role, page.roles)
        # headline: P80 hits 0/5 (opus) + 1/1 (haiku, 40 s <= 60 s) = 1/6 = 17%; each run once, so the
        #   per-kind rows must not be added on top of the per-model rollups.
        summary = " ".join(page.roles["summary"].split())
        self.assertIn("1/6", summary)
        self.assertIn("17%", summary)
        coverage = " ".join(page.roles["coverage"].split())
        self.assertIn("opus", coverage)
        self.assertIn("0/5", coverage)
        self.assertIn("p80-undercoverage", " ".join(page.roles["flags"].split()))
        self.assertIn("haiku", " ".join(page.roles["new"].split()))
        self.assertIn("thread=t0", " ".join(page.roles["misses"].split()))

    def test_hostile_pointer_is_text_and_page_is_offline(self):
        # why: pointers come from transcripts and vigil labels: untrusted text.
        # catches: interpolating a pointer unescaped into a card.
        target = self.root / "cal.html"
        self.cal("--html", str(target))
        text = target.read_text()
        page = Page(text)
        self.assertNotIn("img", page.tags)
        self.assertEqual(page.tags.count("script"), 1)
        self.assertIn("&lt;/script&gt;&lt;img src=x onerror=alert(1)&gt;", text)
        for forbidden in ("<link", 'src="http', "@import"):
            self.assertNotIn(forbidden, text.lower())

    def test_one_owner_only_file_and_stdout_unchanged(self):
        # why: the page holds thread ids and transcript paths like the markdown, so it is 0600,
        #      and it must not change what `calibration` prints.
        # catches: Path.write_text (0644), printing a notice, or writing next to the state.
        _, plain_out, _ = self.cal()
        out_dir = self.root / "out"
        out_dir.mkdir()
        before = sorted(p.name for p in self.state.rglob("*"))
        _, html_out, _ = self.cal("--html", str(out_dir / "cal.html"))
        self.assertEqual(html_out, plain_out)
        self.assertGreater(len(plain_out), 100)
        self.assertEqual([p.name for p in out_dir.iterdir()], ["cal.html"])
        self.assertEqual(stat.S_IMODE((out_dir / "cal.html").stat().st_mode), 0o600)
        self.assertEqual(sorted(p.name for p in self.state.rglob("*")), before)

    def test_empty_store_page_says_so(self):
        # why: honest absence: no observations is a sentence, not an empty chart.
        # catches: a crash or empty cards when the store has nothing.
        empty = self.root / "empty"
        out = self.root / "empty.html"
        with mock.patch.dict(os.environ, {"AGENTPACE_STATE": str(empty)}), contextlib.redirect_stdout(io.StringIO()):
            code = cli.main(["calibration", "--at", str(NOW), "--html", str(out)])
        self.assertEqual(code, 0)
        self.assertIn("No observations yet", " ".join(Page(out.read_text()).roles["summary"].split()))


if __name__ == "__main__":
    unittest.main()
