"""Pilot invariants, independent of the owner's private transcripts."""
import json
import tempfile
import unittest
from pathlib import Path
from agentpace import local_runs as lr


class LocalRunsTest(unittest.TestCase):
    def test_pre_run_features_and_privacy(self):
        # why: post-run counters and raw prose must never enter prediction rows.
        # catches: storing the brief or using measured edit size.
        row = lr.make_row(start=100, end=160, model='gpt-6.1-sol', brief='- Update src/a.py\n- Test src/b.py\nPhase 2: ship', project='/tmp/demo', session='s')
        self.assertEqual(row['features']['items'], 2)
        self.assertEqual(row['features']['files'], 2)
        self.assertEqual(row['features']['phases'], 1)
        self.assertNotIn('Update', json.dumps(row))
        self.assertEqual(row['model'], 'sol')

    def test_ledger_fixture(self):
        # why: the separate vigil writer's literal v1 format must be accepted.
        # catches: using expect_s as the actual or accepting incomplete/negative runs.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'items.jsonl'
            fixture = {'v': 1, 'hold': 'h', 'label': 'review', 'spec_kind': 'task', 'expect_s': 600, 'started_at': '2026-11-01T00:00:00Z', 'ended_at': '2026-11-01T00:02:00Z', 'outcome': 'done', 'overdue_noted': False, 'session_id': 's', 'cwd': '/tmp/demo', 'meta': {'model': 'opus', 'kind': 'review'}}
            path.write_text(json.dumps(fixture)+'\n{broken\n'+json.dumps(dict(fixture, ended_at=None))+'\n')
            rows, skipped = lr.read_ledger(path)
        self.assertEqual((len(rows), skipped), (1, 2))
        self.assertEqual(rows[0]['actual_s'], 120)
        self.assertEqual(rows[0]['expect_s'], 600)

    def test_machine_is_added_once(self):
        # why: measured suite time is a separate additive term.
        # catches: fitting total time then adding machine duration again.
        rows = [lr.make_row(start=i, end=i+150, model='sol', kind='implementation', machine_s=100, session=str(i)) for i in range(30)]
        model = lr.fit(rows)
        query = dict(rows[0], machine_s=0)
        without = lr.predict(model, query)
        with_machine = lr.predict(model, dict(query, machine_s=100))
        self.assertAlmostEqual(without['p50'], 50, places=3)
        self.assertAlmostEqual(with_machine['p50']-without['p50'], 100, places=6)

    def test_session_split_never_leaks(self):
        # why: sessions spanning weeks must stay out of training until they finish.
        # catches: splitting individual rows by start only.
        rows = [lr.make_row(start=s, end=e, session=g) for s,e,g in [(1,2,'a'), (1,700000,'cross'), (800000,800001,'b')]]
        folds = lr.time_folds(rows, minimum=1)
        for _, train, test in folds:
            self.assertFalse({r['session'] for r in train} & {r['session'] for r in test})
            self.assertLess(max(r['end'] for r in train), min(r['start'] for r in test))

    def test_git_uses_base_not_worktree(self):
        # why: historical size must not see future edits.
        # catches: reading present-day file contents for historical features.
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            def git(*args):
                return subprocess.check_output(['git','-C',directory,*args], text=True).strip()
            git('init','-q'); git('config','user.name','Test'); git('config','user.email','test@example.com')
            path = Path(directory)/'a.py'; path.write_text('one\n')
            git('add','.'); git('commit','-qm','base'); base=git('rev-parse','HEAD')
            path.write_text('future\n'*100)
            features=lr.extract_features('Change a.py', directory, base)
            self.assertEqual(features['code_lines'], 1)
            self.assertEqual(features['git_known'], 1)

    def test_pinball_independent_arithmetic(self):
        # why: underprediction at P80 costs four times the same overprediction.
        # catches: absolute error presented as quantile loss.
        self.assertAlmostEqual(lr.pinball(100, 80, .8), 16)
        self.assertAlmostEqual(lr.pinball(100, 120, .8), 4)

    def test_codex_first_brief_not_final(self):
        # why: final response text cannot be used to classify a task.
        # catches: choosing the last user/assistant text as a brief.
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'rollout.jsonl'
            records=[{'timestamp':'2026-11-01T00:00:00Z','type':'session_meta','payload':{'id':'s','cwd':directory,'originator':'T3 Code'}}, {'timestamp':'2026-11-01T00:00:00Z','type':'event_msg','payload':{'type':'task_started','turn_id':'t'}}, {'timestamp':'2026-11-01T00:00:01Z','type':'event_msg','payload':{'type':'user_message','message':'Implement two files'}}, {'timestamp':'2026-11-01T00:00:02Z','type':'turn_context','payload':{'model':'gpt-6.1-sol'}}, {'timestamp':'2026-11-01T00:01:00Z','type':'event_msg','payload':{'type':'task_complete','turn_id':'t','last_agent_message':'Review done'}}]
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            rows=lr.read_codex(path)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['kind'],'implementation')
        self.assertEqual(rows[0]['actual_s'],60)

    def test_machine_catalogue_requires_matching_project(self):
        # why: a Flutter suite and Python suite have different machine costs.
        # catches: pooling every project into a generic full-suite duration.
        records=[{'project': lr.identity('/a'), 'label':'full suite', 'seconds':20}, {'project':lr.identity('/b'), 'label':'full suite','seconds':300}]
        self.assertEqual(lr.machine_time(records,'full suite','/a'),20)
        with self.assertRaises(ValueError):
            lr.machine_time(records,'APK','/a')

    def test_cli_expect_is_one_integer(self):
        # why: shell substitution into vigil cannot contain explanation text.
        # catches: printing the ordinary report with --expect-s.
        from agentpace.cli import main
        import io
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'rows.jsonl'
            lr.write_rows(path,[lr.make_row(start=i,end=i+60,model='sol',kind='implementation',session=str(i)) for i in range(20)])
            output=io.StringIO()
            with redirect_stdout(output):
                code=main(['estimate-run','--model','sol','--kind','implementation','--data',str(path),'--expect-s'])
            self.assertEqual(code,0)
            self.assertEqual(output.getvalue(),'60\n')

    def test_timing_table_actual_column_only(self):
        # why: expected durations in TIMINGS are not machine measurements.
        # catches: treating an estimate column as an observed suite time.
        text='| Check | Initial estimate | Actual | Notes |\n| --- | --- | --- | --- |\n| Full suite | 10 min | 8m08s | warm |\n| APK | 20 min | unknown | |\n'
        records=lr.parse_timings(text,'/demo','fixture')
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]['seconds'],488)
        self.assertEqual(records[0]['label'],'Full suite')

    def test_machine_only_run_has_no_agent_overhead(self):
        # why: --kind tests must return measured suite time, not suite plus coding.
        # catches: adding an agent prior to a pure machine task.
        rows=[lr.make_row(start=i,end=i+60,model='sol',kind='tests',session=str(i)) for i in range(20)]
        query=dict(rows[0],machine_s=100)
        self.assertEqual(lr.predict(lr.fit(rows),query)['p50'],100)

    def test_git_refuses_network_commands(self):
        # why: repository metadata extraction stays local even with bad callers.
        # catches: allowing fetch/push/clone through the subprocess helper.
        for operation in ('fetch','push','clone','-c'):
            with self.assertRaises(ValueError):
                lr.git('/tmp',operation)

    def test_simultaneous_items_are_not_duplicates(self):
        # why: one shell call can start several different vigil items together.
        # catches: deduplicating pair rows by timestamp/session alone.
        a=lr.make_row(start=1,end=61,session='same')
        b=lr.make_row(start=1,end=121,session='same')
        self.assertEqual(len(lr.merge_ledger([a,b],[])),2)
