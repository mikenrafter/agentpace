"""Research reliability and one execution per observation."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
from agentpace import local_runs as lr,deliverable as dl

class ResearchTest(unittest.TestCase):
    def test_duplicate_run_keeps_earliest_dispatch_without_selecting_duration(self):
        # why: copied holds must not multiply one execution's loss or alter its quote.
        from agentpace.research import deduplicate
        a=dict(run_id='r',start=10,end=99,actual_s=89,expect_s=50)
        b=dict(run_id='r',start=20,end=30,actual_s=10,expect_s=900)
        c=dict(run_id='continuation',start=20,end=30,actual_s=10)
        self.assertEqual(deduplicate([b,c,a]),[a,c])

    def test_unknown_run_ids_do_not_merge_unrelated_holds(self):
        # why: lack of identity is not evidence that two observations are one run.
        from agentpace.research import deduplicate
        a=dict(session='a',start=1,end=5);b=dict(session='b',start=1,end=5)
        self.assertEqual(deduplicate([a,b,a]),[a,b])

    def test_run_cli_admits_unreliable_research(self):
        # why: the agent must see uncertainty through the command it actually calls.
        from agentpace.cli import main
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'runs.jsonl'
            lr.write_rows(path,[lr.make_row(start=0,end=100,kind='research',model='sol')])
            out=io.StringIO()
            with contextlib.redirect_stdout(out):
                main(['estimate-run','--model','sol','--kind','research','--method','reference','--data',str(path),'--json'])
            result=json.loads(out.getvalue())
        self.assertFalse(result['reliable'])
        self.assertEqual(result['research_history']['p90'],100)

    def test_remaining_research_prevents_confident_plan_total(self):
        # why: numeric simulated totals must not look like a bounded commitment.
        rows=[lr.make_row(start=0,end=100,kind='research',model='sol')]
        plan=dict(tasks=[dict(id='r',kind='research',model='sol')])
        result=dl.simulate(plan,rows,{'forecast_method':'reference'},draws=100)
        self.assertFalse(result['reliable'])
        self.assertEqual(result['unbounded_tasks'],['r'])
        self.assertFalse(result['tasks'][0]['reliable'])
        self.assertEqual(result['tasks'][0]['confidence'],'low')
        plan['tasks'][0].update(status='done',actual_s=100)
        self.assertTrue(dl.simulate(plan,rows,{},draws=100)['reliable'])
