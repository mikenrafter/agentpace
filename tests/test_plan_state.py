"""Persistent plan events and remaining clocks."""
import json
import tempfile
import unittest
from pathlib import Path
from agentpace import plan_state as ps, machine_timings as mt, deliverable as dl


class PlanStateTest(unittest.TestCase):
    def test_done_and_running_are_not_restarted(self):
        # why: an ETA asked mid-run must forecast only unfinished work.
        # catches: treating all tasks as fresh todo tasks.
        plan={'started_at':100,'tasks':[{'id':'a','type':'machine','seconds':50,'basis':'fixture','status':'done','started_at':100,'ended_at':140,'actual_s':40},{'id':'b','type':'machine','seconds':100,'basis':'fixture','status':'running','started_at':140,'depends_on':['a']}]}
        result=dl.simulate(plan,[],{},now=170,draws=100)
        self.assertEqual(result['p50'],70)
        self.assertEqual(result['total_p50'],140)
        self.assertEqual(result['actual_work_s'],40)

    def test_vigil_closes_once_and_eta_is_scored(self):
        # why: ledger rereads must not duplicate events or outcome scores.
        # catches: using ingestion time as completion or scoring every read.
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'ship.json'
            ps.create(path,{'id':'ship','tasks':[{'id':'suite','type':'machine','seconds':100,'basis':'fixture'}]},now=10)
            ps.update(path,'start','suite',now=10)
            ledger=Path(root)/'items.jsonl'
            ledger.write_text(json.dumps({'v':1,'hold':'h','label':'suite','started_at':10,'ended_at':70,'outcome':'done','meta':{'task':'ship/suite'}})+'\n')
            ps.sync_ledger(path,ledger)
            ps.sync_ledger(path,ledger)
            plan=ps.load(path)
            self.assertEqual(plan['tasks'][0]['actual_s'],60)
            self.assertEqual(sum(e['action']=='ledger-done' for e in plan['events']),1)
            ps.issue(path,{'commit':20,'p50':30,'p80':40},now=40)
            scores=ps.deliver(path,now=80)
            self.assertEqual(scores[0]['actual_remaining_s'],40)
            self.assertEqual(scores[0]['within_p80'],True)

    def test_record_and_import_do_not_make_up_durations(self):
        # why: expected time and unparsed prose are not observed machine work.
        # catches: importing estimate columns or dropping failed parse diagnostics.
        text='| Task | Took | Expected |\n|---|---|---|\n| Full suite | 3:56 | 5 min |\n| APK | unknown | 10 min |\n'
        records,unparsed=mt.import_text(text,'/demo','fixture')
        self.assertEqual(records[0]['seconds'],236)
        self.assertEqual(records[0]['what'],'suite:full')
        self.assertEqual(len(unparsed),1)
        with tempfile.TemporaryDirectory() as root:
            record=mt.record(Path(root)/'timings.jsonl',project='/demo',what='build:apk',host='remote:4cpu',took=120,now=1000,n_tests=50,contention=True)
            self.assertEqual((record['started_at'],record['ended_at']),(880,1000))
            self.assertEqual(record['n_tests'],50)

    def test_eta_snapshots_do_not_recursively_copy_forecasts(self):
        # why: repeated ETA asks must not grow snapshots exponentially.
        # catches: snapshotting a plan including its prior forecast snapshots.
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'ship.json'
            ps.create(path,{'tasks':[]},now=1)
            result={'commit':10,'p50':20,'p80':30}
            ps.issue(path,result,now=2,snapshot=ps.load(path))
            ps.issue(path,result,now=3,snapshot=ps.load(path))
            self.assertNotIn('forecasts',ps.load(path)['forecasts'][-1]['snapshot'])
