"""Deliverable clocks and layered commitment policy."""
import unittest
from agentpace import deliverable as dl


class DeliverableTest(unittest.TestCase):
    def test_dependency_clock(self):
        # why: independent tasks overlap, gates wait for both.
        # catches: adding all durations despite explicit dependencies.
        tasks=[{'id':'a'},{'id':'b'},{'id':'gate','depends_on':['a','b']}]
        self.assertEqual(dl.makespan(tasks,{'a':10,'b':20,'gate':5}),25)
        with self.assertRaises(ValueError):
            dl.makespan([{'id':'a','depends_on':['a']}],{'a':1})

    def test_layered_appetite(self):
        # why: release gates retain conservative cuts inside an aggressive pilot.
        # catches: overwriting per-task policy with a single global quantile.
        config={'risk_appetite':'balanced','projects':{'/demo':{'risk_appetite':'conservative'}}}
        plan={'cwd':'/demo','deliverable':'pilot-apk','risk_appetite':'aggressive'}
        self.assertEqual(dl.appetite(config,plan,{'id':'code'})[:2],('aggressive',.3))
        self.assertEqual(dl.appetite(config,plan,{'id':'gate','risk_appetite':'conservative'})[:2],('conservative',.5))
        self.assertEqual(dl.appetite(config,plan,{'id':'gate','commit_quantile':.45})[1],.45)

    def test_plan_cut_from_task_cuts(self):
        # why: commit is built from task cuts while slip is from whole-plan draws.
        # catches: treating a sum of P80s as the plan's P80.
        plan={'deliverable':'pilot-apk','tasks':[{'id':'compile','type':'machine','seconds':100,'basis':'measured fixture'},{'id':'wait','type':'wait','seconds':20,'basis':'stated fixture','depends_on':['compile'],'risk_appetite':'conservative'}]}
        result=dl.simulate(plan,[],{},draws=100)
        self.assertEqual((result['commit'],result['p50'],result['p80']),(120,120,120))
        self.assertEqual([t['appetite'] for t in result['tasks']],['aggressive','conservative'])

    def test_tradeoff_override_restores_gate(self):
        # why: --keep full-suite restores the measured gate and its time.
        # catches: listing a trade-off without actually changing the simulation.
        plan={'tasks':[{'id':'gate','type':'machine','seconds':100,'basis':'measured'}], 'tradeoffs':[{'id':'targeted','task':'gate','seconds':20,'basis':'measured targeted set','risk':'less coverage','protects':['full-suite'],'savings_s':80}], 'apply':['targeted']}
        short=dl.simulate(plan,[],{},draws=100)
        full=dl.simulate(plan,[],{},draws=100,keep=['full-suite'])
        self.assertEqual(short['p50'],20)
        self.assertEqual(short['tradeoffs'][0]['savings_s'],80)
        self.assertEqual(full['p50'],100)
