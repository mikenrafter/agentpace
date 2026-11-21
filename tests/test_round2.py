"""Round two invariants: labels, survival, causal history and spending."""
import unittest
from agentpace import run_labels as labels, run_forecast as forecast, deliverable as dl
from agentpace import local_runs as lr


class ForecastTest(unittest.TestCase):
    def test_survival_not_subtracted_median(self):
        # why: surviving past 150 rules out the 100-second completion.
        # catches: max(0, median(total)-elapsed).
        result=forecast.remaining([100,200,400],150)
        self.assertEqual(result['p50'],150)
        self.assertEqual(result['support'],2)
        self.assertGreater(forecast.remaining([100,200],500)['p80'],0)

    def test_history_uses_only_completions_before_start(self):
        # why: concurrent and future outcomes are not pre-run context.
        # catches: aggregating entire-session outcomes before forecasting a row.
        a=lr.make_row(start=1,end=11,session='s',project='p',expect_s=20)
        future=lr.make_row(start=2,end=1000,session='s',project='p',expect_s=1)
        query=lr.make_row(start=12,end=30,session='s',project='p')
        feature=forecast.context(query,[a,future])
        self.assertEqual(feature['prior_ratio'],.5)
        self.assertEqual(feature['project_count'],1)

    def test_matching_requires_unique_dispatch(self):
        # why: a preceding brief can refer to a different active delegate.
        # catches: selecting a nearest end time or ambiguous matching prompt.
        pair={'start':100,'end':500,'brief_time':90,'brief_hash':'x','model':'sol'}
        runs=[{'start':91,'end':190,'brief_hash':'x','model':'sol','run_id':'a'}]
        result=labels.match(pair,runs)
        self.assertEqual(result['actual_s'],99)
        self.assertEqual(result['label_source'],'transcript')
        self.assertEqual(labels.match(pair,runs+runs)['label_source'],'hold-fallback')

    def test_high_cost_needs_user_override(self):
        # why: a plan may not buy its way to an ETA automatically.
        # catches: accepting a high-cost entry merely because plan.apply names it.
        plan={'tasks':[{'id':'build','type':'machine','seconds':100,'basis':'observed'}], 'tradeoffs':[{'id':'large-sandbox','task':'build','seconds':20,'savings_s':80,'basis':'stated','risk':'more spend','cost':{'class':'high','sandbox_hours':1}}], 'apply':['large-sandbox']}
        blocked=dl.simulate(plan,[],{},draws=100)
        allowed=dl.simulate(plan,[],{},draws=100,allow_spend=['large-sandbox'],owner_grant='Owner: use a large sandbox for this build')
        self.assertEqual(blocked['commit'],100)
        self.assertEqual(blocked['available_spend'][0]['id'],'large-sandbox')
        self.assertEqual(allowed['commit'],20)

    def test_spend_flag_alone_is_not_authority(self):
        # why: the agent is the caller; a flag cannot authorize its own spend.
        # catches: treating --allow-spend as sufficient owner permission.
        plan={'tasks':[{'id':'b','type':'machine','seconds':100,'basis':'observed'}], 'tradeoffs':[{'id':'fast','task':'b','seconds':10,'savings_s':90,'risk':'spend','basis':'stated','cost':{'class':'high','tokens':10000}}], 'apply':['fast']}
        result=dl.simulate(plan,[],{},draws=100,allow_spend=['fast'])
        self.assertEqual(result['commit'],100)
        self.assertTrue(result['available_spend'][0]['needs_owner'])

    def test_continuations_are_separate_and_aborts_censored(self):
        # why: a thread_send continuation has its own pre-run brief and clock.
        # catches: counting a whole multi-turn delegate as one normal run.
        import json,tempfile
        from pathlib import Path
        records=[{'timestamp':0,'type':'session_meta','payload':{'id':'s'}},
                 {'timestamp':1,'type':'event_msg','payload':{'type':'task_started','turn_id':'a'}},
                 {'timestamp':1,'type':'turn_context','payload':{'model':'sol','effort':'high'}},
                 {'timestamp':2,'type':'event_msg','payload':{'type':'user_message','message':'Implement feature'}},
                 {'timestamp':11,'type':'event_msg','payload':{'type':'task_complete','turn_id':'a'}},
                 {'timestamp':21,'type':'event_msg','payload':{'type':'task_started','turn_id':'b'}},
                 {'timestamp':22,'type':'event_msg','payload':{'type':'user_message','message':'Review implementation'}},
                 {'timestamp':41,'type':'event_msg','payload':{'type':'turn_aborted','turn_id':'b','reason':'interrupted'}}]
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'rollout.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            rows=labels.codex_turns(path)
        self.assertEqual([r['actual_s'] for r in rows],[10,20])
        self.assertEqual(rows[1]['censor_reason'],'interrupted')
        self.assertNotEqual(rows[0]['brief_hash'],rows[1]['brief_hash'])

    def test_cap_suggestion_does_not_launch_parallel_work(self):
        # why: the target cannot override the user's configured agent cap.
        # catches: treating missing dependencies as permission for extra agents.
        rows=[lr.make_row(start=i,end=i+100,model='sol',kind='implementation',session=str(i)) for i in range(20)]
        plan={'forecast_method':'reference','tasks':[{'id':'a','model':'sol','kind':'implementation'},{'id':'b','model':'sol','kind':'implementation'}], 'tradeoffs':[{'id':'parallel','mechanism':'extra-agents','agent_cap':2,'basis':'stated overlap','risk':'twice concurrent usage','cost':{'class':'high','usage_window_pct':10}}], 'apply':['parallel']}
        blocked=dl.simulate(plan,rows,{},draws=100)
        allowed=dl.simulate(plan,rows,{},draws=100,allow_spend=['parallel'],owner_grant='Owner: use two parallel agents')
        self.assertEqual(blocked['commit'],200)
        self.assertEqual(blocked['available_spend'][0]['eta_gain_s'],100)
        self.assertEqual(allowed['commit'],100)

    def test_max_effort_cannot_be_a_speed_shortcut(self):
        # why: more reasoning is excluded from the accelerator catalogue.
        # catches: accepting it just because a row declares hypothetical savings.
        plan={'tasks':[{'id':'b','type':'machine','seconds':100,'basis':'observed'}], 'tradeoffs':[{'id':'max','mechanism':'max-reasoning','cost':{'class':'high','tokens':10000}}]}
        with self.assertRaises(ValueError): dl.simulate(plan,[],{},draws=100)

    def test_brief_matching_survives_transport_prefix(self):
        # why: T3 prepends transport instructions to a delegated user brief.
        # catches: sampling matching anchors on incompatible word boundaries.
        brief=' '.join('taskword'+str(i) for i in range(50))
        wrapped='Transport prefix context '+brief
        pair={'start':100,'end':500,'brief_time':90,'brief_hash':labels.fingerprint(brief),'brief_anchors':labels.anchors(brief),'model':'sol'}
        run={'start':91,'end':190,'brief_hash':labels.fingerprint(wrapped),'brief_anchors':labels.anchors(wrapped),'model':'sol','run_id':'a'}
        self.assertEqual(labels.match(pair,[run])['label_source'],'transcript')

    def test_late_estimate_is_not_a_pre_run_feature(self):
        # why: hold add can occur after the delegate has already started.
        # catches: using its later estimate to predict total duration at dispatch.
        row=lr.make_row(start=1,end=101,model='sol',expect_s=1000)
        query=lr.make_row(start=200,end=201,model='sol',expect_s=10000)
        row['expect_is_prerun']=False
        query['expect_is_prerun']=False
        self.assertEqual(forecast.forecast(forecast.distribution([row],query,'expect'))['p50'],100)

    def test_task_risk_increase_requires_owner_quote(self):
        # why: the calling agent cannot authorize a more aggressive commitment.
        # catches: applying per-task appetite before checking the configured layer.
        plan={'tasks':[{'id':'b','type':'machine','seconds':100,'basis':'measured','risk_appetite':'aggressive'}]}
        blocked=dl.simulate(plan,[],{'risk_appetite':'balanced'},draws=100)
        allowed=dl.simulate(plan,[],{'risk_appetite':'balanced'},draws=100,owner_grant='Owner: aggressive is okay for b')
        self.assertEqual(blocked['tasks'][0]['quantile'],.4)
        self.assertTrue(blocked['tasks'][0]['needs_owner'])
        self.assertEqual(allowed['tasks'][0]['quantile'],.3)

    def test_whitespace_is_not_an_owner_grant(self):
        # why: empty quoted input conveys no owner authorization.
        # catches: using string truthiness as evidence of consent.
        plan={'tasks':[{'id':'b','type':'machine','seconds':100,'basis':'measured','risk_appetite':'aggressive'}]}
        result=dl.simulate(plan,[],{'risk_appetite':'balanced'},draws=100,owner_grant='   ')
        self.assertTrue(result['tasks'][0]['needs_owner'])
