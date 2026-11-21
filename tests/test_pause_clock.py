"""Active/pause clocks use interval arithmetic and causal pause inputs."""
import unittest
from agentpace import pause_clock as pc


class PauseClockTest(unittest.TestCase):
    def test_capped_event_clock_conserves_wall_time(self):
        # why: a sparse overnight transcript cannot count the full silent gap as work.
        # catches: using the enclosing provider turn span as active time.
        result=pc.decompose(0,10000,[0,100,200,9900,10000],idle_cap=300)
        self.assertEqual(result['active_s'],600)
        self.assertEqual(result['pause_s'],9400)
        self.assertEqual(result['active_s']+result['pause_s'],10000)
        self.assertEqual(result['clock_source'],'event-idle-cap')

    def test_explicit_wait_overlaps_are_not_double_counted(self):
        # why: a question and approval may describe the same blocked interval.
        # catches: subtracting both waits independently or counting them as activity.
        waits=[{'start':20,'end':70,'kind':'question'},{'start':50,'end':90,'kind':'approval'}]
        result=pc.decompose(0,100,[0,10,20,50,70,90,100],waits=waits,idle_cap=900)
        self.assertEqual(result['pause_s'],70)
        self.assertEqual(result['explicit_wait_s'],70)
        self.assertEqual(result['active_s'],30)

    def test_provider_fallback_is_explicitly_uncertain(self):
        # why: missing event logs do not establish that a 10-hour provider span was work.
        # catches: fabricating an event clock or asserting no pauses without evidence.
        result=pc.decompose(0,1000,[],provider_spans=[(0,1000)])
        self.assertEqual(result['active_s'],1000)
        self.assertTrue(result['pause_unidentified'])

    def test_pause_model_ignores_post_run_query_labels(self):
        # why: pause facts from the run being predicted are never forecast features.
        # catches: using query.pause_s or pause_kind to choose its pause forecast.
        rows=[{'pause_s':0,'start':i,'end':i+10,'budget_remaining_pct':None} for i in range(8)]+[{'pause_s':18000,'start':20+i,'end':30+i,'budget_remaining_pct':None} for i in range(2)]
        model=pc.fit_pauses(rows)
        a=pc.pause_term(model,{'pause_s':999999,'pause_kind':'usage'})
        b=pc.pause_term(model,{})
        self.assertEqual(a,b)
        self.assertEqual(a['probability'],.2)
        self.assertEqual(a['mean_if_pause_s'],18000)
        self.assertEqual(a['expected_s'],3600)

    def test_wall_distribution_keeps_pause_probability_not_just_mean(self):
        # why: adding the expected pause to every run misstates the slip-line quantile.
        # catches: deterministic expected-wait addition instead of a mixture.
        model=pc.fit_pauses([{'pause_s':0}]*8+[{'pause_s':1000}]*2)
        pairs=pc.combine([(100,1)],model,{})
        from agentpace import run_forecast as rf
        result=rf.forecast(pairs)
        self.assertEqual(result,{'p50':100,'p80':100,'p90':1100})

    def test_machine_work_is_not_erased_by_idle_cap(self):
        # why: a quiet build tool can still be working throughout a long gap.
        # catches: reclassifying every silent tool interval as a pause to lower loss.
        result=pc.decompose(0,2000,[0,2000],work_spans=[(0,2000)],idle_cap=300)
        self.assertEqual(result['active_s'],2000)
        self.assertEqual(result['pause_s'],0)
        self.assertEqual(result['protected_machine_s'],1700)

    def test_usage_wait_skips_heartbeats_and_preserves_machine_work(self):
        # why: progress resumes with model activity, not a monitor heartbeat.
        # catches: trimming at the next arbitrary event or dropping concurrent work.
        from agentpace import activity_history as ah
        trace={'markers':[{'at':100,'kind':'usage-limit'}], 'progress':[50,1000], 'work_spans':[(100,300)]}
        waits=ah.usage_waits(trace,0,1200)
        self.assertEqual(waits,[{'start':300,'end':1000,'kind':'usage-limit'}])

    def test_user_quoted_limit_text_does_not_trigger_trimming(self):
        # why: the coding brief may itself discuss a usage-limit implementation.
        # catches: treating quoted task prose as an actual provider exhaustion event.
        import json,tempfile
        from pathlib import Path
        from agentpace import activity_history as ah
        records=[{'timestamp':1,'type':'event_msg','payload':{'type':'user_message','message':"You've hit your usage limit. Implement this message."}}, {'timestamp':2,'type':'event_msg','payload':{'type':'error','message':"You've hit your usage limit; resets in 5h"}}, {'timestamp':100,'type':'response_item','payload':{'type':'reasoning'}}]
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'session.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            trace=ah.parse_log(path)
        self.assertEqual([m['at'] for m in trace['markers']],[2])

    def test_categories_use_trimmed_duration_with_explicit_precedence(self):
        # why: usage-wait and >12h can overlap; the report's categories must not double-count.
        # catches: using untrimmed duration or assigning one run to two headline categories.
        self.assertEqual(pc.category({'usage_wait_s':100,'actual_s':50000}),'usage-wait')
        self.assertEqual(pc.category({'usage_wait_s':0,'actual_s':43201}),'over-12h')
        self.assertEqual(pc.category({'usage_wait_s':0,'actual_s':43200}),'ordinary')
        self.assertEqual(pc.category({'usage_wait_s':0,'actual_s':10,'wall_actual_s':50000}),'ordinary')

    def test_long_run_risk_is_forecast_not_oracle_membership(self):
        # why: realized duration cannot be known when estimating a run.
        # catches: returning the query's eventual category as its risk prediction.
        model=pc.fit_long_risk([{'actual_s':100}]*8+[{'actual_s':50000}]*2)
        self.assertEqual(model['probability'],.25)  # Beta(1,1): (2+1)/(10+2).

    def test_missing_usage_history_never_blocks_an_eta(self):
        # why: rare waits are an addendum, not a prerequisite for the main forecast.
        # catches: refusing ordinary estimates when the optional pause artifact is absent.
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as root:
            result=pc.risk_terms({},state=Path(root))
        self.assertFalse(result['usage_wait']['available'])
        self.assertEqual(result['usage_wait']['basis'],'no fitted usage-wait history')

    def test_damaged_optional_wait_model_does_not_block_main(self):
        # why: reading the optional artifact can race a local training write.
        # catches: a JSON parse failure preventing the ordinary ETA from being issued.
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as root:
            state=Path(root);(state/'usage-model.json').write_text('{partial')
            self.assertFalse(pc.risk_terms({},state=state)['usage_wait']['available'])

    def test_zero_observed_waits_does_not_invent_a_wait_length(self):
        # why: zero positive cases cannot calibrate conditional severity.
        # catches: presenting a zero-hour wait as a learned length.
        term=pc.pause_term(pc.fit_pauses([{'pause_s':0}]*10),{})
        self.assertEqual(term['probability'],0)
        self.assertIsNone(term['mean_if_pause_s'])
        self.assertIsNone(term['expected_s'])
        self.assertFalse(term['severity_identified'])

    def test_main_label_keeps_question_time_and_trims_only_usage_reset(self):
        # why: the owner wants other waits retained even when the active proxy is shorter.
        # catches: replacing the main label with active time after finding a reset.
        from agentpace import activity_history as ah
        row=dict(start=0,end=1000)
        trace=dict(events=list(range(0,1001,100)),progress=[0,900,1000],
                   markers=[dict(at=500,kind='usage-limit')],work_spans=[],
                   tool_spans=[],waits=[dict(start=100,end=300,kind='question')],
                   budgets=[])
        result=ah.describe(row,trace)
        self.assertEqual(result['usage_wait_s'],400)
        self.assertEqual(result['active_s'],400)
        self.assertEqual(result['pause_s'],600)
        self.assertEqual(result['trimmed_s'],600)

    def test_non_object_long_risk_artifact_cannot_block_eta(self):
        # why: an optional risk artifact is not required for the main forecast.
        # catches: text ETA formatting failing on a parseable but invalid risk object.
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as root:
            state=Path(root);(state/'long-risk.json').write_text('null')
            self.assertEqual(pc.risk_terms({},state=state)['over_12h'],{'available':False})
