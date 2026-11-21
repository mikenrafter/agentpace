"""Direct pinball fitting and timestamp-safe features."""
import unittest
from agentpace import quantiles as qt
from agentpace import local_runs as lr


class QuantilesTest(unittest.TestCase):
    def test_direct_quantiles_fit_asymmetric_distribution(self):
        # why: upper quantiles cannot be residual offsets around a shared mean.
        # catches: returning the mean or using the median for every quantile.
        rows=[dict(lr.make_row(start=i,end=i+v,model='sol'),features={}) for i,v in enumerate([10]*50+[100]*30+[1000]*20)]
        model=qt.fit(rows,rounds=0)
        result=qt.predict(model,dict(rows[0],start=10000))
        self.assertAlmostEqual(result['p50'],10)
        self.assertAlmostEqual(result['p80'],100)
        self.assertAlmostEqual(result['p90'],1000)

    def test_no_post_run_columns_enter_feature_matrix(self):
        # why: runtime/status/counter fields are targets, never predictor inputs.
        # catches: generic numeric-field expansion leaking tool counts or actuals.
        row=lr.make_row(start=100,end=200,model='sol')
        a=qt.features(row)
        changed=dict(row,actual_s=100000,end=100100,tool_calls=99999,outcome='failed')
        self.assertEqual(a,qt.features(changed))

    def test_repeated_split_learns_scope_difference(self):
        # why: the learned stumps must actually use pre-run scope to predict.
        # catches: always returning the intercept despite a predictive feature.
        rows=[]
        for i in range(80):
            row=lr.make_row(start=i,end=i+(10 if i%2==0 else 1000),model='sol')
            row['features']['items']=1 if i%2==0 else 20; rows.append(row)
        model=qt.fit(rows,rounds=15,rate=.3,min_leaf=10)
        self.assertLess(qt.predict(model,rows[0])['p50'],qt.predict(model,rows[1])['p50']/5)

    def test_simulator_grid_preserves_the_fitted_quantiles(self):
        # why: vigil P80 and plan inputs must agree with the scored direct model.
        # catches: midpoint grids that shift P80/P90 below their fitted values.
        from agentpace import run_forecast as rf
        rows=[lr.make_row(start=i,end=i+v,model='sol') for i,v in enumerate([10]*50+[100]*30+[1000]*20)]
        model=qt.fit(rows,rounds=0)
        self.assertEqual(rf.forecast(qt.samples(model,rows[0])),{'p50':10,'p80':100,'p90':1000})

    def test_delegate_weight_changes_pinball_mass_not_targets(self):
        # why: chat-sized root turns should not overwhelm the target agent-job population.
        # catches: accepting role weights but fitting unweighted quantiles.
        rows=[]
        for i in range(50):
            r=lr.make_row(start=i,end=i+(10 if i<40 else 1000),model='sol');r['delegated']=i>=40;rows.append(r)
        self.assertEqual(qt.fit(rows,rounds=0,delegate_weight=5)['models'][0]['intercept'],1000)

    def test_model_kind_interaction_has_opposite_effects(self):
        # why: a model's relative speed can differ for review and implementation.
        # catches: additive main effects that cannot fit opposite conditional effects.
        rows=[]
        for i in range(160):
            model,kind,duration=[('sol','review',10),('sol','implementation',1000),('opus','review',1000),('opus','implementation',10)][i%4]
            rows.append(lr.make_row(start=i,end=i+duration,model=model,kind=kind))
        fitted=qt.fit(rows,rounds=25,rate=.3,min_leaf=10,drop=('clock',))
        predictions=[qt.predict(fitted,r)['p50'] for r in rows[:4]]
        self.assertLess(predictions[0],predictions[1]/5)
        self.assertLess(predictions[3],predictions[2]/5)

    def test_live_annotation_preserves_pre_run_context_and_privacy(self):
        # why: the trained T3 feature representation must also exist at forecast time.
        # catches: live estimates silently lacking contention, effort or explicit difficulty.
        brief='PRIVATE_BRIEF human_minutes: 120\n- Review a.py'
        row=lr.make_row(start=100,end=101,model='sol',brief=brief)
        annotated=qt.annotate(row,brief,ordinal=2,effort='high',active_runs=4)
        self.assertEqual(annotated['features']['human_minutes'],120)
        self.assertEqual(annotated['features']['active_runs'],4)
        self.assertEqual(annotated['hints']['review'],1)
        self.assertEqual(qt.features(annotated)['continuation'],1)
        import json
        self.assertNotIn('PRIVATE_BRIEF',json.dumps(annotated))
