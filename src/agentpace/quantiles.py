"""Direct pinball gradient-boosted stumps, with pre-run feature allowlists.

Each quantile has its own intercept and leaf quantile updates in seconds.
Candidate settings are selected only on earlier validation rows by experiments.
"""
from __future__ import annotations
from agentpace import paths

import math
from collections import Counter
from agentpace import local_runs as lr

GROUPS={'scope':('items','files','phases','words','brief_known'), 'git':('code_lines','age_days','churn','git_known'), 'contention':('active_runs','queue_wait_s'), 'human':('human_minutes','human_known')}
WORDS=('implement','review','fix','test','build','research','audit','migration','refactor','port','merge','apk','ui','screen','database','benchmark','compile','red','green','pilot','docs','lint','rebase','conflict','commit','suite','widget','nix','sandbox','android','flutter','rust','python','kotlin','typescript')


def features(row,drop=()):
    values={}
    for group,names in GROUPS.items():
        if group not in drop:
            for name in names: values[name]=math.log1p(max(0,row.get('features',{}).get(name,0)))
    for field in ('model','kind','project','provider','effort'):
        if field not in drop: values[field+'='+str(row.get(field,'unknown'))]=1.0
    if 'model' not in drop and 'kind' not in drop:
        values['model_kind='+str(row.get('model','unknown'))+'/'+str(row.get('kind','unknown'))]=1.0
    if 'project' not in drop and 'kind' not in drop:
        values['project_kind='+str(row.get('project','unknown'))+'/'+str(row.get('kind','unknown'))]=1.0
    if 'structure' not in drop:
        values['delegated']=float(row.get('delegated',False));values['continuation']=float(row.get('ordinal',1)>1)
    if 'clock' not in drop:
        hour=(row.get('prediction_at',row['start'])%86400)/86400
        values['hour_sin']=math.sin(2*math.pi*hour);values['hour_cos']=math.cos(2*math.pi*hour)
    if 'expect' not in drop:
        known=bool(row.get('expect_s') and row.get('expect_is_prerun',True))
        values['expect_known']=float(known)
        values['expect_log']=math.log1p(row['expect_s']) if known else 0.0
    if 'lexicon' not in drop:
        for word in WORDS: values['word:'+word]=math.log1p(row.get('hints',{}).get(word,0))
    return values


def order_quantile(ordered,q):
    return ordered[max(0,min(len(ordered)-1,math.ceil(q*len(ordered))-1))]


def fit(rows,rounds=60,rate=.15,min_leaf=25,drop=(),log_target=False,delegate_weight=1):
    rows=[r for r in rows if r.get('censor_reason','none')=='none' and r.get('actual_s',0)>0]
    if len(rows)<10: raise ValueError('need at least ten completed runs for quantile fitting')
    vectors=[features(r,drop) for r in rows];keys=sorted(set().union(*(v.keys() for v in vectors)))
    y=[math.log(r['actual_s']) if log_target else r['actual_s'] for r in rows]
    weights=[delegate_weight if r.get('delegated') else 1.0 for r in rows]
    total_weight=sum(weights)
    def leaf(indices,residual,q):
        ordered=sorted((residual[i],weights[i]) for i in indices)
        threshold=q*sum(w for _,w in ordered);running=0
        for value,weight in ordered:
            running+=weight
            if running>=threshold: return value
        return ordered[-1][0]
    n=len(y); splits=[]
    for key in keys:
        ordered=sorted(range(n),key=lambda i:vectors[i].get(key,0))
        positions=[]
        for i in range(min_leaf,n-min_leaf+1):
            low=vectors[ordered[i-1]].get(key,0);high=vectors[ordered[i]].get(key,0)
            if low!=high: positions.append((i,(low+high)/2))
        if len(positions)>24:
            positions=[positions[round(i*(len(positions)-1)/23)] for i in range(24)]
        if positions: splits.append((key,ordered,dict(positions)))
    fitted=[]
    for q in lr.QUANTILES:
        intercept=leaf(range(n),y,q); pred=[intercept]*n;trees=[]
        for _ in range(rounds):
            gradients=[w*(q if actual>guess else q-1 if actual<guess else 0.0) for actual,guess,w in zip(y,pred,weights)]
            total=sum(gradients);best=[];best_gain=1e-12
            for key,ordered,positions in splits:
                left=0;left_weight=0
                for position,index in enumerate(ordered,1):
                    left+=gradients[index];left_weight+=weights[index]
                    if position not in positions: continue
                    gain=left*left/left_weight+(total-left)**2/(total_weight-left_weight)-total*total/total_weight
                    if gain>best_gain+1e-10:
                        best_gain=gain;best=[(key,positions[position],ordered,position)]
                    elif gain>1e-12 and abs(gain-best_gain)<=1e-10:
                        best.append((key,positions[position],ordered,position))
            if not best: break
            residual=[actual-guess for actual,guess in zip(y,pred)]
            selected=None;best_improvement=0
            # Resolve tied gradient splits by actual pinball reduction, preventing
            # lexical tie order from repeatedly fitting an already-correct class.
            for key,threshold,ordered,position in best:
                left,right=ordered[:position],ordered[position:]
                lower=rate*leaf(left,residual,q);upper=rate*leaf(right,residual,q)
                improvement=sum(weights[i]*(lr.pinball(y[i],pred[i],q)-lr.pinball(y[i],pred[i]+lower,q)) for i in left)
                improvement+=sum(weights[i]*(lr.pinball(y[i],pred[i],q)-lr.pinball(y[i],pred[i]+upper,q)) for i in right)
                if improvement>best_improvement:
                    best_improvement=improvement;selected=(key,threshold,left,right,lower,upper)
            if selected is None: break
            key,threshold,left,right,lower,upper=selected
            for i in left: pred[i]+=lower
            for i in right: pred[i]+=upper
            trees.append((key,threshold,lower,upper))
        fitted.append(dict(q=q,intercept=intercept,trees=trees))
    return dict(models=fitted,drop=list(drop),log_target=log_target,rows=n,settings=dict(rounds=rounds,rate=rate,min_leaf=min_leaf,log_target=log_target,delegate_weight=delegate_weight))


def predict(model,row):
    vector=features(row,model['drop']);raw=[]
    for member in model['models']:
        value=member['intercept']+sum(left if vector.get(key,0)<=threshold else right for key,threshold,left,right in member['trees'])
        raw.append(max(.001,math.exp(min(25,value)) if model['log_target'] else value))
    # Quantiles are independently fitted; rearrangement prevents crossing at prediction.
    return dict(zip(('p50','p80','p90'),sorted(raw)))


def samples(model,row,count=100):
    """Interpolate fitted quantiles for the existing plan/conditional simulator."""
    p=predict(model,row);points=[(0,max(.001,p['p50']*.1)),(.5,p['p50']),(.8,p['p80']),(.9,p['p90']),(1,p['p90']+max(1,p['p90']-p['p80']))]
    pairs=[]
    for i in range(count):
        q=(i+1)/count
        for (low,a),(high,b) in zip(points,points[1:]):
            if low<=q<=high:
                fraction=min(1.0,max(0.0,(q-low)/(high-low)))
                pairs.append((a+(b-a)*fraction,1));break
    return pairs


def annotate(row,brief='',*,delegated=True,ordinal=1,effort='unknown',active_runs=None,human_minutes=None):
    """The live CLI/plan counterpart of T3's derived brief features."""
    import re
    from agentpace import t3_history
    words=re.findall(r'\w+',brief.casefold())
    row=dict(row,features=dict(row['features']),hints={word:words.count(word) for word in WORDS},delegated=delegated,ordinal=ordinal,effort=effort,provider='codex' if row['model'] in ('sol','codex') else 'claudeAgent')
    if human_minutes is None:
        matched=re.search(r'\bhuman[_ -]minutes\s*[:=]\s*(\d+(?:\.\d+)?)',brief,re.I)
        human_minutes=float(matched[1]) if matched else 0
    row['features'].update(human_minutes=human_minutes,human_known=int(bool(human_minutes)))
    if active_runs is None: active_runs=t3_history.active_at(row.get('prediction_at',row['start']))
    row['features']['active_runs']=active_runs or 0
    row['contention_unknown']=active_runs is None
    return row


def load_runtime():
    import json
    from pathlib import Path
    state=paths.home()
    policy=json.loads((state/'model-policy.json').read_text()) if (state/'model-policy.json').exists() else {}
    path=Path(policy.get('model_file',str(state/'quantile-model.json')))
    if not path.exists(): raise ValueError('no fitted quantile model; run the local training experiment first')
    return json.loads(path.read_text())
