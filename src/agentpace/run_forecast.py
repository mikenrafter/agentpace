"""Reference-class distributions and elapsed-aware remaining forecasts.

Candidate selection belongs to experiments, on training data only. These
functions never inspect evaluation outcomes to choose a distribution.
"""
from __future__ import annotations

import math
import statistics
from agentpace import local_runs as lr


def context(query, history):
    previous=[r for r in history if max(r['end'],r.get('run_end',r['end']))<query.get('prediction_at',query['start']) and r.get('censor_reason','none')=='none']
    project=sorted((r for r in previous if r['project']==query['project']),key=lambda r:r['end'])
    session=[r for r in previous if r['session']==query['session'] and r.get('expect_s')]
    return dict(project_count=len(project),unfamiliar=int(not project),recent_s=statistics.median(r['actual_s'] for r in project[-10:]) if project else 0,prior_ratio=statistics.median(r['actual_s']/r['expect_s'] for r in session[-5:]) if session else 0,prior_count=len(session))


def weighted_quantile(pairs,q):
    pairs=sorted(pairs)
    total=sum(w for _,w in pairs)
    threshold=q*total; current=0
    for value,weight in pairs:
        current+=weight
        if current>=threshold:
            return value
    return pairs[-1][0]


def distribution(train,query,mode='reference',drop=()):
    if mode=='quantile':
        from agentpace import quantiles as qt
        model=query.get('_quantile_model') or qt.load_runtime()
        if 'reference' in model:
            return distribution(model['reference'],query,'reference',model['drop'])
        return qt.samples(model,query)
    usable=[r for r in train if r.get('censor_reason','none')=='none' and r['actual_s']>0]
    if not usable:
        raise ValueError('no uncensored completed reference runs')
    if mode=='ridge':
        model=lr.fit(usable)
        terms=lr.vector(query)
        center=sum(v*model['weights'].get(k,0) for k,v in terms.items())
        return [(math.exp(min(20,center+e)),1) for e in model['residual']]
    pairs=[]; current=context(query,usable)
    for row in usable:
        weight=1.0
        if 'kind' not in drop and row['kind']==query['kind']: weight*=4
        if 'model' not in drop and row['model']==query['model']: weight*=2
        if 'project' not in drop and row['project']==query['project']: weight*=3
        if 'verifiable' not in drop and row['features'].get('verifiable')==query['features'].get('verifiable'):
            weight*=1.25
        if 'human' not in drop and query['features'].get('human_minutes') and row['features'].get('human_minutes'):
            weight*=math.exp(-abs(math.log(query['features']['human_minutes']/row['features']['human_minutes'])))
        if 'scope' not in drop:
            distance=sum(abs(math.log1p(row['features'].get(k,0))-math.log1p(query['features'].get(k,0))) for k in ('items','files','phases'))
            weight*=math.exp(-.15*distance)
        if 'effort' not in drop and query.get('effort','unknown')!='unknown' and row.get('effort')==query.get('effort'):
            weight*=1.5
        if mode in ('recency','context') and 'recency' not in drop:
            age=max(0,query.get('prediction_at',query['start'])-row['end'])/86400
            weight*=max(.05,2**(-age/7))
        value=row['actual_s']
        if mode in ('expect','recency','context') and 'expect' not in drop and query.get('expect_s') and row.get('expect_s') and query.get('expect_is_prerun',True) and row.get('expect_is_prerun',True):
            value*=query['expect_s']/row['expect_s']
        if mode=='context' and 'prior' not in drop and current['prior_count']:
            past=context(row,usable)
            if past['prior_count'] and past['prior_ratio']:
                weight*=math.exp(-abs(math.log(current['prior_ratio']/past['prior_ratio'])))
        if mode=='context' and 'familiarity' not in drop:
            past=context(row,usable)
            if past['unfamiliar']==current['unfamiliar']: weight*=1.5
        pairs.append((max(1,value),weight))
    return pairs


def forecast(pairs):
    return {f'p{int(q*100)}':weighted_quantile(pairs,q) for q in lr.QUANTILES}


def remaining(samples,elapsed):
    if not math.isfinite(elapsed) or elapsed<0:
        raise ValueError('elapsed seconds must be finite and nonnegative')
    pairs=[(float(x),1) if isinstance(x,(int,float)) else x for x in samples]
    survivors=[(v-elapsed,w) for v,w in pairs if v>elapsed]
    if survivors:
        values=sorted(v for v,_ in survivors)
        # Equal-weight samples preserve the pilot's interpolated quantile convention.
        prediction={f'p{int(q*100)}':lr.quantile(values,q) for q in lr.QUANTILES} if len({w for _,w in survivors})==1 else forecast(survivors)
        return dict(prediction,support=len(survivors),extrapolated=False,pairs=survivors)
    # Outside empirical support: expose extrapolation instead of returning zero ETA.
    scale=max(1,weighted_quantile(pairs,.5))
    tail=[(-math.log(1-(i+.5)/100)*scale,1) for i in range(100)]
    return dict(forecast(tail),support=0,extrapolated=True,pairs=tail)
