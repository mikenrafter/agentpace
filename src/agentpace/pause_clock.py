"""Observed activity proxy plus a separately forecast pause mixture.

An idle-capped event clock is not CPU time. Explicit waits override apparent
activity; recognized machine-tool intervals are protected from idle capping.
Unobserved gaps remain uncertain, not proof that an owner or provider paused.
"""
from __future__ import annotations
from agentpace import paths

import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from agentpace import local_runs as lr,run_forecast as rf

IDLE_CAP = 900


def union(spans,start,end):
    result=[]
    for a,b in sorted((max(start,a),min(end,b)) for a,b in spans if b>a):
        if b<=a: continue
        if result and a<=result[-1][1]: result[-1]=(result[-1][0],max(b,result[-1][1]))
        else: result.append((a,b))
    return result


def subtract(spans,removed):
    result=[]
    for a,b in spans:
        cursor=a
        for x,y in removed:
            if y<=cursor or x>=b: continue
            if x>cursor: result.append((cursor,x))
            cursor=max(cursor,y)
        if cursor<b: result.append((cursor,b))
    return result


def decompose(start,end,events,*,waits=(),signals=(),provider_spans=(),work_spans=(),tool_spans=(),idle_cap=IDLE_CAP):
    start,end=float(start),float(end)
    if not math.isfinite(start+end) or end<=start or idle_cap<=0: raise ValueError('positive finite span and idle cap required')
    times=sorted({float(t) for t in events if start<=t<=end})
    measured=len(times)>=2
    if measured:
        boundaries=sorted(set([start,end]+times))
        busy=[(a,min(b,a+idle_cap)) for a,b in zip(boundaries,boundaries[1:])]
        source='event-idle-cap'
    else:
        busy=list(provider_spans) or [(start,end)]
        source='provider-span-fallback' if provider_spans else 'wall-fallback'
    capped=union(busy,start,end)
    busy=union(busy+list(work_spans),start,end)
    protected=sum(b-a for a,b in busy)-sum(b-a for a,b in capped)
    explicit=union([(w['start'],w['end']) for w in waits],start,end)
    busy=subtract(busy,explicit)
    pauses=subtract([(start,end)],busy)
    pieces=[]
    for a,b in pauses:
        explicit_kinds=[w.get('kind','question') for w in waits if w['start']<b and w['end']>a]
        usage=[s for s in signals if s['kind']=='usage-limit' and a-idle_cap<=s['at']<=b]
        quiet_tool=any(x<b and y>a for x,y in tool_spans)
        if explicit_kinds: kind='/'.join(sorted(set(explicit_kinds)));confidence='explicit-request'
        elif usage: kind='usage-limit';confidence='text-associated-gap'
        elif b-a>=3600 and 0<=datetime.fromtimestamp((a+b)/2,(timezone.utc if paths.clock_timezone()=='UTC' else ZoneInfo(paths.clock_timezone()))).hour<7: kind='idle-overnight';confidence='clock-inferred'
        elif quiet_tool: kind='quiet-tool-unresolved';confidence='unresolved'
        else: kind='unobserved-gap';confidence='unresolved'
        pieces.append(dict(start=a,end=b,seconds=b-a,kind=kind,confidence=confidence))
    active=sum(b-a for a,b in busy)
    return dict(active_s=active,pause_s=end-start-active,wall_s=end-start,clock_source=source,pause_unidentified=not measured,event_count=len(times),idle_cap_s=idle_cap,protected_machine_s=max(0,protected),pauses=pieces,explicit_wait_s=sum(b-a for a,b in explicit))


def fit_pauses(rows):
    observed=[dict(pause_s=max(0,float(r['pause_s'])),model=r.get('model','unknown'),kind=r.get('kind','other'),project=r.get('project','unknown'),delegated=r.get('delegated'),budget_remaining_pct=r.get('budget_remaining_pct')) for r in rows if not r.get('pause_unidentified',False) and r.get('pause_s') is not None]
    if not observed: return dict(rows=[],unidentified=True)
    return dict(rows=observed,unidentified=False)


def _weighted_pauses(model,query):
    rows=model['rows']; budget=query.get('budget_remaining_pct')
    budget_rows=[r for r in rows if r['budget_remaining_pct'] is not None and (r['budget_remaining_pct']<=20)==(budget<=20)] if budget is not None else []
    use_budget=len(budget_rows)>=10
    pairs=[]
    for row in budget_rows if use_budget else rows:
        weight=1.0
        for name,multiplier in (('model',2),('kind',3),('project',2),('delegated',3)):
            if name in query and row.get(name)==query[name]: weight*=multiplier
        pairs.append((row['pause_s'],weight))
    return pairs,'dispatch-budget-bin' if use_budget else 'reference-base-rate'


def pause_term(model,query):
    pairs,basis=_weighted_pauses(model,query)
    total=sum(w for _,w in pairs)
    positive=[(v,w) for v,w in pairs if v>1]
    mass=sum(w for _,w in positive)
    probability=mass/total if total else 0
    mean=sum(v*w for v,w in positive)/mass if mass else None
    return dict(probability=probability,mean_if_pause_s=mean,expected_s=probability*mean if mean is not None else None,basis=basis,observations=len(pairs),positive_observations=len(positive),severity_identified=bool(positive),unidentified=model.get('unidentified',False))


def combine(active_pairs,model,query):
    term=pause_term(model,query);positive=[(v,w) for v,w in _weighted_pauses(model,query)[0] if v>1]
    if not positive: return list(active_pairs)
    p=term['probability']
    grid=[(rf.weighted_quantile(positive,(i+.5)/10),p/10) for i in range(10)]
    if p<1: grid.insert(0,(0,1-p))
    total=sum(w for _,w in active_pairs)
    return [(value+pause,weight/total*mass) for value,weight in active_pairs for pause,mass in grid if mass>0]


def category(row):
    if row.get('usage_wait_s',0)>1: return 'usage-wait'
    if row['actual_s']>43200: return 'over-12h'
    return 'ordinary'


def fit_long_risk(rows):
    count=len(rows);positive=sum(r['actual_s']>43200 for r in rows)
    return dict(probability=(positive+1)/(count+2),observations=count,positive_observations=positive,basis='earlier-run Beta(1,1) base rate')


def risk_terms(query,*,state=None):
    import json
    from pathlib import Path
    state=Path(state) if state else paths.home()
    usage_path=state/'usage-model.json';long_path=state/'long-risk.json'
    term=dict(available=False,basis='no fitted usage-wait history')
    if usage_path.exists():
        try:
            term=pause_term(json.loads(usage_path.read_text()),query)
            term['available']=True
        except (OSError,ValueError,KeyError,TypeError):
            term=dict(available=False,basis='no valid fitted usage-wait history')
    long=dict(available=False)
    if long_path.exists():
        try:
            saved=json.loads(long_path.read_text())
            if isinstance(saved,dict) and isinstance(saved.get('probability'),(int,float)) and 0<=saved['probability']<=1:
                long=saved
        except (OSError,ValueError): pass
    return dict(usage_wait=term,over_12h=long)


def usage_line(term):
    if not term.get('available'): return 'Usage-reset wait: no fitted history; excluded from the main ETA.'
    if not term.get('severity_identified'):
        return f"Usage-reset wait: none observed in {term['observations']} traced earlier runs; wait length unestimated. Excluded from the main ETA."
    return f"Usage-reset wait: {term['probability']:.1%} chance of about {term['mean_if_pause_s']/3600:.1f}h ({term['basis']}); excluded from the main ETA."
