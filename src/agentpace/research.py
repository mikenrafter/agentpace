"""Execution identity and explicit limits of research-duration forecasts."""
from agentpace import run_forecast as rf


def identity(row):
    return ('run',row['run_id']) if row.get('run_id') else ('hold',row.get('session'),row['start'],row['end'])


def deduplicate(rows):
    """Keep the earliest hold for an execution; never select by quote or outcome."""
    seen=set();result=[]
    for row in sorted(rows,key=lambda r:(r['start'],str(r.get('session','')),r['end'])):
        key=identity(row)
        if key not in seen: result.append(row);seen.add(key)
    return result


def history(rows):
    measured=deduplicate([r for r in rows if r.get('kind')=='research' and r.get('censor_reason','none')=='none' and r.get('actual_s',0)>0])
    if not measured:return dict(n=0,p50=None,p90=None)
    return dict(n=len(measured),**{k:v for k,v in rf.forecast([(r['actual_s'],1) for r in measured]).items() if k in ('p50','p90')})


def warning(measured):
    values=f"historical P50 {measured['p50']:.0f}s, P90 {measured['p90']:.0f}s" if measured['n'] else 'no completed research history'
    return f'research: all bets are off (no reliable duration signal; {values}).'
