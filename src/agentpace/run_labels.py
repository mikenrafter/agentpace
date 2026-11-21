"""Transcript-backed labels with conservative brief/dispatch matching.

Matching uses information available at dispatch, never a run's duration.
Raw prose remains in memory; persisted indices contain fingerprints and counts.
"""
from __future__ import annotations

import re
from pathlib import Path
from agentpace import local_runs as lr


def fingerprint(brief):
    words=re.findall(r'\w+',brief.lower())
    return lr.identity(' '.join(words))


def anchors(brief):
    words=re.findall(r'\w+',brief.lower())
    return sorted({lr.identity(' '.join(words[i:i+8])) for i in range(0,max(0,len(words)-7))})


def finish(turn, path, meta, source):
    start,end=turn['start'],turn.get('end') or turn.get('last',turn['start'])
    if start is None or end is None or lr.epoch(end)<=lr.epoch(start):
        return None
    row=lr.make_row(start=start,end=end,model=turn.get('model',''),brief=turn.get('brief',''),project=meta.get('cwd',''),session=meta.get('id',str(path)),source=source)
    row.update(run_id=lr.identity((str(path),turn.get('id'),start)),brief_hash=fingerprint(turn.get('brief','')),brief_anchors=anchors(turn.get('brief','')),effort=turn.get('effort','unknown'),censor_reason=turn.get('censor_reason','none'),outcome=turn.get('outcome','incomplete'))
    human=re.search(r'\bhuman[_ -]minutes\s*[:=]\s*(\d+(?:\.\d+)?)',turn.get('brief',''),re.I)
    row['features']['human_minutes']=float(human[1]) if human else 0
    row['features']['human_known']=int(bool(human))
    row['features']['verifiable']=int(bool(re.search(r'\b(test|tests|suite|compile|assert|lint|build)\b',turn.get('brief',''),re.I)))
    return row


def codex_turns(path):
    rows=[]; turn=None; meta={}; model=''; effort='unknown'
    for r in lr.json_lines(path):
        p=r.get('payload') or {}; kind=r.get('type'); event=p.get('type')
        if kind=='session_meta':
            meta=p
            if p.get('forked_from_id'):
                return []
        if kind=='turn_context':
            model=p.get('model',model); effort=p.get('effort',effort)
            if turn:
                turn.update(model=model,effort=effort)
        if kind=='event_msg' and event=='task_started':
            if turn:
                row=finish(turn,path,meta,'codex-turn')
                if row: rows.append(row)
            turn=dict(start=p.get('started_at') or r.get('timestamp'),id=p.get('turn_id'),brief='',model=model,effort=effort,outcome='incomplete',censor_reason='incomplete')
        if not turn:
            continue
        turn['last']=r.get('timestamp') or turn.get('last')
        if not turn['brief'] and ((kind=='event_msg' and event=='user_message') or (kind=='response_item' and p.get('role')=='user')):
            text=p.get('message') or lr.content_text(p.get('content'))
            if not text.lstrip().startswith(('# AGENTS.md','<environment_context>','<permissions instructions>')):
                turn['brief']=text
        if kind=='event_msg' and event in ('task_complete','turn_aborted'):
            if p.get('turn_id') and p['turn_id']!=turn['id']:
                continue
            reason='none' if event=='task_complete' and not p.get('error') else str(p.get('reason') or 'error')
            final=str(p.get('last_agent_message',''))[:1000]
            if re.search(r'^(?:you.ve hit|usage limit|rate limit|insufficient quota)',final.strip(),re.I):
                reason='usage-limit'
            turn.update(end=p.get('completed_at') or r['timestamp'],outcome='completed' if reason=='none' else 'censored',censor_reason=reason)
            row=finish(turn,path,meta,'codex-turn')
            if row and turn.get('model')!='codex-auto-review': rows.append(row)
            turn=None
    if turn:
        row=finish(turn,path,meta,'codex-turn')
        if row: rows.append(row)
    return rows


def claude_turns(path):
    rows=[]; turn=None; meta={'id':str(path)}
    for r in lr.json_lines(path):
        kind=r.get('type'); msg=r.get('message') or {}; content=msg.get('content'); stamp=r.get('timestamp')
        if not stamp: continue
        if r.get('cwd'): meta['cwd']=r['cwd']
        actual_user=kind=='user' and not r.get('isMeta') and not (isinstance(content,list) and any(b.get('type')=='tool_result' for b in content if isinstance(b,dict)))
        if actual_user:
            if turn:
                row=finish(turn,path,meta,'claude-turn')
                if row: rows.append(row)
            turn=dict(start=stamp,brief=lr.content_text(content),id=r.get('uuid'),outcome='incomplete',censor_reason='incomplete')
        if not turn: continue
        turn['last']=stamp
        if kind=='assistant':
            turn['model']=msg.get('model') or turn.get('model','')
            if msg.get('stop_reason')=='end_turn':
                turn.update(end=stamp,outcome='completed',censor_reason='none')
            if r.get('isApiErrorMessage') or msg.get('stop_reason') in ('max_tokens','refusal'):
                turn.update(end=stamp,outcome='censored',censor_reason='api-error')
        if kind=='system' and r.get('subtype') in ('api_error','turn_aborted'):
            turn.update(end=stamp,outcome='censored',censor_reason=r['subtype'])
    if turn:
        row=finish(turn,path,meta,'claude-turn')
        if row: rows.append(row)
    return rows


def match(pair,runs):
    dispatch=pair.get('brief_time',pair['start'])
    query=set(pair.get('brief_anchors',[]))
    candidates=[]
    for run in runs:
        if not -2<=run['start']-dispatch<=180:
            continue
        if run['end'] < pair['start'] or abs(run['start']-pair['start']) > 180:
            continue
        if pair.get('model','unknown')!='unknown' and run['model']!=pair['model']:
            continue
        overlap=len(query & set(run.get('brief_anchors',[])))
        same=pair.get('brief_hash') and pair['brief_hash']==run.get('brief_hash')
        if same or (len(query)>=3 and overlap>=3 and overlap/len(query)>=.65):
            candidates.append(run)
    if len(candidates)!=1:
        return dict(pair,actual_s=pair['end']-pair['start'],label_source='hold-fallback',match_candidates=len(candidates))
    run=candidates[0]
    return dict(pair,actual_s=run['end']-run['start'],label_source='transcript',run_id=run['run_id'],run_start=run['start'],run_end=run['end'],prediction_at=run['start'],expect_is_prerun=pair['start']<=run['start'],elapsed_at_issue=max(0,pair['start']-run['start']),censor_reason=run.get('censor_reason','none'),effort=run.get('effort','unknown'),features=dict(pair.get('features',{}),**{k:v for k,v in run.get('features',{}).items() if k in ('human_minutes','human_known','verifiable')}))
