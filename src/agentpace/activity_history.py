"""Read local execution evidence without writing T3 or storing transcript prose."""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from agentpace import local_runs as lr,pause_clock as pc,paths as locations

LIMIT=re.compile(r"you(?:'|’)ve hit your (?:usage )?limit|usage[_ -]limit(?: has been)?[_ -](?:reached|exceeded)|usage[_ -]window.*(?:exhausted|reset)|quota.*(?:exceeded|exhausted)",re.I)
MACHINE=re.compile(r'flutter\s+test|cargo\s+(?:test|build)|nix\s+build|gradlew|python\S*\s+-m\s+unittest|npm\s+(?:test|run\s+build)|make\s+(?:test|build)',re.I)


def metadata(path):
    connection=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True,timeout=2)
    try:
        connection.execute('PRAGMA query_only=ON')
        threads={r[0]:json.loads(r[1]) for r in connection.execute('SELECT provider_thread_id,payload_json FROM orchestration_v2_projection_provider_threads')}
        runs={r[0]:json.loads(r[1]) for r in connection.execute('SELECT run_id,payload_json FROM orchestration_v2_projection_runs')}
        attempts=[json.loads(r[0]) for r in connection.execute('SELECT payload_json FROM orchestration_v2_projection_run_attempts')]
        turns=[dict(id=r[0],attempt=r[1],node=r[2],start=r[3],end=r[4]) for r in connection.execute('SELECT provider_turn_id,run_attempt_id,node_id,started_at,completed_at FROM orchestration_v2_projection_provider_turns')]
        requests=[dict(turn=r[0],node=r[1],kind=r[2],start=r[3],end=r[4]) for r in connection.execute('SELECT provider_turn_id,node_id,kind,created_at,resolved_at FROM orchestration_v2_projection_runtime_requests')]
    finally: connection.close()
    by_attempt={a['id']:a.get('runId') for a in attempts};by_turn={};by_node={p.get('rootNodeId'):run for run,p in runs.items()}
    result={}
    for run,p in runs.items():
        provider=threads.get(p.get('providerThreadId'),{})
        native=(provider.get('nativeThreadRef') or {}).get('nativeId')
        result[run]=dict(native=native,provider_spans=[],waits=[],thread_id=p.get('threadId'))
    for attempt in attempts:
        run=attempt.get('runId')
        if run in result and attempt.get('nativeThreadId'): result[run]['native']=attempt['nativeThreadId']
    for turn in turns:
        run=by_attempt.get(turn['attempt']) or by_node.get(turn['node'])
        if run not in result: continue
        by_turn[turn['id']]=run
        if turn['start'] is not None and turn['end'] is not None:
            result[run]['provider_spans'].append((lr.epoch(turn['start']),lr.epoch(turn['end'])))
    for request in requests:
        run=by_turn.get(request['turn']) or by_node.get(request['node'])
        if run not in result or not request['start']: continue
        finish=request['end'] or runs[run].get('completedAt')
        if not finish: continue
        result[run]['waits'].append(dict(start=lr.epoch(request['start']),end=lr.epoch(finish),kind='question' if request['kind']=='user_input' else request['kind']))
    return result


def paths(codex_root=None,claude_root=None):
    native={};sessions={}
    for path in (Path(codex_root) if codex_root else locations.codex_sessions()).rglob('rollout-*.jsonl'):
        sid=path.stem[-36:];native[sid]=path;sessions[lr.identity(sid)]=path
    for path in (Path(claude_root) if claude_root else locations.claude_projects()).rglob('*.jsonl'):
        native.setdefault(path.stem,path);sessions[lr.identity(str(path))]=path
    return native,sessions


def text_of(record):
    body=record.get('payload') or record.get('message') or {}
    if isinstance(body,str): return body
    value=body.get('message') or body.get('content') or record.get('content') or body.get('error') or record.get('error') or ''
    if isinstance(value,dict): return str(value.get('message') or value.get('detail') or '')
    return lr.content_text(value)


def parse_log(path):
    events=[];progress=[];markers=[];tools={};tool_spans=[];work=[];waits=[];budgets=[];counts={'reasoning':0,'tool_calls':0,'limit_text_mentions':0};stats=[]
    for record in lr.json_lines(path):
        if record.get('timestamp') is None: continue
        try: at=lr.epoch(record['timestamp'])
        except (ValueError,TypeError,AttributeError): continue
        events.append(at)
        typ=record.get('type');body=record.get('payload') or record.get('message') or {};body=body if isinstance(body,dict) else {}
        subtype=body.get('type');content=body.get('content');text=text_of(record)
        if 'usage limit' in text.casefold() or 'hit your limit' in text.casefold(): counts['limit_text_mentions']+=1;stats.append((at,'limit_text_mentions'))
        is_model=False
        error=bool(record.get('isApiErrorMessage')) or body.get('model')=='<synthetic>' or subtype in ('error','warning','rate_limit') or typ in ('error','system')
        standard=bool(LIMIT.match(text.strip()))
        limit=bool(LIMIT.search(text)) and (error or standard) and ('usage' in text.casefold() or 'reset' in text.casefold() or 'quota' in text.casefold())
        # User briefs/tool output can quote errors; only platform/model-error channels qualify.
        if typ=='response_item' and subtype in ('custom_tool_call_output','function_call_output'):
            limit=False
        if typ=='user' or subtype=='user_message' or body.get('role') in ('user','developer','system'): limit=False
        if limit: markers.append(dict(at=at,kind='usage-limit',source='platform-limit-message'))
        if typ=='response_item' and subtype in ('reasoning','function_call','custom_tool_call','message') and body.get('role') not in ('user','developer','system'):
            is_model=True
        if typ=='event_msg' and subtype in ('agent_message','agent_reasoning'): is_model=True
        if typ=='assistant' and not error: is_model=True
        if subtype=='reasoning' or (typ=='assistant' and isinstance(content,list) and any(b.get('type')=='thinking' for b in content if isinstance(b,dict))): counts['reasoning']+=1;stats.append((at,'reasoning'))
        if is_model and not limit: progress.append(at)
        if typ=='response_item' and subtype in ('function_call','custom_tool_call','local_shell_call'):
            tool_id=body.get('call_id') or body.get('id');name=body.get('name','');arguments=str(body.get('arguments') or body.get('input') or '')
            tools[tool_id]=(at,bool(MACHINE.search(arguments)),name.endswith('sleep') or name in ('request_user_input','t3_pending_request'));counts['tool_calls']+=1;stats.append((at,'tool_calls'))
        if typ=='response_item' and subtype in ('custom_tool_call_output','function_call_output','local_shell_call_output'):
            tool_id=body.get('call_id')
            if tool_id in tools:
                begin,machine,waiting=tools.pop(tool_id);tool_spans.append((begin,at))
                if machine: work.append((begin,at))
                if waiting: waits.append(dict(start=begin,end=at,kind='explicit-tool-wait'))
        if typ=='assistant' and isinstance(content,list):
            for block in content:
                if block.get('type')=='tool_use':
                    name=block.get('name','');arguments=str(block.get('input') or '')
                    tools[block['id']]=(at,bool(MACHINE.search(arguments)),any(s in name for s in ('AskUserQuestion','request_user_input','t3_pending_request')));counts['tool_calls']+=1;stats.append((at,'tool_calls'))
        if typ=='user' and isinstance(content,list):
            for block in content:
                if block.get('type')=='tool_result' and block.get('tool_use_id') in tools:
                    begin,machine,waiting=tools.pop(block['tool_use_id']);tool_spans.append((begin,at))
                    if machine: work.append((begin,at))
                    if waiting: waits.append(dict(start=begin,end=at,kind='question'))
        # Only actual system/error advisories, never task prose, supply budget snapshots.
        if error:
            match=re.search(r'(?:remaining|left)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%',text,re.I)
            if match and ('budget' in text.casefold() or 'window' in text.casefold()): budgets.append((at,float(match[1])))
    return dict(events=sorted(set(events)),progress=sorted(set(progress)),markers=markers,tool_spans=tool_spans,work_spans=work,waits=waits,budgets=budgets,counts=counts,stats=stats)


def usage_waits(trace,start,end):
    result=[]
    for signal in trace['markers']:
        if not start<=signal['at']<end: continue
        resumed=next((t for t in trace['progress'] if signal['at']<t<=end),end)
        if resumed>signal['at']:
            # Do not subtract independently observed machine work during the reset wait.
            pieces=pc.subtract([(signal['at'],resumed)],pc.union(trace['work_spans'],start,end))
            result.extend(dict(start=a,end=b,kind='usage-limit') for a,b in pieces if b-a>1)
    return [dict(start=a,end=b,kind='usage-limit') for a,b in pc.union([(r['start'],r['end']) for r in result],start,end)]


def describe(row,trace,meta=None,idle_cap=pc.IDLE_CAP):
    start=row.get('run_start',row['start']);end=row.get('run_end',row['end']);meta=meta or {}
    if end is None: return None
    trace=trace or dict(events=[],progress=[],markers=[],tool_spans=[],work_spans=[],waits=[],budgets=[],counts={})
    usage=usage_waits(trace,start,end)
    waits=list(meta.get('waits',[]))+trace['waits']+usage
    result=pc.decompose(start,end,trace['events'],waits=waits,signals=trace['markers'],provider_spans=meta.get('provider_spans',[]),work_spans=trace['work_spans'],tool_spans=trace['tool_spans'],idle_cap=idle_cap)
    usage_s=sum(w['end']-w['start'] for w in usage)
    budget=[(at,value) for at,value in trace['budgets'] if 0<=start-at<=60]
    boundaries=sorted(set([start,end]+[t for t in trace['events'] if start<=t<=end]))
    maximum_gap=max((b-a for a,b in zip(boundaries,boundaries[1:])),default=None)
    result.update(usage_wait_s=usage_s,trimmed_s=end-start-usage_s,usage_waits=usage,usage_limit_markers=sum(start<=s['at']<=end for s in trace['markers']),maximum_event_gap_s=maximum_gap,budget_remaining_pct=budget[-1][1] if budget else None,trace_counts={kind:sum(start<=at<=end and tag==kind for at,tag in trace.get('stats',[])) for kind in ('reasoning','tool_calls','limit_text_mentions')})
    return result
