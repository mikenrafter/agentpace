"""Read-only T3 run labels and exact transcript tool-result identities.

Connections are closed before feature extraction. Raw messages are read only
in memory; the derived rows writer never receives them.
"""
from __future__ import annotations

import heapq
import json
import re
from pathlib import Path
from agentpace import local_runs as lr, paths


# Runs that finished at or after `since`, plus every run that may overlap one of them (still
# open, or finished after the earliest request among them), so `active_runs` stays exact.
_SINCE_FILTER = """
    WHERE r.completed_at IS NULL OR julianday(r.completed_at) >= (
        SELECT min(julianday(requested_at)) FROM orchestration_v2_projection_runs
        WHERE completed_at IS NOT NULL AND julianday(completed_at) >= julianday(?, 'unixepoch'))
"""


def read_database(path, since=None, stats=None):
    """Derived T3 run rows. With `since` (epoch), only runs that finished at or after it and the
    runs overlapping them are read, so an incremental caller never parses old history.

    A row whose JSON does not parse is skipped and counted in `stats['skipped_invalid']`.
    """
    import sqlite3
    connection=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True,timeout=2)
    try:
        connection.execute('PRAGMA query_only=ON')
        connection.row_factory=sqlite3.Row
        records=[dict(row) for row in connection.execute('''
            SELECT r.run_id,r.thread_id,r.ordinal,r.provider,r.status,
                   r.requested_at,r.completed_at,r.payload_json,
                   m.payload_json AS message_json,p.workspace_root
            FROM orchestration_v2_projection_runs r
            LEFT JOIN orchestration_v2_projection_messages m
              ON m.message_id=json_extract(CASE WHEN json_valid(r.payload_json) THEN r.payload_json END,'$.userMessageId')
            LEFT JOIN orchestration_v2_projection_threads t ON t.thread_id=r.thread_id
            LEFT JOIN projection_projects p ON p.project_id=t.project_id
        '''+(_SINCE_FILTER if since is not None else ''),(since,) if since is not None else ())]
    finally:
        connection.close()
    rows=[]
    from agentpace.quantiles import WORDS
    for record in records:
        try:
            payload=json.loads(record.pop('payload_json'))
            message=json.loads(record.pop('message_json') or '{}')
            if not isinstance(payload,dict) or not isinstance(message,dict): raise ValueError('not an object')
            brief=str(message.get('text') or '')
            selection=payload.get('modelSelection') or {}
            start=payload.get('startedAt') or record['requested_at']
            end=payload.get('completedAt') or record['completed_at']
            if not start: continue
            started=lr.epoch(start)
            completed=lr.epoch(end) if end else None
            if completed is not None and completed<=started: continue
            # Incomplete runs remain in the concurrency index, not completion training.
            row=lr.make_row(start=started,end=completed or started+1,model=selection.get('model',''),brief=brief,project=record.get('workspace_root') or '',session=record['thread_id'],source='t3-database')
            row.update(run_id=record['run_id'],thread_id=record['thread_id'],ordinal=record['ordinal'],provider=record['provider'],execution_started=payload.get('startedAt') is not None,status=record['status'],requested_at=lr.epoch(record['requested_at']),prediction_at=started,label_source='t3-database',censor_reason='none' if record['status']=='completed' else record['status'],expect_s=None,expect_is_prerun=False,effort='unknown',delegated=record['thread_id'].startswith('thread:delegated-task:'),delegated_task_id=payload.get('delegatedTaskId') or None,model_raw=str(selection.get('model') or ''))
        except (ValueError,TypeError,KeyError,AttributeError):
            if stats is not None: stats['skipped_invalid']=stats.get('skipped_invalid',0)+1
            continue
        for option in selection.get('options') or []:
            if isinstance(option,dict) and option.get('id')=='reasoningEffort': row['effort']=option.get('value','unknown')
        if completed is None: row.update(end=None,actual_s=None)
        words=re.findall(r'\w+',brief.casefold())
        row['hints']={word:words.count(word) for word in WORDS}
        human=re.search(r'\bhuman[_ -]minutes\s*[:=]\s*(\d+(?:\.\d+)?)',brief,re.I)
        row['features']['human_minutes']=float(human[1]) if human else 0
        row['features']['human_known']=int(bool(human))
        row['features']['queue_wait_s']=max(0,started-row['requested_at'])
        row['features']['verifiable']=int(any(word in brief.casefold() for word in ('test','compile','assert','lint','build')))
        rows.append(row)
    active=concurrency(rows)
    for row in rows: row['features']['active_runs']=active[row['run_id']]
    return rows


def concurrency(runs):
    active=[]; result={}
    grouped={}
    for row in runs: grouped.setdefault(row['start'],[]).append(row)
    for started,group in sorted(grouped.items()):
        while active and active[0][0]<=started: heapq.heappop(active)
        executing=[row for row in group if row.get('execution_started',True)]
        for row in group: result[row['run_id']]=len(active)+len(executing)-int(row.get('execution_started',True))
        for row in executing: heapq.heappush(active,(row.get('end') or float('inf'),row['run_id']))
    return result


def result_ids(content):
    """Read structured tool result JSON; never scrape arbitrary transcript prose."""
    if isinstance(content,str):
        try: content=json.loads(content)
        except ValueError: return []
    if isinstance(content,list):
        return [item for block in content if isinstance(block,dict) for item in result_ids(block.get('text') or block)]
    if not isinstance(content,dict): return []
    if content.get('runId') or content.get('childRunId'):
        return [dict(runId=content.get('runId') or content['childRunId'],childThreadId=content.get('childThreadId') or content.get('threadId'),taskId=content.get('taskId'))]
    return [item for key,value in content.items() if key in ('content','structuredContent','result','task','run') for item in result_ids(value)]


def agent_ids(record,block):
    """The subagent id a background Agent call's result names (structured result first)."""
    result=record.get('toolUseResult')
    if isinstance(result,dict) and result.get('agentId'):
        return [dict(agentId=str(result['agentId']))]
    found=re.findall(r'\bagentId:\s*([\w-]+)',lr.content_text(block.get('content')))
    return [dict(agentId=found[0])] if len(set(found))==1 else []


def call_index(paths):
    """Dispatch call key -> the identities its result names: a T3 run (delegate_task,
    t3_thread_send) or a background subagent (Agent)."""
    index={}
    for path in paths:
        calls={}
        for record in lr.json_lines(path):
            if not record.get('timestamp'): continue
            blocks=(record.get('message') or {}).get('content')
            if not isinstance(blocks,list): continue
            for block in blocks:
                if not isinstance(block,dict): continue
                if block.get('type')=='tool_use' and (any(n in block.get('name','') for n in ('delegate_task','t3_thread_send')) or block.get('name')=='Agent'):
                    calls[block['id']]=((str(path),lr.epoch(record['timestamp']),block['id']),block.get('name')=='Agent')
                elif block.get('type')=='tool_result' and block.get('tool_use_id') in calls:
                    key,agent=calls[block['tool_use_id']]
                    ids=agent_ids(record,block) if agent else result_ids(block.get('content'))
                    if len(ids)==1: index.setdefault(key,[]).append(ids[0])
    return {key:list({item.get("runId") or item.get("agentId"):item for item in values}.values()) for key,values in index.items()}


def relabel(pair,identifiers,runs):
    lookup={r['run_id']:r for r in runs}
    row=lookup.get(identifiers.get('runId'))
    if not row or row.get('end') is None:
        return dict(pair,label_source=pair.get('label_source','hold-fallback'))
    result=dict(pair,actual_s=row['actual_s'],run_start=row['start'],run_end=row['end'],run_id=row['run_id'],t3_thread_id=row['thread_id'],prediction_at=row['start'],label_source='t3-run-id',censor_reason='none' if row.get('status')=='completed' else row.get('status','unknown'),expect_is_prerun=pair['start']<=row['start'])
    if 'features' in row: result['features']=dict(row['features'])
    for field in ('hints','model','kind','project','provider'):
        if field in row: result[field]=row[field]
    result.update(effort=row.get('effort','unknown'),delegated=row.get('delegated',True),ordinal=row.get('ordinal',1))
    return result


def _is_vigil(token):
    # `H=$(vigil hold ...)` and `(vigil ...)` name the same command as a bare `vigil`.
    base=token.rsplit('$(',1)[-1].lstrip('(`')
    return base=='vigil' or base.endswith('/vigil')


def hold_index(paths,agents=False):
    """Bind hold registrations to the exact preceding dispatch tool-use id.

    With `agents`, a background Agent call is a dispatch too (learn links holds to subagent
    transcripts); by default an Agent call clears the binding, as the round-3 replay expects."""
    import shlex
    links={}
    for path in paths:
        last_call=None
        for record in lr.json_lines(path):
            stamp=record.get('timestamp')
            blocks=(record.get('message') or {}).get('content')
            if stamp is None or not isinstance(blocks,list): continue
            for block in blocks:
                if not isinstance(block,dict) or block.get('type')!='tool_use': continue
                name=block.get('name','');inp=block.get('input') or {}
                if any(part in name for part in ('delegate_task','t3_thread_send')):
                    last_call=(str(path),lr.epoch(stamp),block['id'])
                elif name=='Agent':
                    last_call=(str(path),lr.epoch(stamp),block['id']) if agents else None
                elif name=='Bash' and last_call:
                    try: tokens=shlex.split(inp.get('command',''))
                    except ValueError: continue
                    for i,token in enumerate(tokens):
                        if _is_vigil(token) and tokens[i+1:i+2]==['hold']:
                            rest=tokens[i+2:]
                            if rest[:1]==['done']: continue
                            if '--label' in rest and '--expect-s' in rest:
                                label=rest[rest.index('--label')+1]
                                links[(str(path),lr.epoch(stamp),label)]=last_call
                elif name.startswith('mcp__vigil__hold') and inp.get('label') and inp.get('expect_s') and last_call:
                    links[(str(path),lr.epoch(stamp),inp['label'])]=last_call
    return links


def cohort_key(record):
    session=lr.identity(record['file']) if 'file' in record else record['session']
    return session,record['start'],record['end'],record.get('expect_s')


def resolve_links(rows,previous):
    groups={}
    for row in rows:
        if row.get('label_source')=='t3-run-id': groups.setdefault(row['run_id'],set()).add(row.get('hold_identity'))
    return [dict(previous[i],t3_match='duplicate-run-reference') if row.get('label_source')=='t3-run-id' and len(groups[row['run_id']])>1 else row for i,row in enumerate(rows)]


def active_at(when,path=None):
    """A bounded live count; no prompts or completed labels are fetched."""
    import sqlite3
    path=Path(path) if path else paths.t3_database()
    if not path.exists(): return None
    connection=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=2)
    try:
        return connection.execute('''SELECT count(*) FROM orchestration_v2_projection_runs
            WHERE julianday(json_extract(payload_json,'$.startedAt'))<=julianday(?,'unixepoch')
              AND (json_extract(payload_json,'$.completedAt') IS NULL
                   OR julianday(json_extract(payload_json,'$.completedAt'))>julianday(?,'unixepoch'))''',(when,when)).fetchone()[0]
    except sqlite3.Error:
        return None
    finally: connection.close()


def active_now(path=None):
    from time import time
    return active_at(time(),path)
