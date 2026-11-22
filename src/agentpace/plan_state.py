"""Persistent deliverable state and issued-ETA/outcome audit log.

Plan mutations are serialized with a local file lock. ETA snapshots live next
to their plan; only derived forecast inputs belong in the run rows store.
"""
from __future__ import annotations
from agentpace import paths

import copy
import json
import fcntl
import math
from contextlib import contextmanager
from pathlib import Path
from time import time
from agentpace import local_runs as lr


def load(path):
    return json.loads(Path(path).read_text())


def save(path,plan):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp'); temporary.write_text(json.dumps(plan,indent=2)+'\n'); temporary.replace(path)


@contextmanager
def transaction(path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        plan=load(path)
        yield plan
        save(path,plan)


def create(path,plan,now=None):
    now=time() if now is None else lr.epoch(now)
    if Path(path).exists(): raise ValueError('plan already exists')
    plan=copy.deepcopy(plan); plan.setdefault('id',Path(path).stem); plan.setdefault('cwd',str(Path.cwd())); plan.setdefault('tasks',[])
    plan.update(created_at=now,events=[],forecasts=[],outcomes=[])
    for task in plan['tasks']: task.setdefault('status','todo')
    save(path,plan)
    return plan


def update(path,action,task_id,now=None,took=None):
    now=time() if now is None else lr.epoch(now)
    with transaction(path) as plan:
        task=next((t for t in plan['tasks'] if t['id']==task_id),None)
        if not task: raise ValueError('unknown plan task')
        if action=='start':
            if task.get('status','todo')!='todo': raise ValueError('only todo tasks can start')
            dependencies=[t for t in plan['tasks'] if t['id'] in task.get('depends_on',[])]
            if any(t.get('status')!='done' for t in dependencies): raise ValueError('task dependencies are not done')
            task.update(status='running',started_at=now)
            plan.setdefault('started_at',now)
        elif action=='done':
            if task.get('status')=='done': return copy.deepcopy(task)
            start=task.get('started_at',now-float(took) if took is not None else None)
            if start is None: raise ValueError('task needs start or --took')
            actual=now-start if took is None else float(took)
            if actual<0: raise ValueError('negative task duration')
            task.update(status='done',started_at=start,ended_at=now,actual_s=actual)
        else: raise ValueError('unknown task action')
        plan['events'].append(dict(action=action,task=task_id,at=now))
        result=copy.deepcopy(task)
    return result


def scope(path,*,added=None,removed=None,reason='',now=None):
    now=time() if now is None else lr.epoch(now)
    with transaction(path) as plan:
        if added:
            if any(t['id']==added['id'] for t in plan['tasks']): raise ValueError('duplicate task id')
            added=dict(added,status='todo'); plan['tasks'].append(added)
        if removed:
            task=next((t for t in plan['tasks'] if t['id']==removed),None)
            if not task: raise ValueError('unknown removed task')
            if task.get('status')=='done': raise ValueError('cannot remove completed work')
            task['status']='removed'; task['removed_at']=now
        plan['events'].append(dict(action='scope',at=now,added=added,removed=removed,reason=reason))
    return load(path)


NOTE_KINDS=('cut','parallel','note')
NOTE_LIMIT=2000


def note(path,task_id,*,kind,text,saves_s=None,now=None):
    """Attach an agent's note to an open task: what could be cut, what parallelism would give, or plain context.

    Notes are the agent's claims. `saves_s` is a claim in seconds of ETA; eta checks it against the
    scheduler and says whether it is backed. Stored in `agent_notes` in the order written; a refused
    call writes nothing.
    """
    now=time() if now is None else lr.epoch(now)
    text=text.strip()
    if kind not in NOTE_KINDS: raise ValueError('note kind must be one of '+', '.join(NOTE_KINDS))
    if not text: raise ValueError('note text must not be empty')
    if len(text)>NOTE_LIMIT: raise ValueError(f'note text is limited to {NOTE_LIMIT} characters')
    if saves_s is not None:
        if kind=='note': raise ValueError('--saves-s needs --kind cut or parallel')
        if not math.isfinite(saves_s) or saves_s<0: raise ValueError('--saves-s must be a finite number of seconds, zero or more')
    with transaction(path) as plan:
        task=next((t for t in plan['tasks'] if t['id']==task_id),None)
        if not task: raise ValueError('unknown plan task')
        if task.get('status','todo') in ('done','removed'): raise ValueError(f"notes attach to an open task; {task_id} is {task['status']}")
        entry=dict(kind=kind,text=text,saves_s=saves_s,at=now)
        task.setdefault('agent_notes',[]).append(entry)
        if 'events' in plan: plan['events'].append(dict(action='note',task=task_id,at=now))
    return copy.deepcopy(entry)


def sync_ledger(path,ledger):
    if not Path(ledger).exists(): return dict(closed=0)
    closed=0
    with transaction(path) as plan:
        for record in lr.json_lines(ledger):
            if record.get('v')!=1 or record.get('outcome') not in ('done','completed','success','exited'): continue
            linked=(record.get('meta') or {}).get('task','')
            prefix,separator,task_id=linked.rpartition('/')
            if not separator or prefix not in (plan['id'],str(Path(path).resolve()),str(path)): continue
            task=next((t for t in plan['tasks'] if t['id']==task_id),None)
            if not task or task.get('status') in ('done','removed'): continue
            try: start,end=lr.epoch(record['started_at']),lr.epoch(record['ended_at'])
            except (KeyError,ValueError,TypeError,AttributeError): continue
            if end<start: continue
            task.update(status='done',started_at=start,ended_at=end,actual_s=end-start,actual_source='vigil-ledger')
            plan.setdefault('started_at',start)
            plan['events'].append(dict(action='ledger-done',task=task_id,at=end,ledger_id=lr.identity((record.get('hold'),record.get('label'),start))))
            closed+=1
    return dict(closed=closed)


def issue(path,result,now=None,owner_grant=None,snapshot=None):
    now=time() if now is None else lr.epoch(now)
    with transaction(path) as plan:
        event=dict(id=lr.identity((now,len(plan['forecasts']))),issued_at=now,forecast={k:result[k] for k in ('commit','p50','p80')},snapshot=copy.deepcopy({k:v for k,v in (snapshot if snapshot is not None else plan).items() if k not in ('forecasts','outcomes')}))
        event['forecast'].update({k:copy.deepcopy(result[k]) for k in ('reliable','unbounded_tasks','commit_is_actionable') if k in result})
        if owner_grant:
            event['owner_grant']=copy.deepcopy(owner_grant) if isinstance(owner_grant,dict) else dict(quote=owner_grant,at=now)
        for summary in result.get('tasks',[]):
            task=next((t for t in plan['tasks'] if t['id']==summary['id']),None)
            if task and task.get('status','todo')=='todo': task.setdefault('forecast_s',summary['p50'])
        plan['forecasts'].append(event)
        plan['events'].append(dict(action='eta',at=now,id=event['id']))
    return event


def deliver(path,now=None):
    now=time() if now is None else lr.epoch(now)
    with transaction(path) as plan:
        if plan.get('delivered_at'):
            return plan['outcomes']
        unfinished=[t['id'] for t in plan['tasks'] if t.get('status','todo') not in ('done','removed')]
        if unfinished: raise ValueError('close or remove unfinished tasks before deliver: '+', '.join(unfinished))
        outcomes=[]
        for event in plan['forecasts']:
            actual=now-event['issued_at']
            if actual<0: raise ValueError('delivery predates issued ETA')
            p=event['forecast']
            outcomes.append(dict(forecast_id=event['id'],actual_remaining_s=actual,within_commit=actual<=p['commit'],within_p50=actual<=p['p50'],within_p80=actual<=p['p80'],pinball_p50=lr.pinball(actual,p['p50'],.5),pinball_p80=lr.pinball(actual,p['p80'],.8)))
        plan.update(delivered_at=now,outcomes=outcomes)
        plan['events'].append(dict(action='deliver',at=now))
    return outcomes


def cmd_plan(args):
    path=Path(args.plan).resolve(); now=lr.epoch(args.at) if args.at else time()
    action=args.plan_action
    if action=='init': result=create(path,load(args.source) if args.source else dict(id=path.stem,cwd=str(Path(args.cwd).resolve()),tasks=[]),now)
    elif action=='add':
        task=json.loads(args.task_json) if args.task_json else dict(id=args.task,model=args.model,kind=args.kind,type=args.type,depends_on=args.depends_on)
        if args.seconds is not None: task.update(seconds=args.seconds,basis=args.basis)
        result=scope(path,added=task,reason=args.reason,now=now)
    elif action in ('start','done'): result=update(path,action,args.task,now,took=args.took)
    elif action=='note': result=note(path,args.task,kind=args.kind,text=args.text,saves_s=args.saves_s,now=now)
    elif action=='scope': result=scope(path,added=json.loads(args.add_json) if args.add_json else None,removed=args.remove,reason=args.reason,now=now)
    elif action=='deliver': result=deliver(path,now)
    elif action=='sync': result=sync_ledger(path,args.ledger)
    else: result=load(path)
    print(json.dumps(result,indent=2) if args.json else f'Plan {path.stem}: {action} recorded. Use agentpace eta {path} --json for the current remainder.')
    return 0


def add_commands(commands):
    root=commands.add_parser('plan',help='agent: update the persistent deliverable plan as work proceeds')
    sub=root.add_subparsers(dest='plan_action',required=True)
    for action in ('init','add','start','done','note','scope','deliver','sync','show'):
        parser=sub.add_parser(action); parser.add_argument('plan'); parser.add_argument('--json',action='store_true'); parser.add_argument('--at')
        if action=='init': parser.add_argument('--from',dest='source'); parser.add_argument('--cwd',default='.')
        if action in ('add','start','done'): parser.add_argument('task'); parser.add_argument('--took',type=float)
        if action=='note': parser.add_argument('task'); parser.add_argument('--kind',choices=NOTE_KINDS,required=True); parser.add_argument('--text',required=True); parser.add_argument('--saves-s',type=float,help='claimed seconds saved on the ETA; eta checks it against the scheduler')
        if action=='add':
            parser.add_argument('--task-json'); parser.add_argument('--model',default='sol'); parser.add_argument('--kind',default='implementation'); parser.add_argument('--type',default='agent',choices=('agent','machine','wait')); parser.add_argument('--seconds',type=float); parser.add_argument('--basis',default=''); parser.add_argument('--depends-on',action='append',default=[]); parser.add_argument('--reason',default='')
        if action=='scope': parser.add_argument('--add-json'); parser.add_argument('--remove'); parser.add_argument('--reason',required=True)
        if action=='sync': parser.add_argument('--ledger',default=str(paths.vigil_ledger()))
        parser.set_defaults(handler=cmd_plan)
