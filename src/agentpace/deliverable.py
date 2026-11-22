"""Local deliverable ETA: dependency clock, explicit shortcuts, layered cuts.

Commit is the makespan of individual task cuts, not a claimed plan quantile.
P50/P80 come from simulated whole plans with a shared residual rank, preserving
positive dependence between runs. This deliberately avoids independent-task
averaging when session correlation is poorly measured in this pilot.
"""
from __future__ import annotations
from agentpace import paths

import copy
import json
import math
import random
from pathlib import Path
from agentpace import local_runs as lr

CUTS = {'conservative': .5, 'balanced': .4, 'aggressive': .3}


def appetite(config, plan, task):
    """Resolve policy in increasing precedence; return name, cut and source."""
    policy = {'risk_appetite': config.get('risk_appetite', 'balanced')}
    if 'commit_quantile' in config:
        policy['commit_quantile'] = config['commit_quantile']
    source = 'user default'
    inferred = str(plan.get('deliverable', '')) + ' ' + str(task.get('kind', ''))
    if any(word in inferred for word in ('migration', 'release', 'upstream')):
        inferred = 'conservative'
    elif any(word in inferred for word in ('pilot', 'research')):
        inferred = 'aggressive'
    else:
        inferred = None
    layers = []
    if inferred:
        layers.append(('inferred type/kind', {'risk_appetite': inferred}))
    project = str(Path(plan.get('cwd', '.')).resolve())
    layers.extend([
        ('project', config.get('projects', {}).get(project, {})),
        ('deliverable', config.get('deliverables', {}).get(plan.get('deliverable', ''), {})),
        ('plan', plan), ('task', task),
    ])
    for label, layer in layers:
        if 'risk_appetite' in layer:
            policy = {'risk_appetite': layer['risk_appetite']}
            source = label
        if 'commit_quantile' in layer:
            policy['commit_quantile'] = layer['commit_quantile']
            source = label
    name = policy['risk_appetite']
    if name not in CUTS:
        raise ValueError('risk_appetite must be conservative, balanced or aggressive')
    cut = float(policy.get('commit_quantile', CUTS[name]))
    if not math.isfinite(cut) or not 0 < cut <= .5:
        raise ValueError('commit_quantile must be above zero and at most 0.5')
    return name, cut, source


def ordered(tasks):
    ids = [task['id'] for task in tasks]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError('a plan needs unique task ids and at least one task')
    waiting = list(tasks)
    result = []
    done = set()
    while waiting:
        ready = [t for t in waiting if set(t.get('depends_on', [])) <= done]
        if not ready:
            raise ValueError('plan dependencies contain a cycle or unknown task')
        for task in ready:
            result.append(task)
            done.add(task['id'])
            waiting.remove(task)
    return result


def makespan(tasks, durations):
    ends = {}
    for task in ordered(tasks):
        ends[task['id']] = max((ends[d] for d in task.get('depends_on', [])), default=0) + durations[task['id']]
    return max(ends.values())


def schedule(tasks, durations, agent_cap=1):
    """List-schedule agent tasks under the configured cap; machine steps are explicit."""
    ends = {}
    slots = [0.0] * agent_cap
    for task in ordered(tasks):
        ready = max((ends[d] for d in task.get('depends_on', [])), default=0)
        if task.get('type', 'agent') == 'agent' and durations[task['id']] > 0:
            slot = min(range(agent_cap), key=lambda i: slots[i])
            start = max(ready, slots[slot])
            ends[task['id']] = start + durations[task['id']]
            slots[slot] = ends[task['id']]
        else:
            ends[task['id']] = ready + durations[task['id']]
    return max(ends.values())


def apply_tradeoffs(plan, keep, no_defer, allow_spend=(), authorized=()):
    tasks = copy.deepcopy(plan['tasks'])
    applied, available = [], []
    catalogue = {entry['id']: entry for entry in plan.get('tradeoffs', [])}
    for entry in catalogue.values():
        mechanism = entry.get('mechanism', '')
        if mechanism in ('max-reasoning', 'max-effort', 'reasoning-effort'):
            raise ValueError('reasoning effort is not a speed accelerator')
        cost = entry.get('cost', {'class': 'none'})
        if cost.get('class') not in ('none', 'low', 'high'):
            raise ValueError('trade-off cost class must be none, low or high')
        if mechanism in ('fast-mode', 'ultra-fast', 'extra-agents', 'large-sandbox', 'extra-sandbox', 'idle-sandbox') and cost['class'] != 'high':
            raise ValueError('budget-heavy accelerator must declare high cost')
        if cost['class'] == 'high' and not (entry['id'] in allow_spend and entry['id'] in authorized):
            available.append(dict(entry, needs_owner=True, available_with='--allow-spend '+entry['id']))
    for name in plan.get('apply', []):
        if name not in catalogue:
            raise ValueError('unknown trade-off: ' + name)
        entry = catalogue[name]
        if any(e['id'] == name for e in available):
            continue
        if set(entry.get('protects', [])) & set(keep) or entry.get('task') in no_defer:
            continue
        if not entry.get('risk') or not entry.get('basis'):
            raise ValueError('trade-offs need a declared basis and added risk')
        if entry.get('mechanism') == 'extra-agents':
            applied.append(dict(entry, savings_s=0))
            continue
        target = next((task for task in tasks if task['id'] == entry.get('task')), None)
        if target and target.get('status','todo') in ('done','removed','running'):
            continue
        if not target or target.get('type') not in ('machine', 'wait'):
            raise ValueError('duration shortcuts replace measured/stated machine or wait tasks only')
        before, after = float(target['seconds']), float(entry['seconds'])
        if not math.isfinite(before + after) or not 0 <= after <= before:
            raise ValueError('trade-off duration must be finite and between zero and original')
        saved = before - after
        if abs(saved - float(entry['savings_s'])) > .01:
            raise ValueError('declared savings do not match task durations')
        target.update(seconds=after, basis=entry['basis'])
        applied.append(dict(entry, savings_s=saved, cost=entry.get('cost', {'class': 'none'})))
    return tasks, applied, available


def simulate(plan, rows, config, *, draws=4000, keep=(), no_defer=(), timings=(), now=None, allow_spend=(), owner_grant=None):
    from time import time
    from agentpace import run_forecast as rf, machine_timings as mt
    from agentpace import research
    rows=research.deduplicate(rows)
    if draws < 100:
        raise ValueError('draws must be at least 100')
    if not isinstance(plan, dict) or not isinstance(plan.get('tasks'), list):
        raise ValueError('plan needs a tasks list')
    for task in plan['tasks']:
        if not isinstance(task, dict) or not isinstance(task.get('id'), str):
            raise ValueError('each task needs a string id')
        if task.get('type', 'agent') == 'agent' and not all(isinstance(task.get(k), str) for k in ('model', 'kind')):
            raise ValueError('agent tasks need model and kind')
    now = time() if now is None else lr.epoch(now)
    trusted_spend = set(config.get('allow_spend', []))
    for layer in (config.get('projects', {}).get(str(Path(plan.get('cwd','.')).resolve()), {}), config.get('deliverables', {}).get(plan.get('deliverable',''), {})):
        if 'allow_spend' in layer: trusted_spend = set(layer['allow_spend'])
    authorized = trusted_spend | (set(allow_spend) if owner_grant and owner_grant.strip() else set())
    tasks, tradeoffs, available = apply_tradeoffs(plan, keep, no_defer, allow_spend, authorized)
    ordered(tasks)
    cap = int(config.get('agent_cap', 1))
    if cap < 1: raise ValueError('configured agent_cap must be positive')
    requested = int(plan.get('agent_cap', cap))
    if requested > cap:
        raise ValueError('raising agent_cap requires an authorized extra-agents catalogue entry')
    cap = requested
    for entry in tradeoffs:
        if entry.get('mechanism') == 'extra-agents':
            cap = int(entry['agent_cap'])
    if cap < 1: raise ValueError('agent cap must be positive')
    rng = random.Random(20261120)
    ranks = [rng.random() for _ in range(draws)]
    samples, cuts, summaries = {}, {}, []
    actual_work = 0
    history = list(rows)
    plan_session = lr.identity(plan.get('id', 'unrecorded-plan'))
    for task in tasks:
        if task.get('status') == 'done' and task.get('started_at') is not None:
            actual = float(task.get('actual_s', lr.epoch(task['ended_at'])-lr.epoch(task['started_at'])))
            history.append(lr.make_row(start=task['started_at'], end=lr.epoch(task['started_at'])+max(actual,.001), model=task.get('model',''), kind=task.get('kind','other'), project=plan.get('cwd','.'), session=plan.get('id','unrecorded-plan'), expect_s=task.get('forecast_s')))
    for task in tasks:
        name, cut, source = appetite(config, plan, task)
        base_plan = {k:v for k,v in plan.items() if k not in ('risk_appetite','commit_quantile')}
        base_task = {k:v for k,v in task.items() if k not in ('risk_appetite','commit_quantile')}
        base_name, base_cut, _ = appetite(config, base_plan, base_task)
        needs_owner = cut < base_cut and not (owner_grant and owner_grant.strip())
        if needs_owner:
            name, cut, source = base_name, base_cut, 'configured policy; requested increase needs owner'
        status = task.get('status', 'todo')
        elapsed = max(0, now-lr.epoch(task['started_at'])) if status == 'running' else 0
        typ = task.get('type', 'agent')
        support = None; extrapolated = False; task_context = None; support_source = 'empirical-reference'; separate_risks = None
        if status in ('done','removed'):
            actual_work += float(task.get('actual_s',0)) if status == 'done' else 0
            pairs = [(0.0,1)]
        elif typ in ('machine','wait'):
            if 'seconds' in task:
                if not task.get('basis'): raise ValueError('explicit machine/wait seconds need a measured or stated basis')
                values = [float(task['seconds'])]
            else:
                label = task.get('what',task.get('label'))
                matched = mt.select(timings,plan.get('cwd','.'),label,task.get('host'))
                if not matched:
                    project = lr.identity(str(Path(plan.get('cwd','.')).resolve()))
                    matched = [r for r in timings if r['project']==project and r.get('label')==label and 'what' not in r]
                values = [r['seconds'] for r in matched]
                if not values: raise ValueError('no measured timings for task '+task['id'])
            if any(not math.isfinite(v) or v < 0 for v in values): raise ValueError('machine/wait times must be finite and nonnegative')
            pairs = [(v,1) for v in values]
        elif typ == 'agent':
            cwd = task.get('cwd',plan.get('cwd','.'))
            brief = Path(task['brief']).read_text() if task.get('brief') else ''
            base = lr.git(cwd,'rev-parse','HEAD').strip() or None
            query = lr.make_row(start=now,end=now+1,model=task['model'],kind=task['kind'],brief=brief,project=cwd,cwd=cwd,base=base,session=plan.get('id','unrecorded-plan'),expect_s=task.get('forecast_s'))
            query['features']['human_minutes'] = task.get('human_minutes',0)
            query['features']['verifiable'] = int(task.get('verifiable',False))
            query['effort'] = task.get('effort','unknown')
            mode = plan.get('forecast_method', config.get('forecast_method','ridge'))
            if mode=='quantile':
                from agentpace import quantiles as qt
                support_source='interpolated-quantiles'
                query=qt.annotate(query,brief,delegated=task.get('delegated',True),ordinal=task.get('ordinal',1),effort=task.get('effort','unknown'),active_runs=task.get('active_runs'),human_minutes=task.get('human_minutes'))
                if config.get('quantile_model'): query['_quantile_model']=config['quantile_model']
            pairs = rf.distribution(history,query,mode)
            task_context = rf.context(query,history)
            from agentpace import pause_clock
            separate_risks=pause_clock.risk_terms(dict(query,budget_remaining_pct=task.get('budget_remaining_pct')))
        else: raise ValueError('task type must be agent, machine or wait')
        if status == 'running':
            conditioned = rf.remaining(pairs,elapsed)
            pairs = conditioned['pairs']; support=conditioned['support']; extrapolated=conditioned['extrapolated']
        durations = [rf.weighted_quantile(pairs,rank) for rank in ranks]
        committed = rf.weighted_quantile(pairs,cut)
        samples[task['id']] = durations; cuts[task['id']] = committed
        summaries.append(dict(id=task['id'],type=typ,status=status,appetite=name,quantile=cut,source=source,needs_owner=needs_owner,commit=committed,p50=rf.weighted_quantile(pairs,.5),elapsed_s=elapsed,support=support,extrapolated=extrapolated,context=task_context,support_source=support_source,separate_risks=separate_risks))
    for entry in tradeoffs:
        if entry.get('mechanism') == 'extra-agents':
            entry['eta_savings_s'] = schedule(tasks,cuts,int(config.get('agent_cap',1)))-schedule(tasks,cuts,cap)
        else:
            restored=dict(cuts); restored[entry['task']] += entry['savings_s']
            entry['eta_savings_s'] = schedule(tasks,restored,cap)-schedule(tasks,cuts,cap)
    for entry in available:
        if entry.get('mechanism') == 'extra-agents':
            entry['eta_gain_s'] = max(0,schedule(tasks,cuts,cap)-schedule(tasks,cuts,int(entry['agent_cap'])))
        elif entry.get('task') in cuts:
            shorter=dict(cuts); shorter[entry['task']]=min(shorter[entry['task']],float(entry.get('seconds',shorter[entry['task']])))
            entry['eta_gain_s']=schedule(tasks,cuts,cap)-schedule(tasks,shorter,cap)
    totals=sorted(schedule(tasks,{key:values[i] for key,values in samples.items()},cap) for i in range(draws))
    elapsed_plan=max(0,now-lr.epoch(plan['started_at'])) if plan.get('started_at') is not None else 0
    p50,p80=lr.quantile(totals,.5),lr.quantile(totals,.8)
    commit=schedule(tasks,cuts,cap)
    unbounded=[]
    for summary in summaries:
        task=next(t for t in tasks if t['id']==summary['id'])
        uncertain=summary['type']=='agent' and task.get('kind')=='research' and summary['status'] not in ('done','removed')
        summary.update(reliable=not uncertain,confidence='low' if uncertain else 'modelled',unbounded=uncertain)
        if uncertain:
            unbounded.append(summary['id'])
            summary['research_history']=research.history(rows)
            summary['warning']=research.warning(summary['research_history'])
    return dict(reliable=not unbounded,unbounded_tasks=unbounded,commit_is_actionable=not unbounded,eta_status='illustrative-unbounded-research' if unbounded else 'modelled',commit=commit,p50=p50,p80=p80,experimental=any(t['support_source']=='interpolated-quantiles' for t in summaries),total_p50=elapsed_plan+p50,total_p80=elapsed_plan+p80,elapsed_s=elapsed_plan,actual_work_s=actual_work,tasks=summaries,tradeoffs=tradeoffs,available_spend=available,agent_cap=cap,deadline_met=None if unbounded or not plan.get('target_s') else commit<=plan['target_s'],dependence='shared residual rank across remaining tasks',what_would_change_it=['scope changes','running-task tail support','machine host/contention','owner-authorized policy changes'])


def cmd_eta(args):
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    plan.setdefault('cwd', str(plan_path.parent))
    if not Path(plan['cwd']).is_absolute():
        plan['cwd'] = str((plan_path.parent / plan['cwd']).resolve())
    for task in plan['tasks']:
        if task.get('brief') and not Path(task['brief']).is_absolute():
            task['brief'] = str(plan_path.parent / task['brief'])
    if args.risk_appetite:
        plan['risk_appetite'] = args.risk_appetite
    if args.commit_quantile is not None:
        plan['commit_quantile'] = args.commit_quantile
    for override in args.task_risk:
        task_id, separator, policy = override.partition('=')
        if not separator or policy not in CUTS:
            raise ValueError('--task-risk needs TASK=conservative|balanced|aggressive')
        task = next((t for t in plan['tasks'] if t['id'] == task_id), None)
        if task is None:
            raise ValueError('unknown task in --task-risk')
        task['risk_appetite'] = policy
        task.pop('commit_quantile', None)
    plan['apply'] = list(dict.fromkeys(plan.get('apply', []) + args.apply))
    config = json.loads(Path(args.config).read_text()) if Path(args.config).exists() else {}
    rows = list(lr.json_lines(args.data)) if Path(args.data).exists() else []
    policy=paths.home()/'model-policy.json'
    if policy.exists(): config.setdefault('forecast_method',json.loads(policy.read_text())['method'])
    if config.get('forecast_method')=='quantile' and Path(args.data).resolve()!=(paths.home()/'local-runs.jsonl').resolve():
        from agentpace import quantiles as qt
        artifact=qt.load_runtime()
        config['quantile_model']=dict(reference=rows,drop=[]) if 'reference' in artifact else qt.fit(rows,**artifact['settings'])
    timings = list(lr.json_lines(args.timings)) if Path(args.timings).exists() else []
    from agentpace import plan_state as ps, machine_timings as mt
    persistent = 'events' in plan
    if persistent:
        ps.sync_ledger(plan_path,args.ledger)
        refreshed=ps.load(plan_path)
        for original in plan['tasks']:
            current=next((t for t in refreshed['tasks'] if t['id']==original['id']),None)
            if current:
                original.update({k:v for k,v in current.items() if k in ('status','started_at','ended_at','actual_s')})
        if 'started_at' in refreshed: plan['started_at']=refreshed['started_at']
    timings.extend(mt.all_records())
    request_scope = dict(allow_spend=sorted(args.allow_spend),risk_appetite=args.risk_appetite,commit_quantile=args.commit_quantile,task_risk=sorted(args.task_risk),tradeoffs=[entry for entry in plan.get('tradeoffs',[]) if entry['id'] in args.allow_spend])
    grant_record = args.owner_grant
    if not grant_record:
        for previous in reversed(plan.get('forecasts',[])):
            if previous.get('owner_grant') and previous['snapshot'].get('grant_request') == request_scope:
                grant_record = dict(previous['owner_grant'],reused_from=previous['id'])
                break
    effective_grant = grant_record.get('quote') if isinstance(grant_record,dict) else grant_record
    plan['grant_request'] = request_scope
    result = simulate(plan, rows, config, keep=args.keep, no_defer=args.no_defer, timings=timings,allow_spend=args.allow_spend,owner_grant=effective_grant,now=args.now)
    if not persistent:
        # Legacy ad-hoc plans are upgraded before their first issued ETA.
        if not plan_path.with_name(plan_path.stem+'.state.json').exists():
            ps.create(plan_path.with_name(plan_path.stem+'.state.json'),plan)
        plan_path=plan_path.with_name(plan_path.stem+'.state.json')
    issued=ps.issue(plan_path,result,now=args.now,owner_grant=grant_record,snapshot=plan)
    result['issued_id']=issued['id']
    result['plan_file']=str(plan_path)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if not result['reliable']: print('Low-confidence plan: research tasks are unbounded; totals below are illustrative, not a commitment.')
        print(f"{plan.get('deliverable', 'Deliverable')} ETA: commit {result['commit']/60:.1f} min | P50 {result['p50']/60:.1f} min | slip P80 {result['p80']/60:.1f} min")
        for task in result['tasks']:
            print(f"  {task['id']}: {task['appetite']} P{task['quantile']*100:g} ({task['source']}), cut {task['commit']:.0f}s")
        for entry in result['tradeoffs']:
            print(f"  Trade-off {entry['id']}: saves {entry['eta_savings_s']:.0f}s on the ETA ({entry['savings_s']:.0f}s task work); risk: {entry['risk']}; basis: {entry['basis']}")
        if result['deadline_met'] is False:
            print('Target is not met. No undeclared work was removed.')
        from agentpace import pause_clock
        risk_task=next((t for t in result['tasks'] if t.get('separate_risks')),None)
        if risk_task:
            print(pause_clock.usage_line(risk_task['separate_risks']['usage_wait']))
            long=risk_task['separate_risks']['over_12h']
            if 'probability' in long: print(f"Per-run >12h risk: {long['probability']:.2%} from earlier outcomes; membership is unknown at dispatch.")
        for task in result['tasks']:
            if not task['reliable']: print(f"  {task['id']}: {task['warning']}")
        for entry in result['available_spend']:
            print(f"  Available with {entry['available_with']} and owner grant: saves {entry.get('eta_gain_s',0):.0f}s; spends {json.dumps(entry['cost'])}; needs_owner=true")
        for task in result['tasks']:
            if task['needs_owner']: print(f"  {task['id']}: requested appetite needs_owner=true; configured cut retained")
        print('What changes this ETA: scope, running-task tail support, host/contention, owner-authorized policy changes.')
    if args.html:
        from agentpace import eta_html, state_html
        state_html.write(args.html, eta_html.render(result, plan, keep=args.keep, no_defer=args.no_defer))
    return 0


def add_command(commands):
    parser = commands.add_parser('eta', help='local deliverable ETA: commit, median and slip line')
    parser.add_argument('plan')
    parser.add_argument('--risk-appetite', choices=tuple(CUTS))
    parser.add_argument('--commit-quantile', type=float)
    parser.add_argument('--task-risk', action='append', default=[])
    parser.add_argument('--keep', action='append', default=[])
    parser.add_argument('--no-defer', action='append', default=[])
    parser.add_argument('--apply', action='append', default=[])
    parser.add_argument('--config', default=str(Path.home()/'.config/agentpace/config.json'))
    parser.add_argument('--data', default=str(paths.home()/'local-runs.jsonl'))
    parser.add_argument('--timings', default=str(paths.home()/'machine-times.jsonl'))
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--html', metavar='PATH', help='also write the ETA as one self-contained HTML page with the --json snapshot embedded')
    parser.add_argument('--allow-spend',action='append',default=[])
    parser.add_argument('--owner-grant',help='verbatim owner quote authorizing the requested override; logged, never invented')
    parser.add_argument('--ledger',default=str(paths.vigil_ledger()))
    parser.add_argument('--now',type=float,help='explicit replay timestamp; omit for live ETA')
    parser.set_defaults(handler=cmd_eta)
