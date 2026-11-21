"""Local pre-run pilot. Rows contain derived features, never brief text.

Wall time is the target for vigil. This store is separate from the upstream
shareable schema and must not be passed to share/site commands.
"""
from __future__ import annotations
from agentpace import paths

import hashlib
import json
import math
import re
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

QUANTILES = (.5, .8, .9)
NUMERIC = ('items', 'files', 'phases', 'words', 'code_lines', 'age_days', 'churn', 'git_known', 'brief_known')


def epoch(value):
    if isinstance(value, (float, int)):
        return float(value)
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def identity(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:20]


# Model families by whole word. Shared by every reader (learn, calibration, estimate-run) so a
# learned row and a query name the same model. Extend as models arrive.
MODEL_FAMILIES = ('haiku', 'sonnet', 'opus', 'fable', 'sol', 'neon', 'luna')
_PROVIDER_WORDS = frozenset({'claude', 'gemini', 'gpt', 'antigravity', 'anthropic', 'openai', 'google'})
_GENERIC_WORDS = frozenset({'default', 'pro', 'flash', 'mini', 'latest', 'preview', 'auto'})


def _words(text):
    return [w for w in re.split(r'[^a-z0-9.]+', str(text or '').lower()) if w]


def model_word(text):
    """The model family a free-form label names by whole word (`sol-parser-run` -> sol), else None."""
    words = _words(text)
    return next((name for name in MODEL_FAMILIES if name in words), 'codex' if 'codex' in words else None)


def model_name(value):
    """Family of a model id by whole word; idempotent (`model_name(model_name(x)) == model_name(x)`).

    `claude-fable-5-2` -> fable, `gemini-4-neon` -> neon, `gpt-5.3-codex` -> codex, `console-1` ->
    console (never sol). An id with no family keeps its first non-provider word, or itself.
    """
    lowered = str(value or '').lower().strip()
    if not lowered:
        return 'unknown'
    words = _words(lowered)
    family = next((name for name in MODEL_FAMILIES if name in words), None)
    if family:
        return family
    if 'codex' in words or any(w == 'gpt' or re.fullmatch(r'gpt[\d.]+\w*', w) for w in words):
        return 'codex'
    rest = [w for w in words if w not in _PROVIDER_WORDS and not re.fullmatch(r'[\d.]+', w)]
    if not rest or rest[0] in _GENERIC_WORDS:
        return lowered
    return rest[0]


KINDS = ('review', 'fix', 'red', 'research', 'build', 'tests', 'implementation', 'other')
KIND_ALIASES = {'impl': 'implementation', 'implement': 'implementation', 'test': 'tests', 'bugfix': 'fix', 'reviews': 'review'}


def kind_name(value):
    """An explicit kind, normalized: known kinds and their abbreviations (`impl`) map to the
    keyword kind; any other explicit kind is kept as written, never guessed."""
    lowered = str(value or '').strip().lower()
    if not lowered:
        return 'other'
    return KIND_ALIASES.get(lowered, lowered)


def normalize_rows(rows):
    """Rows stored before the shared normalizer, relabelled with it (copies; inputs untouched)."""
    return [dict(r, model=model_name(r.get('model')), kind=kind_name(r.get('kind'))) for r in rows]


def task_kind(text):
    # Order gives the explicit role at the beginning priority over incidental mentions.
    head = text.lower()[:220].replace('-', ' ')
    rules = [('review', r'review|audit|\bcr\d'), ('fix', r'\bfix|bugfix'), ('red', r'\bred\b|failing tests|write tests'), ('research', r'research|investigat|explor'), ('build', r'\bapk\b|\bbuild\b'), ('tests', r'\bsuite\b|\bgate\b'), ('implementation', r'implement|feature|\bgreen\b|refactor')]
    for kind, pattern in rules:
        if re.search(pattern, head):
            return kind
    return 'other'


def git(cwd, *args):
    if not args or args[0] not in ('show', 'log', 'ls-tree', 'rev-parse'):
        raise ValueError('only local git metadata reads are allowed')
    try:
        return subprocess.check_output(['git', '-C', str(cwd), *args], stderr=subprocess.DEVNULL, text=True, timeout=8)
    except (OSError, subprocess.SubprocessError):
        return ''


def extract_features(brief='', cwd=None, base=None):
    names = sorted(set(re.findall(r'(?<![\w/])(?:[\w.-]+/)*[\w.-]+\.(?:py|rs|kt|kts|java|tsx?|jsx?|md|nix|go|c|cpp|h|json|toml|yaml|yml)\b', brief)))
    directories = re.findall(r'`([\w./-]+/)`', brief)
    features = dict(items=len(re.findall(r'^\s*(?:[-*+] |\d+[.)] )', brief, re.M)), files=len(names), phases=len(re.findall(r'\bphase\s+\d+', brief, re.I)), words=len(brief.split()), code_lines=0, age_days=0, churn=0, git_known=0, brief_known=int(bool(brief)))
    if not cwd or not base or not (names or directories):
        return features
    stamp = git(cwd, 'show', '-s', '--format=%ct', base).strip()
    if not stamp.isdigit():
        return features
    paths = []
    for name in names + directories:
        if name.startswith('/') or '..' in name.split('/'):
            continue
        paths.extend(git(cwd, 'ls-tree', '-r', '--name-only', base, '--', name).splitlines())
    paths = sorted(set(paths))[:100]
    ages = []
    for name in paths:
        content = git(cwd, 'show', f'{base}:{name}')
        features['code_lines'] += len(content.splitlines())
        history = git(cwd, 'log', '-30', '--format=%ct', base, '--', name).splitlines()
        times = [int(t) for t in history if t.isdigit()]
        if times:
            ages.append(max(0, (int(stamp)-max(times))/86400))
        features['churn'] += len(times)
    features['git_known'] = int(bool(paths))
    features['age_days'] = statistics.mean(ages) if ages else 0
    return features


def make_row(*, start, end, model='', kind=None, brief='', project='', session='', cwd=None, base=None, machine_s=0, expect_s=None, source='fixture'):
    start, end = epoch(start), epoch(end)
    if not math.isfinite(start+end) or end <= start:
        raise ValueError('run must have positive finite duration')
    return dict(v=1, start=start, end=end, actual_s=end-start, model=model_name(model), kind=kind_name(kind) if kind else task_kind(brief), project=identity(project), session=identity(session), features=extract_features(brief, cwd, base), machine_s=max(0, float(machine_s)), expect_s=expect_s, source=source)


def json_lines(path):
    with Path(path).open(errors='replace') as handle:
        for line in handle:
            try:
                record=json.loads(line)
                if isinstance(record, dict):
                    yield record
            except ValueError:
                continue


def ledger_row(r):
    """One completed v1 vigil ledger item as a row (brief features, machine time, meta labels).
    Raises ValueError, KeyError, TypeError, AttributeError or OSError for an item that is not one."""
    if r['v'] != 1 or r.get('outcome') not in ('done','completed','success','exited'):
        raise ValueError('not completed')
    meta=r.get('meta') or {}
    # Reject a brief file changed since the run started.
    brief=''
    brief_path=Path(meta['brief']) if meta.get('brief') else None
    if brief_path and brief_path.is_file() and brief_path.stat().st_mtime <= epoch(r['started_at']):
        brief=brief_path.read_text()
    row=make_row(brief=brief,cwd=r.get('cwd'),base=meta.get('base'),start=r['started_at'], end=r['ended_at'], model=meta.get('model',''), kind=meta.get('kind') or task_kind(r['label']), project=r.get('cwd',''), session=r.get('session_id',''), expect_s=r.get('expect_s'), machine_s=float(meta.get('machine_s',0)), source='vigil-ledger')
    row['id']=identity((r['hold'],r['label'],row['start']))
    return row


def read_ledger(path):
    rows, skipped, seen = [], 0, set()
    if not Path(path).exists():
        return rows, skipped
    for line in Path(path).read_text().splitlines():
        try:
            row=ledger_row(json.loads(line))
            if row['id'] in seen:
                continue
            seen.add(row['id']); rows.append(row)
        except (ValueError, KeyError, TypeError, AttributeError, OSError):
            skipped+=1
    return rows, skipped


def content_text(content):
    if isinstance(content,str):
        return content
    if isinstance(content,list):
        return '\n'.join(b.get('text','') for b in content if isinstance(b,dict))
    return ''


def read_codex(path):
    meta, turn, rows, model = {}, None, [], ''
    for r in json_lines(path):
        p=r.get('payload') or {}; typ=r.get('type'); event=p.get('type')
        if typ=='session_meta':
            meta=p
            if meta.get('forked_from_id'):
                return []
        elif typ=='turn_context':
            model=p.get('model',model)
            if turn:
                turn['model']=model
        elif typ=='event_msg' and event=='task_started':
            turn=dict(start=p.get('started_at') or r.get('timestamp'), brief='', model=model, id=p.get('turn_id'))
        elif turn and not turn['brief'] and ((typ=='event_msg' and event=='user_message') or (typ=='response_item' and p.get('role')=='user')):
            candidate=p.get('message') or content_text(p.get('content'))
            if not candidate.lstrip().startswith(('# AGENTS.md', '<environment_context>', '<permissions instructions>')):
                turn['brief']=candidate
        elif turn and typ=='event_msg' and event in ('task_complete','turn_aborted'):
            if p.get('turn_id') and p['turn_id']!=turn['id']:
                continue
            if event=='task_complete' and not p.get('error') and turn['model']!='codex-auto-review':
                try:
                    commit=(meta.get('git') or {}).get('commit_hash')
                    # Session base is only valid for its first turn.
                    rows.append(make_row(start=turn['start'],end=p.get('completed_at') or r['timestamp'],model=turn['model'],brief=turn['brief'],project=meta.get('cwd',''),session=meta.get('id',str(path)),cwd=meta.get('cwd'),base=commit if not rows else None,source='codex'))
                except (ValueError,TypeError):
                    pass
            turn=None
    return rows


def read_claude(path):
    # Only isolated subagents: main session wall spans include owner idle time.
    start=end=None; brief=model=cwd=''; session=str(path)
    for r in json_lines(path):
        msg=r.get('message') or {}; typ=r.get('type')
        if typ=='user' and not brief:
            brief=content_text(msg.get('content'))
            start=r.get('timestamp'); cwd=r.get('cwd','')
        if typ=='assistant':
            model=msg.get('model') or model
            if msg.get('stop_reason')=='end_turn':
                end=r.get('timestamp')
    if start and end and brief:
        try:
            return [make_row(start=start,end=end,model=model,brief=brief,project=cwd,session=session,source='claude-subagent')]
        except (ValueError,TypeError):
            pass
    return []


def read_pairs(pairs):
    rows=[]
    projects={}
    for p in pairs:
        if not p.get('end'):
            continue
        b=p.get('brief') or {}
        # Stale preceding delegates are weak labels. Do not use a future brief.
        valid=0 <= p['start']-b.get('t',-1e20) <= 900
        brief=b.get('prompt','') if valid else ''
        if p['file'] not in projects:
            projects[p['file']]=next((r['cwd'] for r in json_lines(p['file']) if r.get('cwd')),str(Path(p['file']).parent))
        try:
            row=make_row(start=p['start'],end=p['end'],model=next((m for m in ('sol','opus','sonnet','haiku') if m in p.get('label','').lower()),b.get('model','') if valid else ''),kind=(task_kind(p.get('label','')) if task_kind(p.get('label',''))!='other' else task_kind(brief)),brief=brief,project=projects[p['file']],session=p['file'],expect_s=p.get('expect_s'),source='vigil-pair')
            row['brief_link']='preceding-15m' if valid else 'missing'
            rows.append(row)
        except (ValueError,TypeError):
            pass
    return rows


def vector(row, drop=()):
    features={'intercept':1.0}
    for name in NUMERIC:
        if 'scope' not in drop or name not in ('items','files','phases','words','brief_known'):
            if 'git' not in drop or name not in ('code_lines','age_days','churn','git_known'):
                features[name]=math.log1p(max(0,row['features'].get(name,0)))
    if 'model' not in drop:
        features['model='+row['model']]=1
    if 'kind' not in drop:
        features['kind='+row['kind']]=1
    if 'interaction' not in drop:
        features['model-kind='+row['model']+'/'+row['kind']]=1
    if 'project' not in drop:
        features['project='+row['project']]=1
    return features


def fit(rows, drop=()):
    if len(rows)<10:
        raise ValueError('need at least 10 completed local runs; run pilot-ingest')
    vectors=[vector(r,drop) for r in rows]
    keys=sorted(set().union(*(v.keys() for v in vectors)))
    columns=[[v.get(k,0) for v in vectors] for k in keys]
    targets=[math.log(max(1,r['actual_s']-(0 if 'machine' in drop else r.get('machine_s',0)))) for r in rows]
    weights=[0.0]*len(keys); residual=targets[:]
    # Coordinate descent ridge, with an unpenalized intercept. No dependency.
    for _ in range(80):
        change=0
        for j,(key,column) in enumerate(zip(keys,columns)):
            penalty=0 if key=='intercept' else 12
            new=sum(x*(e+x*weights[j]) for x,e in zip(column,residual))/(sum(x*x for x in column)+penalty)
            delta=new-weights[j]; change=max(change,abs(delta)); weights[j]=new
            residual=[e-x*delta for e,x in zip(residual,column)]
        if change<1e-7:
            break
    return dict(weights=dict(zip(keys,weights)), residual=sorted(residual), drop=drop, n=len(rows))


def quantile(values,q):
    position=(len(values)-1)*q; low=int(position); high=min(low+1,len(values)-1)
    return values[low]+(values[high]-values[low])*(position-low)


def predict(model,row):
    terms={k:v*model['weights'].get(k,0) for k,v in vector(row,model['drop']).items()}
    center=sum(terms.values()); machine=0 if 'machine' in model['drop'] else row.get('machine_s',0)
    if machine and row['kind'] in ('tests','build'):
        return dict(p50=machine,p80=machine,p90=machine,machine_s=machine,log_terms={},rows=model['n'])
    result={f'p{int(q*100)}':machine+math.exp(min(20,center+quantile(model['residual'],q))) for q in QUANTILES}
    result.update(machine_s=machine,log_terms=terms,rows=model['n'])
    return result


def week(stamp):
    return datetime.fromtimestamp(stamp,timezone.utc).strftime('%G-W%V')


def time_folds(rows,minimum=30):
    groups=defaultdict(list)
    for r in rows:
        groups[r['session']].append(r)
    for w in sorted({week(min(r['start'] for r in group)) for group in groups.values()}):
        test=[r for group in groups.values() if week(min(x['start'] for x in group))==w for r in group]
        boundary=min(r['start'] for r in test)
        train=[r for group in groups.values() if max(x['end'] for x in group)<boundary and week(min(x['start'] for x in group))<w for r in group]
        if len(train)>=minimum:
            yield w,train,test


def pinball(actual,predicted,q):
    error=actual-predicted
    return max(q*error,(q-1)*error)


def metrics(predictions):
    if not predictions:
        return {}
    n=len(predictions)
    result={'n':n}
    for q in QUANTILES:
        key=f'p{int(q*100)}'
        result['loss_'+key]=sum(pinball(a,p[key],q) for a,p in predictions)/n
        result['coverage_'+key]=sum(a<=p[key] for a,p in predictions)/n
    result['width']=statistics.mean(p['p90']-p['p50'] for _,p in predictions)
    return result


def upstream_rows(rows):
    return [dict(duration_s=max(1,round(r['actual_s'])),outcome='completed',task_kind='other',model=r['model'],risk='normal',parallel_agents=1,batch=r['session'],offset_s=round(r['start']-min(x['start'] for x in rows if x['session']==r['session'])),week=week(r['start']),harness='codex') for r in rows]


def evaluate(rows):
    from agentpace import estimate
    scores=defaultdict(list); folds=[]; details=[]
    variants={'ours':(), **{'without_'+f:(f,) for f in ('scope','git','model','kind','interaction','project','machine')}}
    for w,train,test in time_folds(rows):
        # Compare all methods on identical rows with an actual agent forecast.
        test=[r for r in test if r.get('expect_s') and r['source'].startswith('vigil')]
        if not test:
            continue
        folds.append(dict(week=w,train=len(train),test=len(test)))
        for name,drop in variants.items():
            model=fit(train,drop)
            for r in test:
                prediction=predict(model,r)
                scores[name].append((r['actual_s'],prediction))
                if name=='ours':
                    details.append(dict(kind=r['kind'],model=r['model'],actual_s=r['actual_s'],expect_s=r['expect_s'],prediction=prediction))
        shipped=estimate.fit(upstream_rows(train))
        for r in test:
            # Use the actual shipped simulation for a single run.
            task=estimate.Task(id='run',kind='other',risk='normal',model=r['model'])
            forecast=estimate.estimate(shipped,[task],1,draws=1000,seed=20261120)
            pred={key:getattr(forecast,key) for key in ('p50','p80','p90')}
            scores['upstream'].append((r['actual_s'],pred))
            scores['agents'].append((r['actual_s'],{f'p{int(q*100)}':r['expect_s'] for q in QUANTILES}))
    return dict(folds=folds,metrics={name:metrics(pairs) for name,pairs in scores.items()},by_kind={kind:dict(ours=metrics([(r['actual_s'],r['prediction']) for r in details if r['kind']==kind]),agents=metrics([(r['actual_s'],{f'p{int(q*100)}':r['expect_s'] for q in QUANTILES}) for r in details if r['kind']==kind])) for kind in sorted({r['kind'] for r in details})})


def write_rows(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    temporary.replace(path)


def machine_time(records,label,cwd):
    matches=[r['seconds'] for r in records if r['project']==identity(str(Path(cwd).resolve())) and label.casefold()==r['label'].casefold()]
    if not matches:
        raise ValueError('no measured machine time for this project/label; add a timing record')
    return statistics.median(matches)


def cmd_estimate_run(args):
    from time import time
    from agentpace import run_forecast as rf, run_labels as labels, machine_timings as mt
    cwd=str(Path(args.cwd).resolve())
    from agentpace import research
    rows=list(json_lines(args.data))
    if args.learned:
        # The replayed rows hold everything up to their last end; the learned tail extends them.
        from agentpace import learn
        rows+=learn.reference_rows(paths.state_dir(),after=max((r['end'] for r in rows if r.get('end')),default=0))
    # Rows replayed before the shared normalizer hold raw ids (`claude-fable-5-2`, `impl`).
    rows=research.deduplicate(normalize_rows(rows))
    brief=Path(args.brief).read_text() if args.brief else ''
    machine=0
    observed=[]
    if args.machine:
        observed=mt.select(mt.all_records(),cwd,args.machine,args.host)
        if observed: machine=quantile(sorted(r['seconds'] for r in observed),.5)
        else: machine=machine_time(list(json_lines(args.timings)),args.machine,cwd)
    base=git(cwd,'rev-parse','HEAD').strip() or None
    now=time()
    query=make_row(start=now,end=now+1,model=args.model,kind=args.kind,brief=brief,project=cwd,cwd=cwd,base=base,machine_s=machine,expect_s=args.expect,session=args.session)
    query['features'].update(human_minutes=args.human_minutes or 0,verifiable=int(bool(re.search(r'\b(test|tests|suite|compile|assert|lint|build)\b',brief,re.I))))
    query['effort']=args.effort
    policy=paths.home()/'model-policy.json'
    mode=args.method or (json.loads(policy.read_text())['method'] if policy.exists() else 'ridge')
    if mode=='quantile':
        from agentpace import quantiles as qt
        query=qt.annotate(query,brief,delegated=not args.root_run,ordinal=args.ordinal,effort=args.effort,active_runs=args.active_runs,human_minutes=args.human_minutes)
        # The fitted artifact holds its own training rows; other --data, or learned history on
        # top of it, must be refitted (or referenced) from the rows actually read.
        if args.learned or Path(args.data).resolve()!=(paths.home()/'local-runs.jsonl').resolve():
            artifact=qt.load_runtime()
            query['_quantile_model']=dict(reference=rows,drop=[]) if 'reference' in artifact else qt.fit(rows,**artifact['settings'])
    pairs=rf.distribution(rows,query,mode)
    if machine and args.kind in ('tests','build'):
        pairs=[(r['seconds'],1) for r in observed] if observed else [(machine,1)]
    elif machine:
        pairs=[(value+machine,weight) for value,weight in pairs]
    if args.elapsed:
        result=rf.remaining(pairs,args.elapsed)
        result.pop('pairs')
    else: result=rf.forecast(pairs)
    result.update(method=mode,support_source='interpolated-quantiles' if mode=='quantile' else 'empirical-reference',features=query['features'],context=rf.context(query,rows),machine_s=machine,rows=len(rows),elapsed_s=args.elapsed,label_sources={source:sum(r.get('label_source','unknown')==source for r in rows) for source in sorted({r.get('label_source','unknown') for r in rows})})
    if mode=='quantile':
        fitted=query.get('_quantile_model') or qt.load_runtime()
        result['training_rows']=len(fitted['reference']) if 'reference' in fitted else fitted['rows']
        result['experimental']=True
        result['contention_unknown']=query.get('contention_unknown',False)
    from agentpace import pause_clock
    result['separate_risks']=pause_clock.risk_terms(dict(query,budget_remaining_pct=args.budget_remaining_pct))
    result['reliable']=args.kind!='research'
    if args.kind=='research':
        result['research_history']=research.history(rows)
        result['confidence']='low'
        result['warning']=research.warning(result['research_history'])
    if args.expect_s:
        print(math.ceil(result['p80']-1e-9))
    elif args.json:
        print(json.dumps(result,indent=2))
    else:
        print(f"P50 {result['p50']:.0f}s  P80 {result['p80']:.0f}s  P90 {result['p90']:.0f}s remaining")
        print(f"Method {mode}; measured machine +{machine:.0f}s; {len(rows)} reference runs; elapsed {args.elapsed:.0f}s.")
        if result.get('extrapolated'): print('Beyond empirical tail support; remaining time is extrapolated.')
        print(pause_clock.usage_line(result['separate_risks']['usage_wait']))
        if not result['reliable']: print(result['warning'])
    return 0


def add_commands(commands):
    parser=commands.add_parser('estimate-run',help='local pilot: pre-run wall-time estimate')
    parser.add_argument('--model',required=True)
    parser.add_argument('--kind',required=True)
    parser.add_argument('--brief')
    parser.add_argument('--cwd',default='.')
    parser.add_argument('--machine',help='exact project timing label')
    parser.add_argument('--data',default=str(paths.home()/'local-runs.jsonl'))
    parser.add_argument('--learned',action='store_true',help='add history learned since the replayed rows (agentpace learn)')
    parser.add_argument('--timings',default=str(paths.home()/'machine-times.jsonl'))
    parser.add_argument('--expect-s',action='store_true',help='print P80 seconds as one integer')
    parser.add_argument('--json',action='store_true')
    parser.add_argument('--elapsed',type=float,default=0)
    parser.add_argument('--human-minutes',type=float)
    parser.add_argument('--expect',type=float,help='existing pre-run estimate, if one was actually issued')
    parser.add_argument('--session',default='')
    parser.add_argument('--effort',default='unknown')
    parser.add_argument('--host')
    parser.add_argument('--method',choices=('ridge','reference','expect','recency','context','quantile'))
    parser.add_argument('--budget-remaining-pct',type=float,help='recorded dispatch budget snapshot; omit when unknown')
    parser.add_argument('--active-runs',type=int)
    parser.add_argument('--ordinal',type=int,default=1)
    parser.add_argument('--root-run',action='store_true')
    parser.set_defaults(handler=cmd_estimate_run)


def parse_timings(text,cwd,source):
    """Read only table columns explicitly marked Actual or Took, not estimates.

    Exact project-specific labels preserve host/cache qualifiers in task names.
    Free prose and ranges are intentionally left for manual curation.
    """
    records=[]; header=[]
    for line in text.splitlines():
        if not line.startswith('|'):
            continue
        cells=[c.strip() for c in line.strip('|').split('|')]
        lowered=[c.lower() for c in cells]
        if 'actual' in lowered or 'took' in lowered:
            header=lowered; continue
        if not header or len(cells)!=len(header):
            continue
        column=header.index('actual') if 'actual' in header else header.index('took')
        label_column=next((header.index(k) for k in ('task','check') if k in header),None)
        if label_column is None:
            continue
        duration=cells[column].strip('~ ')
        if any(token in duration for token in ('est','→','–','-','/')):
            continue
        match=re.fullmatch(r'(?:(\d+(?:\.\d+)?)\s*h(?:ours?)?\s*)?(?:(\d+(?:\.\d+)?)\s*m(?:in(?:utes?)?)?\s*)?(?:(\d+(?:\.\d+)?)\s*s(?:ec(?:onds?)?)?\s*)?',duration)
        if not match or not any(match.groups()):
            continue
        seconds=sum(float(v or 0)*unit for v,unit in zip(match.groups(),(3600,60,1)))
        if seconds>0:
            records.append(dict(project=identity(str(Path(cwd).resolve())),label=cells[label_column].strip('`'),seconds=seconds,source=source))
    return records


def merge_ledger(rows, ledger):
    """Preserve simultaneous pair items; dedupe ledger entries by writer identity."""
    known={r['id'] for r in rows if r.get('source')=='vigil-ledger' and 'id' in r}
    return list(rows)+[r for r in ledger if r['id'] not in known]
