"""Structured local machine observations and best-effort Markdown migration."""
from __future__ import annotations
from agentpace import paths

import json
import fcntl
import math
import re
import statistics
import uuid
from collections import defaultdict
from pathlib import Path
from time import time
from agentpace import local_runs as lr

INDEX = paths.home()/'timings-index.json'
# One host name per line; `host <name>` in a migrated TIMINGS header is matched only against these.
HOSTS = Path.home()/'.config/agentpace/hosts'


def known_hosts(path=None):
    try: lines=(path or HOSTS).read_text().splitlines()
    except OSError: return ()
    return tuple(h for h in (x.split('#',1)[0].strip().casefold() for x in lines) if re.fullmatch(r'[a-z0-9][a-z0-9_.:-]*',h))


def key(text):
    text=text.casefold()
    if 'apk' in text: return 'build:apk'
    if 'full' in text and ('suite' in text or 'test' in text or 'gate' in text): return 'suite:full'
    if 'sweeps' in text: return 'suite:sweeps'
    if 'suite' in text or 'test' in text: return 'suite:targeted'
    if 'release' in text: return 'build:release'
    if 'boot' in text or 'emulator' in text: return 'emulator:boot'
    if 'upload' in text or 'scp' in text or 'transfer' in text: return 'remote:transfer'
    if 'build' in text or 'compil' in text or 'assembl' in text: return 'build:compile'
    if 'lint' in text or 'analy' in text: return 'check:lint'
    return 'unclassified'


def duration(text):
    value=text.strip().strip('~≈ ')
    # A leading scalar before an annotation is acceptable; a range is not.
    value=re.split(r'\s*\(',value)[0].strip()
    if re.search(r'\d\s*[–-]\s*\d|est\b|unknown',value,re.I): return None
    clock=re.fullmatch(r'(\d+):(\d{2})(?::(\d{2}))?',value)
    if clock:
        a,b,c=clock.groups(); return int(a)*3600+int(b)*60+int(c) if c else int(a)*60+int(b)
    compact=re.fullmatch(r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?',value)
    if compact and any(compact.groups()): return sum(float(v or 0)*u for v,u in zip(compact.groups(),(3600,60,1)))
    matches=list(re.finditer(r'(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b',value,re.I))
    # Compact 8m08s needs a letter/digit boundary, not a word boundary.
    if not matches:
        compact=re.fullmatch(r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?',value)
        if compact and any(compact.groups()): return sum(float(v or 0)*u for v,u in zip(compact.groups(),(3600,60,1)))
    if matches:
        leftover=value
        for m in reversed(matches): leftover=leftover[:m.start()]+leftover[m.end():]
        if leftover.strip(' ,~about'):
            return None
        return sum(float(m[1])*(3600 if m[2].lower().startswith('h') else 60 if m[2].lower().startswith('m') else 1) for m in matches)
    return None


def store_for(project):
    path=Path(project).resolve()
    return path.with_name(path.name+'.research')/'timings.jsonl'


def _index_locked(path,project):
    INDEX.parent.mkdir(parents=True,exist_ok=True)
    entries=json.loads(INDEX.read_text()) if INDEX.exists() else {}
    entries[str(Path(project).resolve())]=str(Path(path).resolve())
    temporary=INDEX.with_suffix('.tmp')
    temporary.write_text(json.dumps(entries,indent=2)+'\n'); temporary.replace(INDEX)


def index(path,project):
    INDEX.parent.mkdir(parents=True,exist_ok=True)
    with INDEX.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _index_locked(path,project)


def append(path,record):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle: handle.write(json.dumps(record,sort_keys=True)+'\n')


def record(path,*,project,what,host,took,now=None,n_tests=None,n_files=None,contention=False,note='',started_at=None,run_id=None):
    now=time() if now is None else lr.epoch(now)
    took=float(took)
    if not math.isfinite(took) or took<=0: raise ValueError('took must be positive finite seconds')
    if not re.fullmatch(r'[a-z]+:[a-z0-9_-]+',what): raise ValueError('what must be normalized, e.g. suite:full')
    for count in (n_tests,n_files):
        if count is not None and (not isinstance(count,int) or count<0): raise ValueError('counts must be nonnegative integers')
    item=dict(v=1,id=run_id or uuid.uuid4().hex,project=lr.identity(str(Path(project).resolve())),what=what,label=what,host=host,seconds=took,started_at=now-took if started_at is None else started_at,ended_at=now,n_tests=n_tests,n_files=n_files,contention=bool(contention),note=note,source='record',needs_review=host=='unknown')
    append(path,item)
    return item


def import_text(text,project,source,hosts=None):
    rows=[]; unparsed=[]; header=[]
    names=known_hosts() if hosts is None else tuple(h.casefold() for h in hosts)
    host_match=re.search(r'host\s+('+'|'.join(map(re.escape,names))+r')(?![\w.:-])',text[:500],re.I) if names else None
    host=host_match[1].lower() if host_match else 'unknown'
    for number,line in enumerate(text.splitlines(),1):
        if not line.startswith('|'):
            if re.search(r'^\s*[-*] .*\d.*(?:min|sec|\d+m\d+s)',line):
                unparsed.append(dict(line=number,reason='free prose requires review'))
            continue
        cells=[x.strip() for x in line.strip('|').split('|')]
        lowered=[c.casefold() for c in cells]
        if any(c in lowered for c in ('task','check','job')) and any(c in lowered for c in ('actual','took')):
            header=lowered; continue
        if not header or len(cells)!=len(header) or all(re.fullmatch(r'[: -]+',c or '-') for c in cells): continue
        actual=header.index('actual') if 'actual' in header else header.index('took')
        label=cells[next(header.index(k) for k in ('task','check','job') if k in header)]
        seconds=duration(cells[actual]); normalized=key(label)
        if seconds is None or seconds<=0 or normalized=='unclassified':
            unparsed.append(dict(line=number,reason='duration or task class not parsed'));continue
        # Mixed implementation/rebase tasks cannot supply pure machine measurements.
        if re.search(r'\brebase\b|incl\.|fix.*verify|mockup|implementation',label,re.I):
            unparsed.append(dict(line=number,reason='mixed agent/machine task'));continue
        observed_date=cells[header.index('date')] if 'date' in header else None
        rows.append(dict(v=1,id=lr.identity((source,number,line)),project=lr.identity(str(Path(project).resolve())),what=normalized,label=normalized,host=host,seconds=seconds,started_at=None,ended_at=None,observed_date=observed_date,n_tests=None,n_files=None,contention=('(C)' in line or bool(re.search(r'sharing|other sessions|memory full|3 agents',line,re.I))),note=label,source=source,source_line=number,needs_review=host=='unknown'))
    return rows,unparsed


def all_records(index_path=INDEX):
    paths=json.loads(index_path.read_text()).values() if index_path.exists() else []
    return [r for path in paths if Path(path).exists() for r in lr.json_lines(path)]


def select(records,project,what,host=None):
    return [r for r in records if r['project']==lr.identity(str(Path(project).resolve())) and r.get('what',r.get('label'))==what and (not host or r.get('host')==host) and not r.get('contention') and not r.get('needs_review')]


def emit(value,json_output):
    if json_output:
        print(json.dumps(value,indent=2))
    elif isinstance(value,dict) and 'seconds' in value:
        print(f"Recorded {value['what']}: {value['seconds']:.1f}s on {value['host']}.")
    elif isinstance(value,dict) and 'imported' in value:
        print(f"Imported {value['imported']} timings; {value['unparsed']} lines need review. Originals retained.")
    else:
        print(f"Machine timer {value['id']}: {value['status']}. Stop it with agentpace record stop --id {value['id']}.")


def _record_locked(args):
    now=time(); project=str(Path(args.project).resolve()); path=Path(args.out) if args.out else store_for(project)
    pending=paths.home()/'machine-pending.json'
    active=json.loads(pending.read_text()) if pending.exists() else {}
    if args.action=='start':
        run_id=args.id or uuid.uuid4().hex
        if run_id in active: raise ValueError('machine record already running')
        active[run_id]=dict(project=project,what=args.what,host=args.host,started_at=now,n_tests=args.n_tests,n_files=args.n_files,contention=args.contention,note=args.note,path=str(path))
        result=dict(id=run_id,status='running',started_at=now)
    elif args.action=='stop':
        if not args.id or args.id not in active: raise ValueError('unknown running machine record id')
        item=active.pop(args.id); start=item.pop('started_at'); target=item.pop('path')
        result=record(target,**item,took=now-start,now=now,started_at=start,run_id=args.id)
        index(target,item['project'])
    else:
        if args.took is None: raise ValueError('record needs --took SECONDS, or start/stop')
        result=record(path,project=project,what=args.what,host=args.host,took=args.took,n_tests=args.n_tests,n_files=args.n_files,contention=args.contention,note=args.note,now=now)
        index(path,project)
    pending.parent.mkdir(parents=True,exist_ok=True)
    pending.write_text(json.dumps(active,indent=2)+'\n')
    emit(result,args.json)
    return 0


def cmd_record(args):
    state=paths.home()
    state.mkdir(parents=True,exist_ok=True)
    with (state/'machine-pending.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _record_locked(args)


def cmd_import(args):
    source=Path(args.path).resolve()
    project=Path(args.project).resolve() if args.project else source.parent.with_name(source.parent.name.removesuffix('.research'))
    path=Path(args.out) if args.out else store_for(project)
    rows,unparsed=import_text(source.read_text(),str(project),str(source))
    known={r['id'] for r in lr.json_lines(path)} if path.exists() else set()
    added=[r for r in rows if r['id'] not in known]
    for row in added: append(path,row)
    diagnostic=path.with_name('timings-import.json')
    diagnostic.write_text(json.dumps(dict(source=str(source),unparsed=unparsed),indent=2)+'\n')
    index(path,project)
    emit(dict(imported=len(added),parsed=len(rows),unparsed=len(unparsed),path=str(path),diagnostics=str(diagnostic),needs_review=True),args.json)
    return 0


def cmd_show(args):
    records=all_records(); groups=defaultdict(list)
    for row in records:
        if args.project and row['project']!=lr.identity(str(Path(args.project).resolve())): continue
        groups[(row['project'],row['what'],row['host'],row.get('contention',False),row.get('needs_review',False))].append(row['seconds'])
    result=[dict(project=p,what=k,host=h,contention=c,needs_review=r,n=len(v),p50=lr.quantile(sorted(v),.5),p80=lr.quantile(sorted(v),.8),p90=lr.quantile(sorted(v),.9)) for (p,k,h,c,r),v in sorted(groups.items())]
    if args.json: print(json.dumps(result,indent=2))
    else:
        print('Project | What | Host | n | P50 | P80 | P90 | Review/contention')
        for row in result: print(f"{row['project']} | {row['what']} | {row['host']} | {row['n']} | {row['p50']:.0f}s | {row['p80']:.0f}s | {row['p90']:.0f}s | {row['needs_review']}/{row['contention']}")
    return 0


def add_commands(commands):
    parser=commands.add_parser('record',help='agent: record measured machine time at start/stop or --took')
    parser.add_argument('action',nargs='?',choices=('start','stop'),default=None)
    parser.add_argument('--project',default='.')
    parser.add_argument('--what',default='suite:targeted')
    parser.add_argument('--host',default='unknown')
    parser.add_argument('--id')
    parser.add_argument('--took',type=float)
    parser.add_argument('--n-tests',type=int)
    parser.add_argument('--n-files',type=int)
    parser.add_argument('--contention',action='store_true')
    parser.add_argument('--note',default='')
    parser.add_argument('--out')
    parser.add_argument('--json',action='store_true')
    parser.set_defaults(handler=cmd_record)
    timings=commands.add_parser('timings',help='agent: import old timing notes or inspect machine distributions')
    sub=timings.add_subparsers(dest='timings_action',required=True)
    importer=sub.add_parser('import'); importer.add_argument('path'); importer.add_argument('--project'); importer.add_argument('--out'); importer.add_argument('--json',action='store_true'); importer.set_defaults(handler=cmd_import)
    show=sub.add_parser('show'); show.add_argument('--project'); show.add_argument('--json',action='store_true'); show.set_defaults(handler=cmd_show)
