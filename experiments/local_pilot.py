"""Reproduce the local pilot. No network; raw briefs stay in memory only.

PYTHONPATH=src python3 experiments/local_pilot.py --pairs /tmp/pairs.json
Pass a local JSON pair export; raw briefs are never persisted in derived rows.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from agentpace import local_runs as lr, paths as locations

parser=argparse.ArgumentParser()
parser.add_argument('--pairs',type=Path)
parser.add_argument('--sessions',action='store_true')
parser.add_argument('--timing',type=Path,action='append',default=[])
parser.add_argument('--ledger',type=Path,default=locations.vigil_ledger())
parser.add_argument('--codex-root',type=Path,default=locations.codex_sessions())
parser.add_argument('--claude-root',type=Path,default=locations.claude_projects())
parser.add_argument('--out',type=Path,default=locations.home())
parser.add_argument('--report',type=Path,default=locations.home()/'pilot-results.json')
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
if args.pairs:
    pairs=json.loads(args.pairs.read_text())
    rows=lr.read_pairs(pairs)
    lr.write_rows(args.out/'local-runs.jsonl',rows)
    print('vigil',len(rows),dict(Counter(r['kind'] for r in rows)),flush=True)
else:
    rows=list(lr.json_lines(args.out/'local-runs.jsonl'))
ledger,skipped=lr.read_ledger(args.ledger)
rows=lr.merge_ledger(rows,ledger)
if ledger:
    lr.write_rows(args.out/'local-runs.jsonl',rows)
if args.sessions:
    session_rows=[]
    for source,paths,reader in [('codex',args.codex_root.rglob('rollout-*.jsonl'),lr.read_codex),('claude',args.claude_root.rglob('*.jsonl'),lr.read_claude)]:
        count=0
        for path in paths:
            session_rows.extend(reader(path)); count+=1
        print(source,'files',count,'total rows',len(session_rows),flush=True)
    lr.write_rows(args.out/'session-runs.jsonl',session_rows)
machine_records=[]
for timing in args.timing:
    cwd=timing.parent.with_name(timing.parent.name.removesuffix('.research'))
    machine_records.extend(lr.parse_timings(timing.read_text(),str(cwd),str(timing)))
lr.write_rows(args.out/'machine-times.jsonl',machine_records)
result=lr.evaluate(rows)
result['inventory']={'rows':len(rows),'sources':dict(Counter(r['source'] for r in rows)),'models':dict(Counter(r['model'] for r in rows)),'briefs':sum(r['features']['brief_known'] for r in rows),'git':sum(r['features']['git_known'] for r in rows),'ledger_skipped':skipped}
args.report.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
