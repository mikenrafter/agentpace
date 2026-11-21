"""Replay caller-provided historical plans using only earlier completed sessions.

Input JSON: [{"asked_at": "2000-01-01T00:00:00Z", "actual_s": 100,
              "plan": {"tasks": [...]}}]. Private inputs stay outside the repo.
"""
import argparse
import json
from pathlib import Path
from agentpace import local_runs as lr,deliverable as dl,paths


def replay(records,rows):
    results=[]
    for record in records:
        cutoff=lr.epoch(record['asked_at'])
        excluded={r['session'] for r in rows if r['end']>=cutoff}
        train=[r for r in rows if r['end']<cutoff and r['session'] not in excluded]
        if not train:
            results.append(dict(asked_at=record['asked_at'],status='insufficient earlier rows'));continue
        result=dl.simulate(record['plan'],train,{},draws=1000,now=cutoff)
        results.append(dict(asked_at=record['asked_at'],actual_s=record['actual_s'],train=len(train),**{k:result[k] for k in ('commit','p50','p80','reliable')}))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--rows',type=Path,default=paths.home()/'local-runs.jsonl')
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    result=replay(json.loads(args.input.read_text()),list(lr.json_lines(args.rows)))
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
