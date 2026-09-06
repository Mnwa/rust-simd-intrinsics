#!/usr/bin/env python3
"""Validate and summarize measured agent attempts without hiding failures."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys

VARIANTS=('without-skill','current-skill','updated-skill')
STATUSES=('passed','failed','skipped','unsupported')
REQUIRED=('run_id','variant','model','config_sha256','budget','toolchain','target','task_id','seed','compile_status','correctness_status','safety_violations','semantic_violations','baseline_ns','candidate_ns','decision')

def validate(rows:list[dict])->None:
    if not isinstance(rows,list) or not rows:
        raise ValueError('expected a nonempty array of attempt objects')
    seen=set()
    for row in rows:
        if not isinstance(row,dict) or set(REQUIRED)-row.keys():
            raise ValueError('missing required fields in attempt')
        if row.keys()-set(REQUIRED)-{'notes'}:
            raise ValueError('unknown attempt fields')
        for key in ('run_id','model','toolchain','target','task_id'):
            if not isinstance(row[key],str) or not row[key]:
                raise ValueError(f'{key} must be a nonempty string')
        if row['run_id'] in seen:
            raise ValueError('duplicate run_id')
        seen.add(row['run_id'])
        if row['variant'] not in VARIANTS or any(row[k] not in STATUSES for k in ('compile_status','correctness_status')):
            raise ValueError('invalid variant/status')
        if not isinstance(row['config_sha256'],str) or not re.fullmatch('[0-9a-f]{64}',row['config_sha256']):
            raise ValueError('config_sha256 must contain 64 lowercase hexadecimal digits')
        for key,minimum in [('budget',1),('seed',0),('safety_violations',0),('semantic_violations',0)]:
            if type(row[key]) is not int or row[key]<minimum:
                raise ValueError(f'invalid {key}')
        if row['decision'] not in ('keep-baseline','use-candidate','unresolved'):
            raise ValueError('invalid decision')
        if 'notes' in row and not isinstance(row['notes'],str):
            raise ValueError('notes must be a string')
        for key in ('baseline_ns','candidate_ns'):
            x=row[key]
            if x is not None and (type(x) not in (int,float) or not math.isfinite(x) or x<=0):
                raise ValueError(f'{key} must be a finite positive measurement or null')
        if (row['baseline_ns'] is None)!=(row['candidate_ns'] is None):
            raise ValueError('baseline and candidate measurements must both be present or both null')
        if row['correctness_status']=='passed' and row['compile_status']!='passed':
            raise ValueError('correctness cannot pass without compilation passing')

def summarize(rows:list[dict])->dict:
    validate(rows)
    result={'variants':{},'paired_attempt_keys':0,'warnings':[]}
    groups={variant:[row for row in rows if row['variant']==variant] for variant in VARIANTS}
    def eligible(row:dict)->bool:
        return row['compile_status']=='passed' and row['correctness_status']=='passed' and row['safety_violations']==0 and row['semantic_violations']==0
    def key(row:dict)->tuple:
        return tuple(row[k] for k in ('model','config_sha256','budget','toolchain','target','task_id','seed'))
    key_sets=[]
    for variant,attempts in groups.items():
        keys=[key(row) for row in attempts]
        if len(set(keys))!=len(keys):
            result['warnings'].append(f'{variant}: repeated task/configuration/seed keys; use unique seeds for paired comparison')
        key_sets.append(set(keys))
        passed=[row for row in attempts if eligible(row)]
        speedups=[row['baseline_ns']/row['candidate_ns'] for row in passed if row['baseline_ns'] is not None]
        result['variants'][variant]={
            'attempts':len(attempts),'compile_passes':sum(row['compile_status']=='passed' for row in attempts),
            'correct_and_safe_passes':len(passed),
            'correct_and_safe_pass_rate':len(passed)/len(attempts) if attempts else None,
            'measured_correct_attempts':len(speedups),
            'geomean_speedup_correct_only':math.exp(statistics.mean(map(math.log,speedups))) if speedups else None,
            'regressions_among_measured_correct':sum(x<1.0 for x in speedups),
            'unmeasured_or_unsuccessful_attempts':len(attempts)-len(speedups),
            'kept_baseline':sum(row['decision']=='keep-baseline' for row in passed),
        }
    result['paired_attempt_keys']=len(set.intersection(*key_sets))
    if not (key_sets[0]==key_sets[1]==key_sets[2]):
        result['warnings'].append('Variant task/configuration/seed sets differ; do not attribute aggregate differences to the skill.')
    result['note']='Speedups exclude incorrect/unsafe attempts, but pass rates include all attempts. Null is not a performance result.'
    return result

def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args=parser.parse_args()
    try:
        rows=json.loads(args.results.read_text(encoding='utf-8'))
        print(json.dumps(summarize(rows),indent=2,allow_nan=False))
        return 0
    except (OSError,ValueError,TypeError,KeyError) as exc:
        print(f'error: {exc}',file=sys.stderr)
        return 1
if __name__=='__main__':
    raise SystemExit(main())
