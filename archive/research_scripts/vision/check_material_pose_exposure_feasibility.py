"""Explicit integer feasibility diagnostic only; no sequence or training authorization."""
import argparse,sys
from collections import Counter
from pathlib import Path
import numpy as np
from scripts.vision.merge_material_pose_candidates import OUT as DATA,prior
from scripts.vision.train_frozen_multiscale import contract

OUT=DATA/'exposure-feasibility-v1'
CLASSES=('transformer','switchgear','capacitor_bank','reactor')


def vector(row):
    return [1]+[row['class_instances'].get(c,0) for c in CLASSES]


def run():
    manifest=DATA/'candidate-export-v1/manifest.json';m=prior.read(manifest);prior.verify(m)
    if m['reference_overlap_gaps'] or len(m['members'])!=36:raise ValueError('Candidate matrix/exclusion failed')
    reviewed=prior.read(DATA/'reviewed-completion.json');prior.verify(reviewed)
    candidates=sorted([x for x in m['members'] if x['variant']=='original'],key=lambda x:x['member_id'])
    if len(candidates)!=12:raise ValueError('Expected twelve originals')
    # Reuse the already frozen local solver distribution; do not install or replace it.
    solver_design=prior.ROOT/'data/research/ml_training_recovery_v1/unified-hold-physical-lighting-control-v1/design.json'
    sd=prior.read(solver_design);prior.verify(sd)
    sf=Path(sd['solver_file'])
    if prior.file_sha256(sf)!=sd['inputs'][str(sf)]:raise ValueError('Solver changed')
    sys.path.insert(0,str(sf.parent.parent))
    from scipy.optimize import milp,Bounds,LinearConstraint
    OUT.mkdir(exist_ok=True);results={};paths=[manifest,DATA/'reviewed-completion.json',solver_design,sf,Path(__file__)]
    for seed in (7,17,27):
        design,source,actual=contract(f'fixed-{seed}')
        pool={x['member_id']:x for x in source['pool_rows']};old=Counter(actual['actual'])
        if actual['actual']!=source['schedules'][f'R-clean-{seed}']:raise ValueError('Reference actual differs')
        if any(old[x] for x in source['held_members']):raise ValueError('Held reference member exposed')
        oldrows=sorted([x for x in pool.values() if x['subset']=='bridge_positive' and old[x['member_id']]>0],key=lambda x:x['member_id'])
        n=len(oldrows);k=len(candidates)
        a=np.array([vector(x) for x in oldrows]+[[-v for v in vector(x)] for x in candidates],dtype=float).T
        lo=np.array([0]*n+[3]*k,dtype=float);hi=np.array([old[x['member_id']] for x in oldrows]+[540]*k,dtype=float)
        objective=np.array([0]*n+[1]*k,dtype=float)
        result=milp(c=objective,integrality=np.ones(n+k),bounds=Bounds(lo,hi),constraints=LinearConstraint(a,0,0),options={'time_limit':60})
        record=dict(seed=seed,solver_status=int(result.status),message=result.message,
            scope='Only remove original bridge exposures and insert new poses; no old member increases. Each new pose >=3 permits all appearances in VM.',
            reference_class_exposures={c:sum(old[mid]*row['class_instances'].get(c,0) for mid,row in pool.items()) for c in CLASSES},
            candidate_vectors={x['member_id']:vector(x) for x in candidates},
            removal_bounds={x['member_id']:old[x['member_id']] for x in oldrows})
        if result.status==0:
            values=np.rint(result.x).astype(int)
            if np.max(np.abs(result.x-values))>1e-6 or np.any(a@values) or np.any(values<lo) or np.any(values>hi):raise ValueError('Integer solution verification failed')
            record.update(status='exact_replacement_counts_feasible_not_training_ready',
                removed={x['member_id']:int(v) for x,v in zip(oldrows,values[:n]) if v},
                inserted={x['member_id']:int(v) for x,v in zip(candidates,values[n:])},
                changed_positions=int(sum(values[n:])),exact_instance_delta={c:0 for c in CLASSES})
        else:record.update(status='restricted_replacement_infeasible' if result.status==2 else 'solver_incomplete',
            broader_design_infeasibility_not_claimed=True)
        paths += [Path(design['source_protocol'])]
        results[str(seed)]=record
        print(seed,record['status'],record.get('changed_positions'),flush=True)
    prior.frozen(OUT/'result.json',dict(results=results,
        status='counts_checked_sequences_not_frozen',training_ready=False,training_started=False,
        objective='Minimize new-pose exposure count subject to at least three per pose; feasibility diagnostic, not final selected sequence.',
        unresolved=['Full source-role dependency audit','Final sequence freeze and actual loaders'],
        inputs={str(p):prior.file_sha256(p) for p in paths}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--solve',action='store_true');args=parser.parse_args()
    if args.solve:run()
    else:print('NO_SOLVE_NO_TRAINING; explicit --solve required')
