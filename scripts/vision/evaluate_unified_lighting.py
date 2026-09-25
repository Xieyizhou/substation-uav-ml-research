"""Explicit paired lighting development evaluation; no training or auto approval."""
import argparse
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from scripts.vision.train_unified_lighting import OUT, KEYS, ready, complete, prior
from scripts.vision.brightness_lr_retention import runtime_check
from scripts.vision.run_fixed_budget_diagnosis import predict, paired_truth, score, summary, VARIANTS
from scripts.vision.finalize_exposure_diagnosis import aggregate, policy_checks, NAMES
from scripts.vision.analyze_visibility_quality_results import paired_change
from scripts.vision.exposure_order_retention import PRIOR, baseline_verify

def forbidden(*args, **kwargs):
    raise AssertionError('Training forbidden during development evaluation')

def validate_record(r, key):
    prior.verify(r)
    if r['status'] != 'complete' or r['cell'] != key or len(r['rows']) != 48 or len(r['negative_rows']) != 48:
        raise ValueError('Incomplete evaluation')
    if len({(x['view_id'], x['variant']) for x in r['negative_rows']}) != 48:
        raise ValueError('Duplicate negative identities')
    if len({(x['pair_id'], x['variant']) for x in r['rows']}) != 48:
        raise ValueError('Duplicate paired identities')

def evaluate(key, p):
    complete(key, p)
    ep=OUT/'evaluation'/f'{key}.json'; cp=OUT/'training'/key/'completion.json'
    if ep.exists():
        r=prior.read(ep); validate_record(r,key); return r
    root=OUT/'evaluation'/key; root.mkdir(parents=True,exist_ok=True)
    n=len(list(root.glob('attempt-*')))+1
    if n>3: raise ValueError('Inference attempt budget exhausted')
    attempt=root/f'attempt-{n:03}'; attempt.mkdir()
    try:
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        cell=prior.read(cp); runtime_check(cell,int(key.split('-')[-1]),.0005)
        rp,np=map(Path,(p['evaluation']['paired_review'],p['evaluation']['negative_review']))
        review,negative=prior.read(rp),prior.read(np)
        for r in (review,negative): prior.verify(r)
        paired,inputs=paired_truth(review['frames'])
        if len(paired)!=48 or len(negative['frames'])!=48: raise ValueError('Incomplete fixed development data')
        for row in review['frames']+negative['frames']:
            if row['decision']!='accepted' or prior.file_sha256(row['image_path'])!=row['image_sha256']: raise ValueError('Stale review')
            inputs[row['image_path']]=row['image_sha256']
        results=[]; neg=[]
        with ExitStack() as stack:
            for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):
                stack.enter_context(patch.object(obj,name,forbidden))
            model=YOLO(cell['weights'])
            with torch.inference_mode():
                for row,truth in paired:
                    results.append(score(row,truth,predict(model,row['image_path'],.37),predict(model,row['image_path'],.001)))
                for row in negative['frames']:
                    a,b=predict(model,row['image_path'],.37),predict(model,row['image_path'],.001)
                    neg.append(dict(view_id=row['view_id'],variant=row['variant'],image_sha256=row['image_sha256'],predictions=a,diagnostic_predictions=b,frame_has_prediction=bool(a)))
        paths=[OUT/'protocol.json',cp,OUT/'training'/key/'tensor-completion.json',rp,np,Path(__file__).resolve(),Path(cell['weights'])]
        paths += [prior.ROOT/'scripts/vision'/name for name in ('run_fixed_budget_diagnosis.py','run_order_diagnosis.py','evaluate_hard_negative_coverage.py','evaluate_exposure_diagnosis.py','exposure_metrics.py')]
        inputs.update({str(x):prior.file_sha256(x) for x in paths})
        r=prior.frozen(ep,dict(status='complete',cell=key,rows=results,negative_rows=neg,
            matching_conflicts=sum(bool(r['matching_conflict']) for r in results),
            summary={v:summary([r for r in results if r['variant']==v]) for v in VARIANTS},
            negative_summary=dict(frame_false_positive_rate=sum(r['frame_has_prediction'] for r in neg)/48,unmatched_predictions=sum(len(r['predictions']) for r in neg)),
            optimizer_created=False,backward_executed=False,validation_run=False,runtime_controls_verified=True,inputs=inputs))
        validate_record(r,key); return r
    except BaseException:
        prior.frozen(attempt/'failure.json',dict(status='failed',error=traceback.format_exc(),child_processes_started=0)); raise

def direct_checks(a,b):
    checks=[]
    for v in ('original','lighting'):
        for name in (None,*NAMES):
            x=a[v] if name is None else a[v]['per_class'][name]
            y=b[v] if name is None else b[v]['per_class'][name]
            delta=x['instance_recall']['mean']-y['instance_recall']['mean']
            checks.append(dict(variant=v,category=name or 'all',delta=delta,passed=delta>=-.05-1e-12))
    return dict(passed=all(x['passed'] for x in checks),checks=checks)

def finish(p):
    dest=OUT/'evaluation/summary.json'
    if dest.exists(): prior.verify(prior.read(dest)); return prior.read(dest)
    paths=[OUT/'evaluation'/f'{k}.json' for k in KEYS]
    records={k:prior.read(x) for k,x in zip(KEYS,paths)}
    for k,r in records.items(): validate_record(r,k)
    refs=[PRIOR/f'evaluation-retained_reference-450-{s}.json' for s in (7,17,27)]
    hp=Path(p['evaluation']['historical_reference']); historical=prior.read(hp); reference=[prior.read(x) for x in refs]
    for r in reference+[historical]: prior.verify(r)
    groups={a:aggregate([records[f'{a}-{s}'] for s in (7,17,27)]) for a in ('R-clean','L-physical')}
    checks={a:policy_checks(g,aggregate(reference),historical['historical_A'],p) for a,g in groups.items()}
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40: raise ValueError('Pinned baseline failed')
    paths += refs+[hp,OUT/'protocol.json',Path(__file__).resolve()]
    return prior.frozen(dest,dict(status='numerical_complete_explicit_error_review_pending',aggregate=groups,policy_results=checks,
        direct_retention=direct_checks(groups['L-physical'],groups['R-clean']),
        paired_changes={str(s):paired_change(records[f'R-clean-{s}']['rows'],records[f'L-physical-{s}']['rows']) for s in (7,17,27)},
        matching_conflicts=sum(r['matching_conflicts'] for r in records.values()),selected_candidate=None,baseline=baseline,
        inputs={str(x):prior.file_sha256(x) for x in paths}))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--evaluate',action='store_true'); a=ap.parse_args()
    p=ready()
    if a.evaluate:
        for k in KEYS: print('EVALUATED',k,evaluate(k,p)['negative_summary'],flush=True)
        print('SUMMARY',finish(p)['status'],flush=True)
    else: print('READY_ONLY_NO_INFERENCE_NO_TRAINING')
