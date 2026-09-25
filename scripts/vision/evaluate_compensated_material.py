"""Explicit endpoint development inference; never trains or approves reviews."""
import argparse
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from scripts.vision.train_compensated_material import OUT, KEYS, complete, contract, prior
from scripts.vision.train_frozen_multiscale import OUT as REFERENCE_OUT, complete as reference_complete, contract as reference_contract
from scripts.vision.evaluate_unified_lighting import forbidden, validate_record, direct_checks
from scripts.vision.run_fixed_budget_diagnosis import predict, paired_truth, score, summary, VARIANTS
from scripts.vision.finalize_exposure_diagnosis import aggregate, policy_checks
from scripts.vision.analyze_visibility_quality_results import paired_change
from scripts.vision.exposure_order_retention import PRIOR, baseline_verify

def evaluate(key):
    complete(key)
    _, p, _ = contract(key)
    dest = OUT/'evaluation'/f'{key}.json'
    if dest.exists():
        r = prior.read(dest); validate_record(r, key); return r
    root = OUT/'evaluation'/key
    root.mkdir(parents=True, exist_ok=True)
    n = len(list(root.glob('attempt-*'))) + 1
    if n > 3: raise ValueError('Inference attempt budget exhausted')
    attempt = root/f'attempt-{n:03}'
    attempt.mkdir()
    try:
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        cp = OUT/'training'/key/'completion.json'
        cell = prior.read(cp)
        rp, np = map(Path, (p['evaluation']['paired_review'], p['evaluation']['negative_review']))
        review, negative = prior.read(rp), prior.read(np)
        for r in (review, negative): prior.verify(r)
        paired, inputs = paired_truth(review['frames'])
        if len(paired) != 48 or len(negative['frames']) != 48: raise ValueError('Incomplete development data')
        for row in review['frames'] + negative['frames']:
            if row['decision'] != 'accepted' or prior.file_sha256(row['image_path']) != row['image_sha256']: raise ValueError('Stale review')
            inputs[row['image_path']] = row['image_sha256']
        results, neg = [], []
        with ExitStack() as stack:
            for obj, name in ((torch.optim.Optimizer,'__init__'), (torch.Tensor,'backward'), (torch.autograd,'backward'), (YOLO,'train'), (YOLO,'val')):
                stack.enter_context(patch.object(obj, name, forbidden))
            model = YOLO(cell['weights'])
            with torch.inference_mode():
                for row, truth in paired:
                    results.append(score(row, truth, predict(model,row['image_path'],.37), predict(model,row['image_path'],.001)))
                for row in negative['frames']:
                    a, b = predict(model,row['image_path'],.37), predict(model,row['image_path'],.001)
                    neg.append(dict(view_id=row['view_id'],variant=row['variant'],image_sha256=row['image_sha256'],predictions=a,diagnostic_predictions=b,frame_has_prediction=bool(a)))
        paths = [OUT/'protocol.json', cp, rp, np, Path(__file__).resolve(), Path(cell['weights'])]
        paths += [prior.ROOT/'scripts/vision'/name for name in ('run_fixed_budget_diagnosis.py','run_order_diagnosis.py','evaluate_hard_negative_coverage.py','evaluate_exposure_diagnosis.py','exposure_metrics.py')]
        inputs.update({str(x):prior.file_sha256(x) for x in paths})
        r = prior.frozen(dest,dict(status='complete',cell=key,rows=results,negative_rows=neg,
            matching_conflicts=sum(bool(x['matching_conflict']) for x in results),
            summary={v:summary([x for x in results if x['variant']==v]) for v in VARIANTS},
            negative_summary=dict(frame_false_positive_rate=sum(x['frame_has_prediction'] for x in neg)/48,unmatched_predictions=sum(len(x['predictions']) for x in neg)),
            optimizer_created=False,backward_executed=False,validation_run=False,inputs=inputs))
        validate_record(r,key)
        return r
    except BaseException:
        prior.frozen(attempt/'failure.json',dict(status='failed',error=traceback.format_exc(),child_processes_started=0))
        raise

def finish():
    _, p, _ = contract(KEYS[0])
    paths = [OUT/'evaluation'/f'{k}.json' for k in KEYS]
    records = {k:prior.read(x) for k,x in zip(KEYS,paths)}
    for k,r in records.items(): complete(k); validate_record(r,k)
    refs = [PRIOR/f'evaluation-retained_reference-450-{s}.json' for s in (7,17,27)]
    hp = Path(p['evaluation']['historical_reference'])
    historical = prior.read(hp); reference = [prior.read(x) for x in refs]
    for r in reference + [historical]: prior.verify(r)
    _, policy, _ = reference_contract('fixed-7')
    direct_records={}
    for seed in (7,17,27):
        key=f'fixed-{seed}';reference_complete(key);ep=REFERENCE_OUT/'evaluation'/f'{key}.json'
        rec=prior.read(ep);validate_record(rec,key)
        training=prior.read(REFERENCE_OUT/'training'/key/'completion.json')
        if rec['inputs'].get(training['weights'])!=training['weights_sha256']:raise ValueError('Direct reference weight mismatch')
        direct_records[str(seed)]=rec;paths.append(ep)
    paths.append(Path(reference_contract('fixed-7')[0]['source_protocol']))
    groups = {a:aggregate([records[f'{a}-{s}'] for s in (7,17,27)]) for a in ('V','VM')}
    baseline = baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified'] != 40: raise ValueError('Baseline failure')
    paths += refs + [hp,OUT/'protocol.json',Path(__file__).resolve()]
    return prior.frozen(OUT/'evaluation/summary.json',dict(status='numerical_complete_explicit_error_review_pending',aggregate=groups,
        policy_results={a:policy_checks(g,aggregate(reference),historical['historical_A'],policy) for a,g in groups.items()},
        direct_retention={a:direct_checks(g,aggregate(list(direct_records.values()))) for a,g in groups.items()},
        V_VM_retention=direct_checks(groups['VM'],groups['V']),
        direct_reference_aggregate=aggregate(list(direct_records.values())),
        direct_paired_changes={a:{s:paired_change(direct_records[s]['rows'],records[f'{a}-{s}']['rows']) for s in direct_records} for a in groups},
        paired_changes={str(s):paired_change(records[f'V-{s}']['rows'],records[f'VM-{s}']['rows']) for s in (7,17,27)},
        matching_conflicts=sum(r['matching_conflicts'] for r in records.values()),selected_candidate=None,baseline=baseline,
        inputs={str(x):prior.file_sha256(x) for x in paths}))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--evaluate',action='store_true'); a=ap.parse_args()
    for k in KEYS: complete(k)
    if a.evaluate:
        for k in KEYS: print('EVALUATED',k,evaluate(k)['negative_summary'],flush=True)
        print('SUMMARY',finish()['status'],flush=True)
    else: print('VERIFIED_NO_INFERENCE_NO_TRAINING')

if __name__=='__main__': main()
