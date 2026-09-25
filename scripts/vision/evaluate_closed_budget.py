"""Explicit endpoint development inference; never trains or approves reviews."""
import argparse
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from scripts.vision.train_closed_budget import OUT, KEYS, complete, ready, prior
def contract(key):
    if key not in KEYS:raise ValueError('Unknown budget cell')
    p,_=ready();return p,p,None
from scripts.vision.benchmark_closed_cpu import OUT as BENCH,CONFIGS
from scripts.vision import train_compensated_material as low_dose
from scripts.vision.train_frozen_multiscale import OUT as REFERENCE_OUT, complete as reference_complete, contract as reference_contract
from scripts.vision.evaluate_unified_lighting import forbidden, validate_record, direct_checks
from scripts.vision.run_fixed_budget_diagnosis import predict, paired_truth, score, summary, VARIANTS
from scripts.vision.finalize_exposure_diagnosis import aggregate, policy_checks
from scripts.vision.analyze_visibility_quality_results import paired_change
from scripts.vision.exposure_order_retention import PRIOR, baseline_verify

def verify_binding(r,key):
    cp=OUT/'training'/key/'completion.json';c=prior.read(cp)
    for path,sha in ((c['weights'],c['weights_sha256']),(str(OUT/'design.json'),prior.file_sha256(OUT/'design.json'))):
        if r['inputs'].get(path)!=sha:raise ValueError('Inference weight/protocol binding mismatch')
    if any(r.get(k) is not False for k in ('optimizer_created','backward_executed','validation_run')):raise ValueError('Forbidden runtime operation')


def evaluate(key):
    complete(key)
    _, p, _ = contract(key)
    dest = OUT/'evaluation'/f'{key}.json'
    if dest.exists():
        r = prior.read(dest); validate_record(r, key); verify_binding(r,key); return r
    root = OUT/'evaluation'/key
    root.mkdir(parents=True, exist_ok=True)
    n = len(list(root.glob('attempt-*'))) + 1
    if n > 3: raise ValueError('Inference attempt budget exhausted')
    attempt = root/f'attempt-{n:03}'
    attempt.mkdir()
    try:
        import torch
        from ultralytics import YOLO
        bench=prior.read(BENCH/'completion.json');prior.verify(bench)
        threads=CONFIGS[bench['selected']][0]
        torch.set_num_threads(threads)
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
                predict(model,review['frames'][0]['image_path'],.37)
                torch.set_num_threads(threads)
                pilot=prior.read(BENCH/'protocol.json')
                if key in pilot['keys']:
                    expected=prior.read(BENCH/'single4'/(key+'.json'));prior.verify(expected)
                    actual=[dict(image_sha256=row['image_sha256'],confidence=c,predictions=predict(model,row['image_path'],c)) for row in pilot['sample'] for c in (.37,.001)]
                    if actual!=expected['outputs'][0]:raise ValueError('Full-run pilot predictions changed')
                for row, truth in paired:
                    results.append(score(row, truth, predict(model,row['image_path'],.37), predict(model,row['image_path'],.001)))
                    if torch.get_num_threads()!=threads:raise ValueError('Effective threads changed')
                for row in negative['frames']:
                    a, b = predict(model,row['image_path'],.37), predict(model,row['image_path'],.001)
                    neg.append(dict(view_id=row['view_id'],variant=row['variant'],image_sha256=row['image_sha256'],predictions=a,diagnostic_predictions=b,frame_has_prediction=bool(a)))
        paths = [OUT/'design.json', cp, rp, np, Path(__file__).resolve(), Path(cell['weights']),BENCH/'completion.json']
        paths += [prior.ROOT/'scripts/vision'/name for name in ('run_fixed_budget_diagnosis.py','run_order_diagnosis.py','evaluate_hard_negative_coverage.py','evaluate_exposure_diagnosis.py','exposure_metrics.py')]
        inputs.update({str(x):prior.file_sha256(x) for x in paths})
        r = prior.frozen(dest,dict(status='complete',cell=key,rows=results,negative_rows=neg,
            matching_conflicts=sum(bool(x['matching_conflict']) for x in results),
            summary={v:summary([x for x in results if x['variant']==v]) for v in VARIANTS},
            negative_summary=dict(frame_false_positive_rate=sum(x['frame_has_prediction'] for x in neg)/48,unmatched_predictions=sum(len(x['predictions']) for x in neg)),
            optimizer_created=False,backward_executed=False,validation_run=False,effective_cpu_threads=threads,benchmark_selection=bench['selected'],inputs=inputs))
        validate_record(r,key); verify_binding(r,key)
        return r
    except BaseException:
        prior.frozen(attempt/'failure.json',dict(status='failed',error=traceback.format_exc(),child_processes_started=0))
        raise


from scripts.vision import train_material_retention_coverage as direct_train

def finish():
    dest=OUT/'evaluation/summary.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    _,p,_=contract(KEYS[0]);paths=[];records={};direct={}
    for key in KEYS:
        complete(key);ep=OUT/'evaluation'/f'{key}.json';r=prior.read(ep);validate_record(r,key);verify_binding(r,key);records[key]=r;paths.append(ep)
    for seed in (7,17,27):
        oldkey=f'T-{seed}';direct_train.complete(oldkey);ep=direct_train.OUT/'evaluation'/f'{oldkey}.json'
        r=prior.read(ep);validate_record(r,oldkey)
        c=prior.read(direct_train.OUT/'training'/oldkey/'completion.json')
        if r['inputs'].get(c['weights'])!=c['weights_sha256']:raise ValueError('Reference weight changed')
        direct[str(seed)]=r;paths.append(ep)
    refs=[PRIOR/f'evaluation-retained_reference-450-{s}.json' for s in (7,17,27)]
    hp=Path(p['evaluation']['historical_reference']);history=prior.read(hp)
    reference=[prior.read(x) for x in refs]
    for r in reference+[history]:prior.verify(r)
    _,policy,_=reference_contract('fixed-7')
    decision=p['acceptance_policy']
    if decision['acceptance_policy']!=policy['acceptance_policy'] or decision['retention']!=policy['retention']:raise ValueError('Policy drift')
    groups={family:aggregate([records[f'{family}-{seed}'] for seed in (7,17,27)]) for family in ('B450','B900')}
    old=aggregate(list(direct.values()))
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline failure')
    paths+=refs+[hp,OUT/'design.json',Path(__file__).resolve()]
    return prior.frozen(dest,dict(status='numerical_complete_explicit_error_review_pending',
        aggregate=groups,direct_reference_aggregate=old,
        policy_results={f:policy_checks(g,aggregate(reference),history['historical_A'],policy) for f,g in groups.items()},
        direct_retention={f:direct_checks(g,old) for f,g in groups.items()},
        paired_changes={str(seed):paired_change(records[f'B450-{seed}']['rows'],records[f'B900-{seed}']['rows']) for seed in (7,17,27)},
        matching_conflicts=sum(r['matching_conflicts'] for r in records.values()),baseline=baseline,selected_candidate=None,
        comparison='B450 vs B900: independent initializations; identical first 450 steps; full sequence and brightness repeated for 900. Historical T is background only. Budget doubles every subset, not only materials.',
        inputs={str(x):prior.file_sha256(x) for x in paths}))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evaluate',action='store_true');ap.add_argument('--cell',choices=KEYS);a=ap.parse_args()
    if a.cell and not a.evaluate:ap.error('--cell requires --evaluate')
    if a.cell:print('EVALUATED',a.cell,evaluate(a.cell)['negative_summary']);return
    for key in KEYS:complete(key)
    if a.evaluate:
        for key in KEYS:print('EVALUATED',key,evaluate(key)['negative_summary'],flush=True)
        print('SUMMARY',finish()['status'],flush=True)
    else:print('VERIFIED_NO_INFERENCE_NO_TRAINING')

if __name__=='__main__':main()
