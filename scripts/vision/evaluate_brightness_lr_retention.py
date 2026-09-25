"""Explicit fixed development and training-fit inference for the LR-only trial."""
import argparse
import traceback
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from scripts.vision.brightness_lr_retention import OUT, SOURCE, ROOT, KEYS, LR, ready, complete, runtime_check, read, verify, frozen, file_sha256
from scripts.vision.run_fixed_budget_diagnosis import predict, paired_truth, score, summary, VARIANTS
from scripts.vision.finalize_exposure_diagnosis import aggregate, policy_checks
from scripts.vision.analyze_visibility_quality_results import paired_change
from scripts.vision.check_lineage_training_fit import OUT as FIT, forbidden, truth_for, scoring


def evaluate(key, p):
    complete(key, p); ep = OUT/'evaluation'/f'{key}.json'; cp = OUT/'training'/key/'completion.json'
    if ep.exists(): verify(read(ep)); return read(ep)
    root = OUT/'evaluation'/key; root.mkdir(parents=True, exist_ok=True)
    n = len(list(root.glob('attempt-*'))) + 1
    if n > 3: raise ValueError('Inference attempt budget exhausted')
    attempt = root/f'attempt-{n:03}'; attempt.mkdir()
    try:
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        cell = read(cp); runtime_check(cell, int(key.split('-')[-1]), LR)
        rp, np = Path(p['evaluation']['paired_review']), Path(p['evaluation']['negative_review'])
        review, negative = read(rp), read(np); verify(review); verify(negative)
        paired, inputs = paired_truth(review['frames'])
        if len(paired) != 48 or len(negative['frames']) != 48: raise ValueError('Incomplete development set')
        for row in review['frames'] + negative['frames']:
            if row['decision'] != 'accepted' or file_sha256(row['image_path']) != row['image_sha256']: raise ValueError('Stale review')
            inputs[row['image_path']] = row['image_sha256']
        results, neg = [], []
        with ExitStack() as stack:
            for obj, name in ((torch.optim.Optimizer, '__init__'), (torch.Tensor, 'backward'), (YOLO, 'train'), (YOLO, 'val')):
                stack.enter_context(patch.object(obj, name, forbidden))
            model = YOLO(cell['weights'])
            with torch.inference_mode():
                for row, truth in paired:
                    results.append(score(row, truth, predict(model,row['image_path'],.37), predict(model,row['image_path'],.001)))
                for row in negative['frames']:
                    a, b = predict(model,row['image_path'],.37), predict(model,row['image_path'],.001)
                    neg.append(dict(view_id=row['view_id'],variant=row['variant'],image_sha256=row['image_sha256'],predictions=a,diagnostic_predictions=b,frame_has_prediction=bool(a)))
        paths = [OUT/'protocol.json', cp, OUT/'training'/key/'brightness-receipt.json', rp, np, Path(__file__), Path(cell['weights']),
                 ROOT/'scripts/vision/run_fixed_budget_diagnosis.py', ROOT/'scripts/vision/evaluate_hard_negative_coverage.py',
                 ROOT/'scripts/vision/evaluate_exposure_diagnosis.py',ROOT/'scripts/vision/exposure_metrics.py']
        inputs.update({str(x):file_sha256(x) for x in paths})
        return frozen(ep, dict(status='complete',cell=key,rows=results,negative_rows=neg,
            matching_conflicts=sum(bool(r['matching_conflict']) for r in results),
            summary={v:summary([r for r in results if r['variant']==v]) for v in VARIANTS},
            negative_summary=dict(frame_false_positive_rate=sum(r['frame_has_prediction'] for r in neg)/48,unmatched_predictions=sum(len(r['predictions']) for r in neg)),
            runtime_controls_verified=True, optimizer_created=False, backward_executed=False, validation_run=False,inputs=inputs))
    except BaseException:
        frozen(attempt/'failure.json',dict(error=traceback.format_exc(),child_processes_started=0)); raise


def fitting(key, p):
    complete(key, p); path = OUT/'fit'/f'{key}.json'
    if path.exists(): verify(read(path)); return read(path)
    root=OUT/'fit'/key; root.mkdir(parents=True,exist_ok=True)
    n=len(list(root.glob('attempt-*')))+1
    if n>3: raise ValueError('Fit inference attempt budget exhausted')
    attempt=root/f'attempt-{n:03}';attempt.mkdir()
    try:
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4); f=read(FIT/'protocol.json');verify(f)
        c=read(OUT/'training'/key/'completion.json');verify(c);rows=[];counts=Counter(read(c['exposure_path'])['draws'])
        with ExitStack() as stack:
            for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(YOLO,'train'),(YOLO,'val')):stack.enter_context(patch.object(obj,name,forbidden))
            m=YOLO(c['weights'])
            with torch.inference_mode():
                for r in f['rows']:
                    rows.append(dict(member_id=r['member_id'],exposures=counts[r['member_id']],scoring=scoring(truth_for(r),predict(m,r['image_path'],.37),predict(m,r['image_path'],.001))))
        return frozen(path,dict(status='complete_training_fit_diagnostic_only',cell=key,rows=rows,optimizer_created=False,backward_executed=False,validation_run=False,
            inputs={str(x):file_sha256(x) for x in [FIT/'protocol.json',OUT/'protocol.json',OUT/'training'/key/'brightness-receipt.json',Path(c['weights']),Path(__file__),ROOT/'scripts/vision/structure_fit.py',ROOT/'scripts/vision/evaluate_exposure_diagnosis.py',ROOT/'scripts/vision/exposure_metrics.py']}))
    except BaseException:
        frozen(attempt/'failure.json',dict(error=traceback.format_exc(),child_processes_started=0));raise


def finish(p):
    path=OUT/'evaluation/summary.json'
    if path.exists():verify(read(path));return read(path)
    records=[read(OUT/'evaluation'/f'{k}.json') for k in KEYS]
    baseline=[read(p['baselines'][str(s)]['evaluation']) for s in (7,17,27)]
    context=[read(p['unaugmented_context'][str(s)]['evaluation']) for s in (7,17,27)]
    for r in records+baseline+context:verify(r)
    if any(r['matching_conflicts'] for r in records):raise ValueError('Matching conflicts block comparison')
    from scripts.vision.exposure_order_retention import PRIOR
    refs=[PRIOR/f'evaluation-retained_reference-450-{s}.json' for s in (7,17,27)]; hist=Path(p['evaluation']['historical_reference'])
    retained=[read(x) for x in refs];h=read(hist)
    for r in retained+[h]:verify(r)
    checks=policy_checks(aggregate(records),aggregate(retained),h['historical_A'],p)
    paths=[OUT/'evaluation'/f'{k}.json' for k in KEYS]+[OUT/'fit'/f'{k}.json' for k in KEYS]+refs+[hist,OUT/'protocol.json',Path(__file__)]
    paths += [Path(p[field][str(s)]['evaluation']) for field in ('baselines','unaugmented_context') for s in (7,17,27)]
    return frozen(path,dict(status='numerical_evaluation_complete_visual_review_pending',
        aggregate=dict(lr0005=aggregate(records),lr001=aggregate(baseline),noaug_context=aggregate(context)),
        policy_results=checks, paired_changes={str(s):paired_change(a['rows'],b['rows']) for s,a,b in zip((7,17,27),baseline,records)},
        selected_candidate=None, inputs={str(x):file_sha256(x) for x in paths}))


def evidence():
    from scripts.vision import inspect_brightness_errors as errors, inspect_brightness_fit as fits
    # Existing evidence renderers only; no inherited visual decisions.
    errors.BASE=OUT;errors.OUT=OUT/'error-review';fits.OUT=OUT
    for path, fn in ((OUT/'error-review/evidence.json',errors.main),(OUT/'fit-review/evidence.json',fits.main)):
        if path.exists():verify(read(path))
        else:fn()
    receipt=OUT/'evidence-generation.json'
    if not receipt.exists():
        paths=[Path(__file__),Path(errors.__file__),Path(fits.__file__),OUT/'error-review/evidence.json',OUT/'fit-review/evidence.json']
        frozen(receipt,dict(status='evidence_generated_no_decisions',inputs={str(x):file_sha256(x) for x in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--evaluate',action='store_true');ap.add_argument('--cell',choices=KEYS);ap.add_argument('--evidence',action='store_true');a=ap.parse_args()
    if a.evaluate:
        p=ready()
        for key in ([a.cell] if a.cell else KEYS):
            r=evaluate(key,p);fitting(key,p);print('EVALUATED_AND_FIT_LR0005',key,r['negative_summary'],flush=True)
        if not a.cell:print('NUMERICAL_GATES',finish(p)['policy_results']['passed'],flush=True)
    elif a.evidence:evidence()
    else:print('READ_ONLY_NO_INFERENCE_NO_TRAINING')
