"""Fixed viewed-development inference, no training or automatic visual decisions."""
import argparse
import csv
import math
import traceback
from pathlib import Path
from collections import Counter
from scripts.vision.train_whole_image_hold import OUT,ROOT,KEYS,ready,complete,read,verify,frozen,file_sha256
from scripts.vision.run_fixed_budget_diagnosis import predict,paired_truth,score,summary,VARIANTS
from scripts.vision.finalize_exposure_diagnosis import aggregate,policy_checks
from scripts.vision.analyze_visibility_quality_results import paired_change

def evaluate(key,p):
    ep=OUT/'evaluation'/f'{key}.json';cp=OUT/'training'/key/'completion.json';complete(cp,p,key)
    if ep.exists():verify(read(ep));return read(ep)
    root=OUT/'evaluation'/key;root.mkdir(parents=True,exist_ok=True)
    attempts=list(root.glob('attempt-*'))
    if len(attempts)>=3:raise ValueError('Inference attempt budget exhausted')
    attempt=root/f'attempt-{len(attempts)+1:03}';attempt.mkdir()
    try:
        from ultralytics import YOLO
        import yaml
        cell=read(cp);run=Path(cell['weights']).parents[1]
        args=yaml.safe_load((run/'args.yaml').read_text())
        from scripts.vision.order_retention_runtime import overrides
        for k,v in overrides(int(key.split('-')[-1])).items():
            if args.get(k)!=v:raise ValueError(f'Runtime configuration drift: {k}')
        with (run/'results.csv').open() as f:curve=list(csv.DictReader(f))
        if len(curve)!=45 or any(not math.isfinite(float(v)) for r in curve for k,v in r.items() if k.startswith('train/')):raise ValueError('Incomplete loss curve')
        if any(abs(float(v)-.001)>1e-12 for r in curve for k,v in r.items() if k.startswith('lr/')):raise ValueError('LR drift')
        rp=Path(p['evaluation']['paired_review']);np=Path(p['evaluation']['negative_review'])
        review=read(rp);negative=read(np);verify(review);verify(negative)
        paired,inputs=paired_truth(review['frames'])
        if len(paired)!=48 or len(negative['frames'])!=48:raise ValueError('Incomplete fixed development set')
        for row in review['frames']+negative['frames']:
            if row['decision']!='accepted' or file_sha256(row['image_path'])!=row['image_sha256']:raise ValueError('Stale review')
            inputs[row['image_path']]=row['image_sha256']
        model=YOLO(cell['weights']);results=[];neg=[]
        for row,t in paired:results.append(score(row,t,predict(model,row['image_path'],.37),predict(model,row['image_path'],.001)))
        for row in negative['frames']:
            a=predict(model,row['image_path'],.37);b=predict(model,row['image_path'],.001)
            neg.append(dict(view_id=row['view_id'],variant=row['variant'],image_sha256=row['image_sha256'],predictions=a,diagnostic_predictions=b,frame_has_prediction=bool(a)))
        paths=[OUT/'protocol.json',cp,rp,np,Path(__file__),Path(cell['weights']),run/'args.yaml',run/'results.csv']
        for name in ['evaluate_hard_negative_coverage.py','run_fixed_budget_diagnosis.py','finalize_exposure_diagnosis.py']:
            paths.append(ROOT/'scripts/vision'/name)
        inputs.update({str(x):file_sha256(x) for x in paths})
        return frozen(ep,dict(status='complete',cell=key,rows=results,negative_rows=neg,matching_conflicts=sum(bool(r['matching_conflict']) for r in results),
            summary={v:summary([r for r in results if r['variant']==v]) for v in VARIANTS},
            negative_summary=dict(frame_false_positive_rate=sum(r['frame_has_prediction'] for r in neg)/48,unmatched_predictions=sum(len(r['predictions']) for r in neg)),
            runtime_controls_verified=True,inputs=inputs))
    except BaseException:
        frozen(attempt/'failure.json',dict(status='failed',error=traceback.format_exc(),child_processes_started=0));raise

def finish(p):
    dest=OUT/'evaluation/summary.json'
    if dest.exists():verify(read(dest));return read(dest)
    records={k:read(OUT/'evaluation'/f'{k}.json') for k in KEYS}
    for r in records.values():verify(r)
    from scripts.vision.run_matched_appearance_training import validate_evaluation
    for k,r in records.items():validate_evaluation(r,k)
    from scripts.vision.exposure_order_retention import PRIOR,baseline_verify
    refpaths=[PRIOR/f'evaluation-retained_reference-450-{s}.json' for s in (7,17,27)]
    refs=[read(x) for x in refpaths]
    for r in refs:verify(r)
    hp=Path(p['evaluation']['historical_reference']);hist=read(hp);verify(hist)
    groups={a:aggregate([records[f'{a}-450-{s}'] for s in (7,17,27)]) for a in ('reference','hold')}
    gates={a:policy_checks(g,aggregate(refs),hist['historical_A'],p) for a,g in groups.items()}
    paths=[OUT/'evaluation'/f'{k}.json' for k in KEYS]+refpaths+[hp,Path(__file__),OUT/'protocol.json']
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Pinned baseline failure')
    return frozen(dest,dict(status='numerical_evaluation_complete_visual_review_not_performed',aggregate=groups,policy_results=gates,
        paired_reference_to_hold={str(s):paired_change(records[f'reference-450-{s}']['rows'],records[f'hold-450-{s}']['rows']) for s in (7,17,27)},
        miss_reasons={k:dict(Counter(f"{r['variant']}:{m['class_name']}:{m['reason']}" for r in v['rows'] for m in r['misses'])) for k,v in records.items()},
        selected_candidate=None,baseline=baseline,inputs={str(x):file_sha256(x) for x in paths}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--evaluate',action='store_true');a=ap.parse_args()
    if a.evaluate:
        p=ready()
        for k in KEYS:r=evaluate(k,p);print('EVALUATED',k,r['negative_summary'],flush=True)
        print('SUMMARY',finish(p)['status'],flush=True)
    else:print('PREFLIGHT_ONLY_NO_INFERENCE_NO_TRAINING')
