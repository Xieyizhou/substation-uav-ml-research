"""Fixed viewed-development evaluation and unchanged training-fit guard subset."""
import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from scripts.vision.train_brightness_transfer import OUT,SOURCE,ROOT,KEYS,pretrain,complete,read,verify,frozen,file_sha256
from scripts.vision import evaluate_whole_image_hold as engine
from scripts.vision.finalize_exposure_diagnosis import aggregate,policy_checks
from scripts.vision.analyze_visibility_quality_results import paired_change
from scripts.vision.check_lineage_training_fit import OUT as FIT,forbidden,truth_for,scoring,predict

def fitting(key,p):
    path=OUT/'fit'/f'{key}.json'
    if path.exists():verify(read(path));return
    (OUT/'fit').mkdir(exist_ok=True);f=read(FIT/'protocol.json');verify(f);c=read(OUT/'training'/key/'completion.json');verify(c);complete(key,p)
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4);rows=[];counts=Counter(read(c['exposure_path'])['draws'])
    with ExitStack() as stack:
        for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(YOLO,'train'),(YOLO,'val')):stack.enter_context(patch.object(obj,name,forbidden))
        m=YOLO(c['weights'])
        with torch.inference_mode():
            for r in f['rows']:rows.append(dict(member_id=r['member_id'],exposures=counts[r['member_id']],scoring=scoring(truth_for(r),predict(m,r['image_path'],.37),predict(m,r['image_path'],.001))))
    frozen(path,dict(status='complete_training_fit_diagnostic_only',cell=key,rows=rows,optimizer_created=False,backward_executed=False,validation_run=False,
        inputs={str(x):file_sha256(x) for x in [FIT/'protocol.json',OUT/'protocol.json',OUT/'training'/key/'brightness-receipt.json',Path(c['weights']),Path(__file__)]}))

def run(only=None):
    p=pretrain()
    for key in ([only] if only else KEYS):
        complete(key,p);previous=engine.OUT;engine.OUT=OUT
        try:r=engine.evaluate(key,p)
        finally:engine.OUT=previous
        if r['matching_conflicts']:raise ValueError('Matching conflict')
        fitting(key,p);print('EVALUATED_AND_FIT',key,r['negative_summary'],flush=True)
    if only:return
    records={k:read(OUT/'evaluation'/f'{k}.json') for k in KEYS};baseline={str(s):read(p['baselines'][str(s)]['evaluation']) for s in (7,17,27)}
    for r in list(records.values())+list(baseline.values()):verify(r)
    from scripts.vision.exposure_order_retention import PRIOR
    refs=[PRIOR/f'evaluation-retained_reference-450-{s}.json' for s in (7,17,27)];hist=Path(p['evaluation']['historical_reference']);h=read(hist);verify(h)
    retained=[read(x) for x in refs]
    for r in retained:verify(r)
    groups=dict(noaug=aggregate(list(baseline.values())),brightness=aggregate([records[k] for k in KEYS]))
    checks=policy_checks(groups['brightness'],aggregate(retained),h['historical_A'],p)
    paths=[OUT/'evaluation'/f'{k}.json' for k in KEYS]+[OUT/'fit'/f'{k}.json' for k in KEYS]+[Path(p['baselines'][str(s)]['evaluation']) for s in (7,17,27)]+refs+[hist,OUT/'protocol.json',Path(__file__)]
    path=OUT/'evaluation/summary.json'
    if path.exists():verify(read(path));return
    frozen(path,dict(status='numerical_evaluation_complete_visual_review_pending',aggregate=groups,policy_results=checks,paired_changes={str(s):paired_change(baseline[str(s)]['rows'],records[f'brightness-450-{s}']['rows']) for s in (7,17,27)},selected_candidate=None,inputs={str(x):file_sha256(x) for x in paths}))
    print('NUMERICAL_GATES',checks['passed'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--evaluate',action='store_true');ap.add_argument('--cell',choices=KEYS);a=ap.parse_args()
    if a.evaluate:run(a.cell)
    else:print('READ_ONLY_NO_INFERENCE_NO_TRAINING')
