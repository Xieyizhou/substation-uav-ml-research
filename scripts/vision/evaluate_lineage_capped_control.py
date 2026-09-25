"""Fixed formal/diagnostic evaluation of three controls; numerical summary only."""
import argparse
from pathlib import Path
from scripts.vision.lineage_capped_control import OUT,RUN,ROOT,KEYS,ready,read,verify,frozen,file_sha256
from scripts.vision.train_lineage_capped_v2 import completed
from scripts.vision import evaluate_whole_image_hold as engine
from scripts.vision.finalize_exposure_diagnosis import aggregate,policy_checks
from scripts.vision.analyze_visibility_quality_results import paired_change

def run():
    p=ready();records={};paths=[OUT/'protocol.json',Path(__file__),Path(engine.__file__)]
    for key in KEYS:
        completed(key,p)
        previous=engine.OUT;engine.OUT=OUT
        try:r=engine.evaluate(key,p)
        finally:engine.OUT=previous
        if r['matching_conflicts']:raise ValueError('Unresolved matching conflict')
        records[key]=r;paths.extend([OUT/'evaluation'/f'{key}.json',OUT/'training'/key/'adapter-binding.json'])
        print('EVALUATED',key,r['negative_summary'],flush=True)
    for arm in ('reference','hold'):
        for seed in (7,17,27):
            key=f'{arm}-450-{seed}';path=RUN/'evaluation'/f'{key}.json';r=read(path);verify(r);records[key]=r;paths.append(path)
    from scripts.vision.exposure_order_retention import PRIOR
    refs=[]
    for s in (7,17,27):
        path=PRIOR/f'evaluation-retained_reference-450-{s}.json';r=read(path);verify(r);refs.append(r);paths.append(path)
    hp=Path(p['evaluation']['historical_reference']);hist=read(hp);verify(hist);paths.append(hp)
    groups={arm:aggregate([records[f'{arm}-450-{s}'] for s in (7,17,27)]) for arm in ('reference','hold','lineage-capped')}
    checks=policy_checks(groups['lineage-capped'],aggregate(refs),hist['historical_A'],p)
    comparisons={arm:{str(s):paired_change(records[f'{arm}-450-{s}']['rows'],records[f'lineage-capped-450-{s}']['rows']) for s in (7,17,27)} for arm in ('reference','hold')}
    dest=OUT/'evaluation/summary.json'
    if dest.exists():verify(read(dest));return
    frozen(dest,dict(status='numerical_evaluation_complete_visual_review_pending',aggregate=groups,policy_results=checks,
        paired_comparisons=comparisons,selected_candidate=None,inputs={str(x):file_sha256(x) for x in paths}))
    print('NUMERICAL_SUMMARY',checks['passed'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--evaluate',action='store_true');a=ap.parse_args()
    if a.evaluate:run()
    else:print('PREFLIGHT_ONLY_NO_INFERENCE_NO_TRAINING')
