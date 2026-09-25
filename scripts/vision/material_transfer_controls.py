"""Freeze observational controls and a separate fixed scale stress test."""
import argparse
from pathlib import Path
from scripts.vision.material_member_fit import OUT as FIT,PRIOR,preflight,prior
from scripts.vision.infer_material_member_fit import validate_unit
from scripts.vision.record_material_feasibility_review import validate_decisions
OUT=FIT.parent/'material-transfer-controls-v1'

def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():prior.verify(prior.read(dest));return
    p=preflight();fc=prior.read(FIT/'completion.json');prior.verify(fc)
    dp=PRIOR/'recomputed-metrics.json';dev=prior.read(dp);prior.verify(dev)
    ep=PRIOR/'initial-gate.json';rp=PRIOR/'independent-review.json';e,r=prior.read(ep),prior.read(rp)
    for x in (e,r):prior.verify(x)
    validate_decisions(e['corrected_target_records'],r['decisions'])
    models={k:v for k,v in p['models'].items() if k!='v2.11'};rows=[]
    paths=[FIT/'protocol.json',FIT/'completion.json',FIT/'review.json',dp,ep,rp,PRIOR/'coverage-census.json',Path(__file__).resolve()]
    for key in models:
        f=FIT/'inference'/f'{key}.json';validate_unit(prior.read(f),key,p);paths.append(f)
    for m in p['members']:
        if any(t['class_name']=='reactor' for t in m['truth']):
            rows.append(dict(id=m['member_id'],role='training_fit',image_path=m['image_path'],image_sha256=m['image_sha256'],truth=m['truth'],variant=m['source_variant']))
    selected={(x['view_id'],x['variant']) for x in e['corrected_target_records'] if x['category']=='reactor'}
    for view,variant in sorted(selected):
        records=[x for x in e['corrected_target_records'] if (x['view_id'],x['variant'])==(view,variant)]
        # Keep original complete truth ordering, never assume a class-only target list.
        original=next(x for x in prior.read(prior.ROOT/'data/research/ml_training_recovery_v1/unified-hold-physical-lighting-control-v1/training-preflight/evaluation/R-clean-7.json')['rows'] if (x['view_id'],x['variant'])==(view,variant))
        rows.append(dict(id=view+'|'+variant,view_id=view,variant=variant,role='development_diagnostic',image_path=records[0]['image_path'],image_sha256=records[0]['image_sha256'],truth=original['truth']))
    if len(rows)!=26:raise ValueError('Expected six training and twenty development frames')
    for row in rows:
        if prior.file_sha256(row['image_path'])!=row['image_sha256']:raise ValueError('Changed image')
        paths.append(Path(row['image_path']))
    OUT.mkdir(exist_ok=True)
    prior.frozen(dest,dict(status='frozen_diagnostic_only',models=models,rows=rows,scales=[320,960],reference_scale=640,
        thresholds=[.37,.001],nms_iou=.7,max_det=300,device='cpu',agnostic_nms=False,
        controls=[
            'C1 same training pose original/neutral/background, complete instance pairing',
            'C2 same development pose original/material, all six seeds, clear/partial/unknown retained',
            'C3 observed bbox dimensions/aspect and source pose concentration, not causal matching',
            'C4 fixed low-threshold retained-prediction confidence and wrong-class evidence',
            'C5 retained candidate intervention and exact per-subset/class/negative-position replacement feasibility',
            'C6 reactor-related full-frame inference at 320/960, reference 640 unchanged'],
        decision_rules=['If train-gray remains accurate but development scale responses are inconsistent, prioritize viewpoint/condition coverage rather than a resolution fix.',
            'A consistent same-direction scale response supports scale sensitivity, not a production resolution or unique scale cause; whole-image context also changes.',
            'Lack of eligible reactor material candidates blocks an immediate reactor material training control; counts-only feasibility is insufficient.',
            'Do not optimize scales/thresholds after results; no training, new capture, model choice or sealed data.'],
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('FROZEN six controls; 26 frames x 6 existing models x 2 diagnostic resolutions')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');a=ap.parse_args()
    if a.freeze:freeze()
    else:prior.verify(prior.read(OUT/'protocol.json'));print('PREFLIGHT_ONLY')
