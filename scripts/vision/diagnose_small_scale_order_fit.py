"""Existing endpoints only: full-pool fit evidence for a future reviewed dataset."""
import argparse
from collections import Counter
from pathlib import Path
from scripts.vision.interleaved_small_scale_control import OUT as TRAIN, KEYS, prior, source
from scripts.vision.structure_fit import truth_for
from scripts.vision import diagnose_material_late_rehearsal_fit as runtime

OUT=TRAIN/'reviewed-dataset-fit-v1'


def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():
        p=prior.read(dest);prior.verify(p);return p
    dp=TRAIN/'design.json';d=prior.read(dp);prior.verify(d)
    audit=TRAIN/'evaluation/error-review-v1/completion.json';prior.verify(prior.read(audit))
    deps=[dp,audit,Path(__file__).resolve(),Path(runtime.__file__).resolve(),
          Path(__file__).with_name('structure_fit.py'),Path(__file__).with_name('evaluate_exposure_diagnosis.py')]
    members=[];models={};exposures={}
    for m in d['pool_rows']:
        for k in ('image','label'):
            p=Path(m[k+'_path'])
            if prior.file_sha256(p)!=m[k+'_sha256']:raise ValueError('Member drift: '+m['member_id'])
            deps.append(p)
        truth=truth_for(m)
        if dict(Counter(t['class_name'] for t in truth))!=m['class_instances']:
            raise ValueError('Full label count drift: '+m['member_id'])
        members.append(dict(m,truth=truth))
    for root,keys in ((TRAIN,KEYS),(source.OUT,source.KEYS)):
        design=prior.read(root/'design.json');prior.verify(design);deps.append(root/'design.json')
        for key in keys:
            cp=root/'training'/key/'completion.json';c=prior.read(cp);prior.verify(c)
            xp=Path(c['exposure_path']);x=prior.read(xp);prior.verify(x)
            if x['actual']!=design['schedules'][key] or c['optimizer_steps']!=1100:
                raise ValueError('Actual exposure or endpoint mismatch')
            w=Path(c['weights'])
            if prior.file_sha256(w)!=c['weights_sha256']:raise ValueError('Weight drift')
            models[key]=dict(weights=str(w),weights_sha256=c['weights_sha256'])
            exposures[key]=dict(Counter(x['actual']));deps.extend([cp,xp,w])
    OUT.mkdir(exist_ok=True)
    return prior.frozen(dest,dict(status='fit_diagnostic_frozen_dataset_not_yet_ready',members=members,
        models=models,actual_exposures=exposures,
        environment={k:d['environment'][k] for k in ('torch','ultralytics')},
        inference=dict(device='cpu',imgsz=640,confidence=[.37,.001],iou=.7,max_det=300,match_iou=.5),
        interpretation='Unexposed members are not training fit. Fit scores never approve labels or certify independent coverage.',
        inputs={str(p):prior.file_sha256(p) for p in deps}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--infer-key');a=ap.parse_args();p=freeze()
    print('FROZEN',len(p['members']),len(p['models']),flush=True)
    if a.infer_key:
        if a.infer_key not in p['models']:raise ValueError('Unknown endpoint')
        runtime.OUT=OUT;runtime.infer(a.infer_key,p)
