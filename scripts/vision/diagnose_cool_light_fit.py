"""Existing endpoint fit diagnostic, no optimizer or training execution."""
import argparse
from collections import Counter
from pathlib import Path
from scripts.vision.cool_light_control import OUT as TRAIN,KEYS,prior
from scripts.vision.evaluate_cool_light import protocol,configure,adapter
configure()
complete=adapter.complete
from scripts.vision.structure_fit import truth_for
from scripts.vision.diagnose_material_late_rehearsal_fit import infer
from scripts.vision import diagnose_material_late_rehearsal_fit as runtime
OUT=TRAIN/'pool-fit-diagnosis-v1'

def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():p=prior.read(dest);prior.verify(p);return p
    p=protocol();members=[];models={};exposures={}
    deps=[TRAIN/'design.json',TRAIN/'evaluation/error-review-v1/completion.json',Path(__file__).resolve(),Path(runtime.__file__).resolve()]
    for key in KEYS:
        c=complete(key);xp=Path(c['exposure_path']);x=prior.read(xp);prior.verify(x)
        models[key]=dict(weights=c['weights'],weights_sha256=c['weights_sha256'])
        exposures[key]=dict(Counter(x['actual']));deps.extend([TRAIN/'training'/key/'completion.json',xp,Path(c['weights'])])
    for row in p['pool_rows']:
        for field in ('image','label'):
            path=Path(row[field+'_path'])
            if prior.file_sha256(path)!=row[field+'_sha256']:raise ValueError('Changed pool input')
            deps.append(path)
        members.append({**row,'truth':truth_for(row)})
    OUT.mkdir(exist_ok=True)
    return prior.frozen(dest,dict(status='existing_weights_fit_diagnostic_frozen',members=members,models=models,actual_exposures=exposures,
        environment={k:p['environment'][k] for k in ('torch','ultralytics')},
        inference=dict(device='cpu',imgsz=640,confidence=[.37,.001],iou=.7,max_det=300,match_iou=.5),
        interpretation='Unexposed members are not training fit; lineage/seed repetition does not establish independent coverage.',
        inputs={str(d):prior.file_sha256(d) for d in deps}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');a=ap.parse_args();p=freeze()
    print('FROZEN',len(p['members']),'members',flush=True)
    if a.infer:
        runtime.OUT=OUT
        for key in KEYS:infer(key,p)

