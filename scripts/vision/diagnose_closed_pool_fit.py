"""Explicit, bounded CPU inference on the frozen pool; no training paths."""
import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import subprocess, sys, time, traceback
from scripts.vision.train_closed_source_control import OUT as TRAIN, KEYS, contract, complete, prior, cleanup
from scripts.vision.structure_fit import truth_for, scoring
from scripts.vision.run_fixed_budget_diagnosis import predict
from scripts.vision.infer_material_member_fit import forbidden

OUT=TRAIN/'pool-fit-diagnosis-v1'

def freeze():
    path=OUT/'protocol.json'
    if path.exists():p=prior.read(path);prior.verify(p);return p
    _,source,_=contract(KEYS[0]);members=[];models={};exposures={}
    deps=[TRAIN/'protocol.json',TRAIN/'cpu-inference-policy-lock.json',Path(__file__).resolve(),TRAIN/'evaluation/error-review-v1/report-receipt.json']
    for m in source['pool_rows']:
        for field in ('image','label'):
            if prior.file_sha256(m[field+'_path'])!=m[field+'_sha256']:raise ValueError('Changed member')
            deps.append(Path(m[field+'_path']))
        truth=truth_for(m)
        if dict(Counter(t['class_name'] for t in truth))!=m['class_instances']:raise ValueError('Full label counts changed')
        members.append(dict(member_id=m['member_id'],image_path=m['image_path'],image_sha256=m['image_sha256'],truth=truth,
            subset=m['subset'],variant=m['variant'],lineage_id=m['lineage_id'],lineage_resolution=m.get('lineage_resolution','unknown')))
    if len({m['member_id'] for m in members})!=len(members):raise ValueError('Duplicate members')
    for key in KEYS:
        complete(key);cp=TRAIN/'training'/key/'completion.json';c=prior.read(cp);xp=Path(c['exposure_path']);x=prior.read(xp);prior.verify(x)
        models[key]=dict(weights=c['weights'],weights_sha256=c['weights_sha256']);exposures[key]=dict(Counter(x['actual']))
        if sum(exposures[key].values())!=2700:raise ValueError('Exposure count')
        deps.extend([cp,xp,Path(c['weights'])])
    for dep in deps:
        if dep.suffix=='.json' and dep.name in ('report-receipt.json','cpu-inference-policy-lock.json'):prior.verify(prior.read(dep))
    OUT.mkdir(exist_ok=True)
    return prior.frozen(path,dict(status='frozen_not_training',members=members,models=models,exposures=exposures,
        inference=dict(device='cpu',imgsz=640,confidence=[.37,.001],iou=.7,max_det=300,agnostic_nms=False,threads=4,parallel=2),
        interpretation='Per-model exposed members only are training fit; unexposed members and shared assets are not independent generalization evidence.',
        inputs={str(d):prior.file_sha256(d) for d in deps}))

def validate(r,p,key):
    prior.verify(r)
    if r['protocol_identity']!=p['identity'] or r['model']!=key or len(r['rows'])!=len(p['members']):raise ValueError('Incomplete identity')
    for row,m in zip(r['rows'],p['members'],strict=True):
        if row['member_id']!=m['member_id'] or row['exposures']!=p['exposures'][key].get(m['member_id'],0):raise ValueError('Member drift')
        if scoring(m['truth'],row['predictions'],row['low_predictions'])!={k:row[k] for k in scoring(m['truth'],row['predictions'],row['low_predictions'])}:raise ValueError('Score drift')

def worker(key):
    p=freeze();dest=OUT/(key+'.json')
    if dest.exists():validate(prior.read(dest),p,key);return
    root=OUT/key;root.mkdir(exist_ok=True);n=len(list(root.glob('attempt-*')))+1
    if n>3:raise ValueError('Attempt cap')
    if any(prior.read(f)['semantic'] for f in root.glob('attempt-*/failure.json')):raise ValueError('Semantic stop')
    attempt=root/f'attempt-{n:03}';attempt.mkdir()
    try:
        import torch
        from ultralytics import YOLO
        rows=[];start=time.monotonic()
        with ExitStack() as stack:
            for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):stack.enter_context(patch.object(obj,name,forbidden))
            model=YOLO(p['models'][key]['weights'])
            with torch.inference_mode():
                predict(model,p['members'][0]['image_path'],.37);torch.set_num_threads(4)
                for m in p['members']:
                    rows.append(dict(member_id=m['member_id'],exposures=p['exposures'][key].get(m['member_id'],0),**scoring(m['truth'],predict(model,m['image_path'],.37),predict(model,m['image_path'],.001))))
                    if torch.get_num_threads()!=4:raise ValueError('Actual CPU threads drift')
        r=prior.frozen(attempt/'result.json',dict(status='complete',model=key,protocol_identity=p['identity'],rows=rows,seconds=time.monotonic()-start,
            optimizer_created=False,backward_executed=False,validation_run=False,effective_threads=4,inputs={str(OUT/'protocol.json'):prior.file_sha256(OUT/'protocol.json')}))
        validate(r,p,key)
        prior.frozen(dest,dict(**{k:v for k,v in r.items() if k not in ('identity','inputs')},inputs={**r['inputs'],str(attempt/'result.json'):prior.file_sha256(attempt/'result.json')}))
    except BaseException as exc:
        prior.frozen(attempt/'failure.json',dict(semantic=isinstance(exc,ValueError),error=traceback.format_exc()));raise

def run():
    p=freeze()
    for i in range(0,len(KEYS),2):
        processes=[]
        try:
            for key in KEYS[i:i+2]:
                if (OUT/(key+'.json')).exists():validate(prior.read(OUT/(key+'.json')),p,key);continue
                proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.diagnose_closed_pool_fit','--worker',key],start_new_session=True,cwd=prior.ROOT)
                processes.append(proc);print('FIT_STARTED',key,proc.pid,flush=True)
            for proc in processes:
                proc.wait(timeout=1800)
                if proc.returncode:raise RuntimeError('Fit worker failed')
        finally:
            for proc in processes:cleanup(proc)
    for key in KEYS:validate(prior.read(OUT/(key+'.json')),p,key)
    print('ALL_SIX_FIT_INFERENCE_COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');ap.add_argument('--worker',choices=KEYS);a=ap.parse_args()
    if a.worker:worker(a.worker)
    elif a.infer:run()
    else:print('PREFLIGHT_ONLY',len(freeze()['members']))
