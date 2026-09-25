"""Explicit inference-only, reviewed-target fit check; never trains."""
import argparse, os, signal, subprocess, sys, traceback
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from scripts.vision.lineage_capped_control import OUT as SOURCE, RUN, ROOT, CONTROL, SECOND, ready, old_ready, read, verify, frozen, file_sha256
from scripts.vision.train_lineage_capped_v2 import completed
from scripts.vision.train_whole_image_hold import complete
from scripts.vision.structure_fit import truth_for, scoring
from scripts.vision.evaluate_exposure_diagnosis import predict
from scripts.vision.decide_risk_capped_control import validate
from scripts.vision.prepare_whole_image_hold import baseline_verify

OUT=SOURCE/'reviewed-target-fit-check-v1'
# Frozen from prior explicit visual descriptions, before any current fit prediction.
CLEAR=set('C02-L2 C03-L2 C04-L3 C05-L0 C07-L5 C09-L4 C10-L4 C12-L2 C13-L3 C14-L3 C15-L1 C17-L2 C23-L2 C24-L0 C25-L7 C26-L4 C30-L10 C31-L0 C34-L1 C35-L0 C37-L1 C38-L0 C40-L4 N01-L0 N02-L1 N03-L1'.split())

def forbidden(*a,**kw): raise RuntimeError('Training, optimizer, backward and validation forbidden')

def check_review(d,r,t):
    if d['image_sha256']!=r['image_sha256'] or d['label_sha256']!=r['label_sha256'] or d['truth']!=t: raise ValueError('Review/image/full-label identity mismatch')

def freeze():
    dest=OUT/'protocol.json'
    if dest.exists(): verify(read(dest)); return read(dest)
    new=ready(); old=old_ready(); verify(read(SOURCE/'completion.json'))
    idx={r['member_id']:r for r in new['pool_rows']}; targets=[]; seen=set()
    paths=[SOURCE/'protocol.json',SOURCE/'completion.json',RUN/'protocol.json',Path(__file__),ROOT/'scripts/vision/structure_fit.py',ROOT/'scripts/vision/evaluate_exposure_diagnosis.py',ROOT/'scripts/vision/evaluate_paired_visual_factors.py',ROOT/'scripts/vision/exposure_metrics.py']
    for folder in (CONTROL,SECOND):
        e=read(folder/'evidence.json'); v=read(folder/'review.json');verify(e);verify(v);validate(e,v['decisions']);paths.extend([folder/'evidence.json',folder/'review.json'])
        for d in v['decisions']:
            if d['truth']['class_name'] not in ('reactor','capacitor_bank') or d['status']!='identifiable_geometry_with_recorded_limits':continue
            r=idx[d['member_id']];t=truth_for(r)[d['truth']['label_line_index']];check_review(d,r,t)
            key=(d['member_id'],t['annotation_id'])
            if key in seen:raise ValueError('Duplicate reviewed instance')
            seen.add(key);targets.append(dict(review=d,clear_primary=d['event_id'] in CLEAR))
    if not CLEAR<={t['review']['event_id'] for t in targets}:raise ValueError('Missing clear target')
    members={t['review']['member_id'] for t in targets};rows=[idx[m] for m in sorted(members)]
    for r in rows:
        for kind in ('image','label'):
            if file_sha256(r[kind+'_path'])!=r[kind+'_sha256']:raise ValueError('Stale member')
            paths.append(Path(r[kind+'_path']))
        if Counter(t['class_name'] for t in truth_for(r))!=Counter(r['class_instances']):raise ValueError('Full class counts mismatch')
    models={}
    for seed in (7,17,27):
        for arm,folder,p in [('reference',RUN,old),('hold',RUN,old),('lineage-capped',SOURCE,new)]:
            key=f'{arm}-450-{seed}';cp=folder/'training'/key/'completion.json'
            if arm=='lineage-capped':completed(key,p)
            else:complete(cp,p,key)
            c=read(cp);ep=Path(c['exposure_path']);actual=read(ep);verify(actual)
            if actual['draws']!=p['schedules'][key]:raise ValueError('Actual sequence mismatch')
            dev=folder/'evaluation'/f'{key}.json';verify(read(dev))
            models[key]=dict(weights=c['weights'],weights_sha256=c['weights_sha256'],draws=actual['draws'],development=str(dev))
            paths.extend([cp,ep,Path(c['weights']),dev])
    OUT.mkdir(exist_ok=True)
    p=frozen(dest,dict(status='frozen_diagnosis_not_training',rows=rows,targets=targets,models=models,
        selection='All previously identifiable reactor/capacitor labels in two explicit reviews; clear primary subset fixed from descriptions before predictions; not whole-pool coverage.',
        inference=dict(imgsz=640,device='cpu',confidence=[.37,.001],nms_iou=.7,max_det=300,agnostic_nms=False,matching='same_class_one_to_one_iou_0.5'),
        inputs={str(x):file_sha256(x) for x in paths}))
    print('FROZEN',len(rows),'IMAGES',len(targets),'TARGETS',len(CLEAR),'CLEAR',flush=True);return p

def valid(u,key,p):
    verify(u)
    if u['status']!='complete' or u['model']!=key or u['protocol_identity']!=p['identity']:raise ValueError('Unit identity mismatch')
    if len(u['rows'])!=len(p['rows']) or {r['member_id'] for r in u['rows']}!={r['member_id'] for r in p['rows']}:raise ValueError('Incomplete inference')
    if any(u[k] is not False for k in ('optimizer_created','backward_executed','training_validation_executed')):raise ValueError('Training forbidden')
    idx={r['member_id']:r for r in p['rows']};counts=Counter(p['models'][key]['draws'])
    for r in u['rows']:
        if r['exposures']!=counts[r['member_id']] or r['image_sha256']!=idx[r['member_id']]['image_sha256']:raise ValueError('Exposure/image mismatch')
        if r['scoring']!=scoring(truth_for(idx[r['member_id']]),r['scoring']['predictions'],r['scoring']['low_predictions']):raise ValueError('Score mismatch')

def worker(key,attempt):
    p=read(OUT/'protocol.json');verify(p);m=p['models'][key]
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4);rows=[];counts=Counter(m['draws'])
    with ExitStack() as stack:
        for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(YOLO,'train'),(YOLO,'val')):stack.enter_context(patch.object(obj,name,forbidden))
        model=YOLO(m['weights'])
        if list(model.names.values())!=['transformer','switchgear','capacitor_bank','reactor']:raise ValueError('Class mapping drift')
        with torch.inference_mode():
            for r in p['rows']:
                rows.append(dict(member_id=r['member_id'],image_sha256=r['image_sha256'],exposures=counts[r['member_id']],scoring=scoring(truth_for(r),predict(model,r['image_path'],.37),predict(model,r['image_path'],.001))))
    u=frozen(attempt/'result.json',dict(status='complete',model=key,protocol_identity=p['identity'],rows=rows,
        optimizer_created=False,backward_executed=False,training_validation_executed=False,
        inputs={str(x):file_sha256(x) for x in [OUT/'protocol.json',Path(__file__),Path(m['weights'])]}))
    valid(u,key,p)
    frozen(OUT/'inference'/f'{key}.json',dict(**{k:v for k,v in u.items() if k!='identity'},inputs={**u['inputs'],str(attempt/'result.json'):file_sha256(attempt/'result.json')}))

def run():
    p=freeze();(OUT/'inference').mkdir(exist_ok=True)
    import fcntl
    with (OUT/'inference.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for key in p['models']:
            cp=OUT/'inference'/f'{key}.json'
            if cp.exists():valid(read(cp),key,p);print('REUSE',key,flush=True);continue
            root=OUT/'inference'/key;root.mkdir(exist_ok=True);n=len(list(root.glob('attempt-*')))+1
            if n>3:raise ValueError('Attempt budget exhausted')
            attempt=root/f'attempt-{n:03}';attempt.mkdir();proc=None
            try:
                with (attempt/'worker.log').open('x') as log:
                    proc=subprocess.Popen([sys.executable,'-m','scripts.vision.check_lineage_training_fit','--worker',key,'--attempt',str(attempt)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    proc.wait(timeout=1800)
                    if proc.returncode:raise RuntimeError('Worker failure: inspect '+str(attempt))
                valid(read(cp),key,p)
            except BaseException:
                if proc and proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
                frozen(attempt/'failure.json',dict(error=traceback.format_exc(),process_cleanup_confirmed=proc is None or proc.poll() is not None));raise
            print('INFERRED',key,flush=True)

def summarize():
    p=read(OUT/'protocol.json');verify(p);records={};paths=[OUT/'protocol.json',Path(__file__)];details=[]
    for key in p['models']:
        path=OUT/'inference'/f'{key}.json';u=read(path);valid(u,key,p);records[key]={r['member_id']:r for r in u['rows']};paths.append(path)
    common={m for m in records[next(iter(records))] if all(rs[m]['exposures']>0 for rs in records.values())}
    for key,rs in records.items():
        for t in p['targets']:
            d=t['review'];r=rs[d['member_id']];i=d['truth']['label_line_index'];s=r['scoring'];hit=any(m['truth_index']==i for m in s['matches']);miss=next((m for m in s['misses'] if m['truth_index']==i),None)
            draws=p['models'][key]['draws'];member=d['member_id']
            details.append(dict(model=key,event_id=d['event_id'],member_id=member,annotation_id=d['truth']['annotation_id'],class_name=d['truth']['class_name'],clear_primary=t['clear_primary'],common_exposed=member in common,actual_exposures=r['exposures'],windows_50_steps=[draws[j:j+300].count(member) for j in range(0,2700,300)],hit=hit,miss=miss,review_reason=d['reason']))
    metrics={}
    for key in records:
        metrics[key]={}
        for scope in ('all_reviewed_identifiable','clear_primary'):
            metrics[key][scope]={}
            for cls in ('reactor','capacitor_bank'):
                ds=[d for d in details if d['model']==key and d['class_name']==cls and d['common_exposed'] and (scope!='clear_primary' or d['clear_primary'])]
                metrics[key][scope][cls]=dict(instances=len(ds),hits=sum(d['hit'] for d in ds),recall=sum(d['hit'] for d in ds)/len(ds) if ds else None,miss_reasons=dict(Counter(d['miss']['reason'] for d in ds if not d['hit'])))
    base=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not base['integrity_passed'] or base['pinned_files_verified']!=40:raise ValueError('Pinned integrity failure')
    frozen(OUT/'summary.json',dict(status='targeted_training_fit_diagnosis_complete',metrics=metrics,details=details,common_exposed_members=len(common),baseline=base,
        interpretation='Training fitting diagnostic on a reviewed subset, not generalization or acceptance; no pixel-visibility certification or automatic review decisions.',inputs={str(x):file_sha256(x) for x in paths}))
    print(metrics,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--infer',action='store_true');ap.add_argument('--summarize',action='store_true');ap.add_argument('--worker');ap.add_argument('--attempt',type=Path);a=ap.parse_args()
    if a.worker:
        if not a.attempt:ap.error('worker requires attempt')
        worker(a.worker,a.attempt)
    elif a.infer:run()
    elif a.summarize:summarize()
    elif a.freeze:freeze()
    else:print('READ_ONLY_NO_TRAINING_NO_INFERENCE')
