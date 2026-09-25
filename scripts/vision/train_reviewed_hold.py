"""Authorized three-seed controlled training; default optimizer-free preflight."""
import argparse,copy,hashlib,os,signal,subprocess,sys,traceback
from collections import Counter
from pathlib import Path
from scripts.vision.reviewed_hold_control import OUT as QUALITY,prior,COUNTS
from scripts.vision.brightness_lr_retention import OUT as REFERENCE,overrides,runtime_check
from scripts.vision import train_whole_image_hold as adapter
from scripts.vision import lineage_capped_control as sequences
from scripts.vision.brightness_transfer_runtime import make_dataset,check_log
from scripts.vision.order_retention_runtime import make_loader,check_actual
from scripts.vision.exposure_protocol import exposures

OUT=QUALITY/'training-control-v1';KEYS=tuple(f'brightness-450-{s}' for s in (7,17,27))


def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():prior.verify(prior.read(dest));return prior.read(dest)
    qp=QUALITY/'quality-review.json';cp=COUNTS/'counts.json';rp=REFERENCE/'protocol.json'
    for x in (qp,cp,rp):prior.verify(prior.read(x))
    q=prior.read(qp)
    if q['status']!='seven_increment_members_reviewed_no_new_hold' or q['new_risk_members']:raise ValueError('Quality gate blocked')
    p=copy.deepcopy(prior.read(rp));p.pop('identity');c=prior.read(cp);rows=p['pool_rows'];idx={r['member_id']:r for r in rows}
    paths=[qp,cp,rp,Path(__file__),Path(adapter.__file__),Path(sequences.__file__)];OUT.mkdir(parents=True,exist_ok=True)
    p['schedules']={};p['listings']={};p['ledger']={};p['baselines']={};p['brightness_factors']={};baseline=prior.read(rp)
    old_digest=sequences.digest
    sequences.digest=lambda *parts:hashlib.sha256(':'.join(map(str,('reviewed-hold-compensation-control-v1',*parts))).encode()).hexdigest()
    try:
        for seed,key in zip((7,17,27),KEYS):
            tc=REFERENCE/'training'/key/'completion.json';t=prior.read(tc);prior.verify(t);ep=Path(t['exposure_path']);prior.verify(prior.read(ep));original=prior.read(ep)['draws']
            runtime_check(t,seed,.0005)
            if original!=baseline['schedules'][key]:raise ValueError('Reference actual/plan conflict')
            counts=c['results'][str(seed)]['counts'];seq=sequences.sequence(rows,original,counts,seed)
            p['schedules'][key]=seq;p['brightness_factors'][key]=baseline['brightness_factors'][key]
            listing=OUT/(key+'.txt');listing.write_text('\n'.join(idx[m]['image_path'] for m in sorted(set(seq)))+'\n');p['listings'][key]=str(listing)
            p['ledger'][key]=dict(total=exposures(rows,seq),windows_50_steps=[exposures(rows,seq[i:i+300]) for i in range(0,2700,300)])
            p['baselines'][str(seed)]=dict(training_receipt=str(tc),evaluation=str(REFERENCE/'evaluation'/f'{key}.json'))
            paths += [tc,ep,listing,Path(t['weights']),REFERENCE/'evaluation'/f'{key}.json']
    finally:sequences.digest=old_digest
    for r in rows:
        for k in ('image','label'):
            path=Path(r[k+'_path'])
            if prior.file_sha256(path)!=r[k+'_sha256']:raise ValueError('Pool changed')
            paths.append(path)
    p.update(status='frozen_quality_checked_preflight_pending',held_members=c['held_members'],learning_rate=.0005,
        quality_scope='Seven increment members only; existing capped risks retained, not full-pool certification.',
        training_ready=False,training_started=False,inputs={str(x):prior.file_sha256(x) for x in paths})
    return prior.frozen(dest,p)


def preflight():
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    torch.set_num_threads(4);p=freeze();paths=[OUT/'protocol.json']
    for key in KEYS:
        dest=OUT/'preflight'/f'{key}.json';dest.parent.mkdir(exist_ok=True)
        if dest.exists():prior.verify(prior.read(dest));paths.append(dest);continue
        actual=[];log=[];batches=0;owner=SimpleNamespace(epoch=0);inverse={r['image_path']:r['member_id'] for r in p['pool_rows']}
        init_seeds(int(key.split('-')[-1]),deterministic=True)
        with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('No optimizer')),patch.object(torch.Tensor,'backward',side_effect=AssertionError('No backward')),patch.object(YOLO,'train',side_effect=AssertionError('No train')),patch.object(YOLO,'val',side_effect=AssertionError('No val')):
            loader=make_loader(make_dataset(p,key,log),p,key,owner)
            for epoch in range(45):
                owner.epoch=epoch
                for batch in loader:
                    if tuple(batch['img'].shape)!=(6,3,640,640):raise ValueError('Batch shape mismatch')
                    actual.extend(inverse[x] for x in batch['im_file']);batches+=1
        check_actual(p,key,actual);check_log(p,key,log)
        if batches!=450:raise ValueError('Wrong batch count')
        prior.frozen(dest,dict(status='actual_loader_verified',draws=actual,actual_brightness=log,batches=batches,optimizer_created=False,backward_executed=False,validation_run=False,inputs={str(OUT/'protocol.json'):prior.file_sha256(OUT/'protocol.json')}));paths.append(dest)
        print('PREFLIGHT_COMPLETE',key,flush=True)
    dest=OUT/'ready.json'
    if not dest.exists():prior.frozen(dest,dict(status='ready_for_training_not_started',cells=list(KEYS),inputs={str(x):prior.file_sha256(x) for x in paths}))
    return p


def ready():
    p=prior.read(OUT/'protocol.json');prior.verify(p);r=prior.read(OUT/'ready.json');prior.verify(r)
    if r['status']!='ready_for_training_not_started' or r['cells']!=list(KEYS):raise ValueError('Readiness absent')
    for key in KEYS:
        u=prior.read(OUT/'preflight'/f'{key}.json');prior.verify(u);check_actual(p,key,u['draws']);check_log(p,key,u['actual_brightness'])
    return p


def complete(key,p):
    adapter.complete(OUT/'training'/key/'completion.json',p,key)
    b=prior.read(OUT/'training'/key/'brightness-receipt.json');prior.verify(b);check_log(p,key,b['actual_brightness'])
    if b['actual_brightness']!=prior.read(OUT/'preflight'/f'{key}.json')['actual_brightness']:raise ValueError('Training tensors differ from preflight')
    runtime_check(prior.read(OUT/'training'/key/'completion.json'),int(key.split('-')[-1]),.0005)


def worker(key):
    p=ready();log=[];adapter.OUT=OUT;adapter.KEYS=KEYS;adapter.ready=lambda:p;adapter.overrides=overrides;adapter.make_dataset=lambda p,k:make_dataset(p,k,log)
    (OUT/'export').mkdir(exist_ok=True);adapter.worker(key);check_log(p,key,log)
    if log!=prior.read(OUT/'preflight'/f'{key}.json')['actual_brightness']:raise ValueError('Actual brightness mismatch')
    cp=OUT/'training'/key/'completion.json'
    prior.frozen(OUT/'training'/key/'brightness-receipt.json',dict(status='complete_brightness_and_lr_verified',actual_brightness=log,learning_rate=.0005,inputs={str(x):prior.file_sha256(x) for x in (cp,Path(__file__),OUT/'ready.json')}))
    complete(key,p)


def run():
    p=preflight()
    for key in KEYS:
        done=OUT/'training'/key/'brightness-receipt.json'
        if done.exists():complete(key,p);continue
        root=OUT/'workers'/key;root.mkdir(parents=True,exist_ok=True);attempt=root/f'attempt-{len(list(root.glob("attempt-*")))+1:03}'
        if int(attempt.name.split('-')[-1])>3:raise ValueError('Attempt cap exhausted')
        attempt.mkdir();proc=None
        try:
            with (attempt/'log.txt').open('x') as log:
                proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.train_reviewed_hold','--worker',key],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                proc.wait(timeout=21600)
                if proc.returncode:raise RuntimeError('Worker failed; inspect '+str(attempt))
            complete(key,p);prior.frozen(attempt/'completion.json',dict(status='complete',process_cleanup_confirmed=True,inputs={str(attempt/'log.txt'):prior.file_sha256(attempt/'log.txt')}))
        except BaseException:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
            prior.frozen(attempt/'failure.json',dict(error=traceback.format_exc(),process_cleanup_confirmed=proc is None or proc.poll() is not None));raise
        print('TRAINED',key,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--worker',choices=KEYS);a=ap.parse_args()
    if a.worker:worker(a.worker)
    elif a.run:run()
    else:preflight()
