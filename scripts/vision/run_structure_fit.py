"""Explicit fixed inference, bounded isolated workers; never training."""
import argparse
from collections import Counter
from contextlib import ExitStack
import os
from pathlib import Path
import signal
import subprocess
import sys
from unittest.mock import patch
from scripts.vision.structure_fit import OUT,ROOT,read,frozen,file_sha256,verify,truth_for,scoring
from scripts.vision.evaluate_exposure_diagnosis import predict

def forbidden(*args,**kwargs):raise RuntimeError('Training/optimizer/backward/validation forbidden in structure-fit diagnosis')

def validate_unit(record,key,p):
    verify(record)
    if record['status']!='complete' or record['model']!=key or record['protocol_identity']!=p['identity']:raise ValueError('Unit identity mismatch')
    if len(record['pool'])!=236 or {r['member_id'] for r in record['pool']}!={r['member_id'] for r in p['rows']}:raise ValueError('Incomplete pool inference')
    if len(record['development'])!=96 or {(r['view_id'],r['variant']) for r in record['development']}!={(r['view_id'],r['variant']) for r in p['development']}:raise ValueError('Incomplete development inference')

def worker(key,attempt):
    p=read(OUT/'protocol.json');verify(p);m=p['models'][key]
    if file_sha256(m['weights'])!=m['weights_sha256']:raise ValueError('Stale weights')
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    inputs={str(OUT/'protocol.json'):file_sha256(OUT/'protocol.json'),m['weights']:m['weights_sha256']}
    counts=Counter(m['draws']);pool=[];development=[]
    with ExitStack() as stack:
        for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(YOLO,'train'),(YOLO,'val')):stack.enter_context(patch.object(obj,name,forbidden))
        model=YOLO(m['weights'])
        if list(model.names.values())!=list(p['rows'][0].get('names',('transformer','switchgear','capacitor_bank','reactor'))):raise ValueError('Class mapping mismatch')
        with torch.inference_mode():
            for i,r in enumerate(p['rows']):
                formal=predict(model,r['image_path'],.37);low=predict(model,r['image_path'],.001)
                pool.append(dict(member_id=r['member_id'],image_sha256=r['image_sha256'],actual_exposures=counts[r['member_id']],
                    exposure_role=m['exposure_role'],**scoring(truth_for(r),formal,low)))
                if (i+1)%40==0:print(key,'POOL',i+1,'/236',flush=True)
            if m.get('development_evaluation'):
                ep=Path(m['development_evaluation']);e=read(ep);verify(e);inputs[str(ep)]=file_sha256(ep)
                rm={(r['view_id'],r['variant']):r for r in e['rows']+e['negative_rows']}
                if len(rm)!=96:raise ValueError('Invalid reusable development unit')
                for r in p['development']:
                    old=rm[r['view_id'],r['variant']]
                    if old['image_sha256']!=r['image_sha256']:raise ValueError('Stale reused pixels')
                    low=old.get('low_predictions',old.get('diagnostic_predictions'))
                    if low is None:raise ValueError('Missing independent low-threshold output')
                    if old.get('truth',[])!=r['truth']:raise ValueError('Reused truth mismatch')
                    development.append(dict(view_id=r['view_id'],variant=r['variant'],pair_id=r.get('pair_id'),image_sha256=r['image_sha256'],
                        **scoring(r['truth'],old['predictions'],low)))
            else:
                for r in p['development']:
                    development.append(dict(view_id=r['view_id'],variant=r['variant'],pair_id=r.get('pair_id'),image_sha256=r['image_sha256'],
                        **scoring(r['truth'],predict(model,r['image_path'],.37),predict(model,r['image_path'],.001))))
    result=frozen(attempt/'result.json',dict(status='complete',model=key,protocol_identity=p['identity'],pool=pool,development=development,
        optimizer_created=False,backward_executed=False,training_validation_executed=False,inputs=inputs))
    validate_unit(result,key,p)
    frozen(OUT/'inference'/f'{key}.json',dict(**{k:v for k,v in result.items() if k!='identity'},inputs={**inputs,str(attempt/'result.json'):file_sha256(attempt/'result.json')}))
    print('UNIT_COMPLETE',key,flush=True)

def cleanup(proc):
    if proc.poll() is not None:return
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)

def run():
    p=read(OUT/'protocol.json');verify(p);(OUT/'inference').mkdir(exist_ok=True)
    lock=OUT/'inference.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        for key in p['models']:
            cp=OUT/'inference'/f'{key}.json'
            if cp.exists():validate_unit(read(cp),key,p);print('REUSE',key,flush=True);continue
            folder=OUT/'inference'/key;folder.mkdir(exist_ok=True)
            for _ in range(3):
                number=len(list(folder.glob('attempt-*')))+1
                if number>3:raise ValueError('Attempt limit reached')
                attempt=folder/f'attempt-{number:03}';attempt.mkdir();proc=None
                try:
                    with (attempt/'worker.log').open('w') as log:
                        proc=subprocess.Popen([sys.executable,'-m','scripts.vision.run_structure_fit','--worker',key,'--attempt',str(attempt)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        code=proc.wait(timeout=3600)
                    if code:
                        frozen(attempt/'failure.json',dict(status='failed',returncode=code,inputs={str(attempt/'worker.log'):file_sha256(attempt/'worker.log')}))
                        if code==3:raise ValueError('Semantic failure; no retry')
                        continue
                    validate_unit(read(cp),key,p);print('COMPLETE',key,flush=True);break
                except BaseException as exc:
                    if proc:cleanup(proc)
                    if not (attempt/'failure.json').exists():frozen(attempt/'failure.json',dict(status='interrupted_or_exception',error=str(exc),inputs={}))
                    raise
                finally:
                    if proc:cleanup(proc)
            else:raise ValueError('Technical retry limit exhausted')
    finally:lock.unlink(missing_ok=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');ap.add_argument('--worker');ap.add_argument('--attempt',type=Path);args=ap.parse_args()
    if args.worker:
        if not args.attempt:ap.error('worker requires attempt')
        try:worker(args.worker,args.attempt)
        except ValueError as exc:print('SEMANTIC_FAILURE',exc,flush=True);sys.exit(3)
    elif args.infer:run()
    else:print('PREFLIGHT_ONLY_NO_INFERENCE_NO_TRAINING')

if __name__=='__main__':main()
