"""Fix receipt publication without changing the frozen inference protocol."""
import argparse,fcntl,os,signal,subprocess,sys,traceback
from pathlib import Path
from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch
from scripts.vision import check_lineage_training_fit as base

def publish(u,key,p,result_path):
    base.valid(u,key,p)
    record={k:v for k,v in u.items() if k not in ('identity','inputs')}
    record['inputs']={**u['inputs'],str(result_path):base.file_sha256(result_path),str(Path(__file__)):base.file_sha256(Path(__file__))}
    return base.frozen(base.OUT/'inference'/f'{key}.json',record)

def worker(key,attempt):
    p=base.read(base.OUT/'protocol.json');base.verify(p);m=p['models'][key]
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4);rows=[];counts=Counter(m['draws'])
    with ExitStack() as stack:
        for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(YOLO,'train'),(YOLO,'val')):stack.enter_context(patch.object(obj,name,base.forbidden))
        model=YOLO(m['weights'])
        if list(model.names.values())!=['transformer','switchgear','capacitor_bank','reactor']:raise ValueError('Class mapping drift')
        with torch.inference_mode():
            for r in p['rows']:
                rows.append(dict(member_id=r['member_id'],image_sha256=r['image_sha256'],exposures=counts[r['member_id']],scoring=base.scoring(base.truth_for(r),base.predict(model,r['image_path'],.37),base.predict(model,r['image_path'],.001))))
    path=attempt/'result.json'
    u=base.frozen(path,dict(status='complete',model=key,protocol_identity=p['identity'],rows=rows,optimizer_created=False,backward_executed=False,training_validation_executed=False,
        inputs={str(x):base.file_sha256(x) for x in [base.OUT/'protocol.json',Path(__file__),Path(m['weights'])]}))
    publish(u,key,p,path)

def run():
    p=base.freeze()
    with (base.OUT/'inference.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for key in p['models']:
            cp=base.OUT/'inference'/f'{key}.json'
            if cp.exists():base.valid(base.read(cp),key,p);print('REUSE',key,flush=True);continue
            root=base.OUT/'inference'/key;root.mkdir(exist_ok=True)
            results=list(root.glob('attempt-*/result.json'))
            if results:
                # Only a complete, byte-valid inference result may be recovered.
                if len(results)!=1:raise ValueError('Ambiguous complete result')
                publish(base.read(results[0]),key,p,results[0]);print('RECOVERED_PUBLICATION',key,flush=True);continue
            n=len(list(root.glob('attempt-*')))+1
            if n>3:raise ValueError('Attempt budget exhausted')
            attempt=root/f'attempt-{n:03}';attempt.mkdir();proc=None
            try:
                with (attempt/'worker.log').open('x') as log:
                    proc=subprocess.Popen([sys.executable,'-m','scripts.vision.check_lineage_training_fit_v2','--worker',key,'--attempt',str(attempt)],cwd=base.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    proc.wait(timeout=1800)
                    if proc.returncode:raise RuntimeError('Worker failure '+key)
                base.valid(base.read(cp),key,p)
            except BaseException:
                if proc and proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
                base.frozen(attempt/'failure.json',dict(error=traceback.format_exc(),process_cleanup_confirmed=proc is None or proc.poll() is not None));raise
            print('INFERRED',key,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');ap.add_argument('--worker');ap.add_argument('--attempt',type=Path);a=ap.parse_args()
    if a.worker:
        if not a.attempt:ap.error('worker requires attempt')
        worker(a.worker,a.attempt)
    elif a.infer:run()
    else:print('READ_ONLY_NO_INFERENCE_NO_TRAINING')
