"""Full actual six-loader traversal; default preflight never trains."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse, subprocess,sys,traceback
from scripts.vision.closed_budget_design import OUT,KEYS,freeze,prior
from scripts.vision.closed_budget_runtime import make_dataset,make_loader,check
from scripts.vision.preflight_unified_lighting import tensor_hash
from scripts.vision.infer_material_member_fit import forbidden
from scripts.vision.train_closed_source_control import cleanup

def receipt(key,p):
    files=list((OUT/'loader-checks'/key).glob('attempt-*/complete.json'))
    if len(files)!=1:raise ValueError('Missing/duplicate preflight')
    r=prior.read(files[0]);prior.verify(r);check(p,key,r['actual'],r['brightness_log'])
    if r['cell']!=key or len(r['batch_records'])*6!=len(p['schedules'][key]):raise ValueError('Incomplete batches')
    return r,files[0]

def worker(key):
    p=freeze();root=OUT/'loader-checks'/key;root.mkdir(parents=True,exist_ok=True)
    if list(root.glob('attempt-*/complete.json')):receipt(key,p);return
    n=len(list(root.glob('attempt-*')))+1
    if n>3:raise ValueError('Preflight attempt cap')
    attempt=root/f'attempt-{n:03}';attempt.mkdir()
    try:
        import torch
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import init_seeds
        torch.set_num_threads(4);init_seeds(int(key.split('-')[-1]),deterministic=True)
        lookup={m['image_path']:m['member_id'] for m in p['pool_rows']};owner=SimpleNamespace(epoch=0);actual=[];logs=[];batches=[]
        with ExitStack() as stack:
            for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):stack.enter_context(patch.object(obj,name,forbidden))
            loader=make_loader(make_dataset(p,key,logs),p,key,owner)
            for epoch in range(p['training_config'][key]['epochs']):
                owner.epoch=epoch
                for batch in loader:
                    if tuple(batch['img'].shape)!=(6,3,640,640):raise ValueError('Batch size drift')
                    members=[lookup[x] for x in batch['im_file']];actual.extend(members)
                    batches.append(dict(step=len(batches),members=members,image_tensor_sha256=tensor_hash(batch['img']),full_supervision={f:tensor_hash(batch[f]) for f in ('cls','bboxes','batch_idx')}))
        check(p,key,actual,logs)
        deps=[OUT/'design.json',Path(__file__).resolve(),Path(__file__).with_name('closed_budget_runtime.py')]
        prior.frozen(attempt/'complete.json',dict(status='actual_loader_verified_no_training',cell=key,actual=actual,brightness_log=logs,batch_records=batches,
            optimizer_created=False,backward_executed=False,validation_run=False,inputs={str(d):prior.file_sha256(d) for d in deps}))
        print('PREFLIGHT_COMPLETE',key,len(actual),flush=True)
    except BaseException:prior.frozen(attempt/'failure.json',dict(error=traceback.format_exc()));raise

def run():
    p=freeze()
    for i in range(0,len(KEYS),2):
        procs=[]
        try:
            for key in KEYS[i:i+2]:
                proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.preflight_closed_budget','--worker',key],cwd=prior.ROOT,start_new_session=True)
                procs.append(proc)
            for proc in procs:
                proc.wait(timeout=1200)
                if proc.returncode:raise RuntimeError('Preflight failed')
        finally:
            for proc in procs:cleanup(proc)
    deps=[OUT/'design.json',Path(__file__).resolve()]
    for seed in (7,17,27):
        short,sp=receipt(f'B450-{seed}',p);long,lp=receipt(f'B900-{seed}',p);deps.extend([sp,lp])
        if long['batch_records'][:450]!=short['batch_records']:raise ValueError('Actual prefix mismatch')
        for a,b in zip(long['batch_records'][450:],short['batch_records'],strict=True):
            if {k:v for k,v in a.items() if k!='step'}!={k:v for k,v in b.items() if k!='step'}:raise ValueError('Second cycle tensor/labels changed')
    dest=OUT/'loader-completion.json'
    if dest.exists():prior.verify(prior.read(dest));return
    prior.frozen(dest,dict(status='six_actual_loaders_verified_cpu_benchmark_pending',checked_exposures=24300,checked_batches=4050,inputs={str(d):prior.file_sha256(d) for d in deps}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=KEYS);a=ap.parse_args()
    if a.worker:worker(a.worker)
    else:run()
