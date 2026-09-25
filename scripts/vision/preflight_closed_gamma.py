"""Full actual six-loader traversal; default preflight never trains."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse, subprocess,sys,traceback
from scripts.vision.closed_gamma_design import OUT,KEYS,freeze,prior
from scripts.vision.closed_gamma_runtime import make_dataset,make_loader,check
from scripts.vision.preflight_closed_budget import receipt as base_receipt
from scripts.vision.closed_budget_design import freeze as base_design
from scripts.vision.preflight_unified_lighting import tensor_hash
from scripts.vision.infer_material_member_fit import forbidden
from scripts.vision.train_closed_source_control import cleanup

def receipt(key,p):
    files=list((OUT/'loader-checks'/key).glob('attempt-*/complete.json'))
    if len(files)!=1:raise ValueError('Missing/duplicate preflight')
    r=prior.read(files[0]);prior.verify(r);check(p,key,r['actual'],r['brightness_log'],r['gamma_log'])
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
        lookup={m['image_path']:m['member_id'] for m in p['pool_rows']};owner=SimpleNamespace(epoch=0);actual=[];logs=[];gamma_logs=[];batches=[]
        with ExitStack() as stack:
            for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):stack.enter_context(patch.object(obj,name,forbidden))
            loader=make_loader(make_dataset(p,key,logs,gamma_logs),p,key,owner)
            for epoch in range(p['training_config'][key]['epochs']):
                owner.epoch=epoch
                for batch in loader:
                    if tuple(batch['img'].shape)!=(6,3,640,640):raise ValueError('Batch size drift')
                    members=[lookup[x] for x in batch['im_file']];actual.extend(members)
                    batches.append(dict(step=len(batches),members=members,image_tensor_sha256=tensor_hash(batch['img']),full_supervision={f:tensor_hash(batch[f]) for f in ('cls','bboxes','batch_idx')}))
        check(p,key,actual,logs,gamma_logs)
        ref,refpath=base_receipt('B900-'+key.split('-')[-1],base_design())
        if actual!=ref['actual'] or logs!=ref['brightness_log']:raise ValueError('Underlying exposure/brightness drift')
        for i,(a,b) in enumerate(zip(batches,ref['batch_records'],strict=True)):
            if a['members']!=b['members'] or a['full_supervision']!=b['full_supervision']:raise ValueError('Full supervision drift')
            if p['gamma_factors'][key][i]==1. and a!=b:raise ValueError('Identity gamma batch changed')
        deps=[OUT/'design.json',Path(__file__).resolve(),Path(__file__).with_name('closed_gamma_runtime.py'),refpath]
        prior.frozen(attempt/'complete.json',dict(status='actual_loader_verified_no_training',cell=key,actual=actual,brightness_log=logs,batch_records=batches,
            gamma_log=gamma_logs,identity_batches_exact=True,full_supervision_exact=True,
            optimizer_created=False,backward_executed=False,validation_run=False,inputs={str(d):prior.file_sha256(d) for d in deps}))
        print('PREFLIGHT_COMPLETE',key,len(actual),flush=True)
    except BaseException:prior.frozen(attempt/'failure.json',dict(error=traceback.format_exc()));raise

def run():
    p=freeze()
    for i in range(0,len(KEYS),2):
        procs=[]
        try:
            for key in KEYS[i:i+2]:
                proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.preflight_closed_gamma','--worker',key],cwd=prior.ROOT,start_new_session=True)
                procs.append(proc)
            for proc in procs:
                proc.wait(timeout=1200)
                if proc.returncode:raise RuntimeError('Preflight failed')
        finally:
            for proc in procs:cleanup(proc)
    deps=[OUT/'design.json',Path(__file__).resolve()]
    for key in KEYS:
        r,rp=receipt(key,p);deps.append(rp)
        if not r['identity_batches_exact'] or not r['full_supervision_exact']:raise ValueError('Baseline equivalence failed')
    dest=OUT/'loader-completion.json'
    if dest.exists():prior.verify(prior.read(dest));return
    prior.frozen(dest,dict(status='three_gamma_loaders_verified_visual_check_and_entry_pending',checked_exposures=16200,checked_batches=2700,inputs={str(d):prior.file_sha256(d) for d in deps}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=KEYS);a=ap.parse_args()
    if a.worker:worker(a.worker)
    else:run()
