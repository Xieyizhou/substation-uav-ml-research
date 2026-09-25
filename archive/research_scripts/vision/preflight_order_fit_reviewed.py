"""Real complete schedules; optimizer, backward and training validation forbidden."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.vision.order_fit_reviewed_schedule import run,DEST,prior
from scripts.vision.closed_budget_runtime import make_dataset,make_loader,check
from scripts.vision.preflight_unified_lighting import tensor_hash


def preflight():
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    p=run();torch.set_num_threads(4);root=DEST/'actual-preflight';root.mkdir(exist_ok=True)
    lookup={r['image_path']:r['member_id'] for r in p['pool_rows']}
    for key in sorted(p['schedules']):
        done=root/(key+'.json')
        if done.exists():r=prior.read(done);prior.verify(r);print('REUSED',key,flush=True);continue
        actual=[];logs=[];batches=[];owner=SimpleNamespace(epoch=0)
        init_seeds(int(key.split('-')[-1]),deterministic=True)
        def forbidden(*args,**kwargs):raise AssertionError('Training forbidden in preflight')
        with ExitStack() as stack:
            for obj,name in [(torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')]:
                stack.enter_context(patch.object(obj,name,forbidden))
            loader=make_loader(make_dataset(p,key,logs),p,key,owner)
            for epoch in range(p['training_config'][key]['epochs']):
                owner.epoch=epoch
                for batch in loader:
                    mids=[lookup[x] for x in batch['im_file']];i=len(batches)
                    if mids!=p['schedules'][key][i*6:i*6+6] or tuple(batch['img'].shape)!=(6,3,640,640):raise ValueError('Actual batch/order mismatch')
                    actual+=mids;batches.append(dict(step=i+1,members=mids,image_tensor_sha256=tensor_hash(batch['img']),
                        full_supervision={k:tensor_hash(batch[k]) for k in ('cls','bboxes','batch_idx')}))
        check(p,key,actual,logs)
        if len(batches)!=1100:raise ValueError('Incomplete optimization-step equivalent')
        paths=[DEST/'design.json',Path(__file__).resolve()]
        paths += [Path(__file__).with_name(n+'.py') for n in ('closed_budget_runtime','order_retention_runtime','brightness_transfer_runtime','preflight_unified_lighting')]
        prior.frozen(done,dict(status='actual_complete_loader_verified_no_training',key=key,actual=actual,
            brightness_log=logs,batch_records=batches,checked_draws=len(actual),checked_batches=len(batches),
            optimizer_created=False,backward_executed=False,validation_run=False,
            inputs={str(path):prior.file_sha256(path) for path in paths}))
        print('PREFLIGHT_COMPLETE',key,len(actual),flush=True)


if __name__=='__main__':preflight()
