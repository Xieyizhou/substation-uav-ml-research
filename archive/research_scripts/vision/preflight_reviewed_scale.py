"""Real complete scale schedules, no optimizer or backward."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.vision.freeze_reviewed_scale_control import OUT,KEYS,freeze,prior
from scripts.vision import reviewed_scale_runtime as runtime
from scripts.vision.preflight_unified_lighting import tensor_hash

def run():
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    p=freeze();torch.set_num_threads(4);root=OUT/'actual-preflight';root.mkdir(exist_ok=True)
    lookup={r['image_path']:r['member_id'] for r in p['pool_rows']}
    for key in KEYS:
        dest=root/(key+'.json')
        if dest.exists():prior.verify(prior.read(dest));print('REUSED',key,flush=True);continue
        log=[];actual=[];batches=[];owner=SimpleNamespace(epoch=0);init_seeds(int(key.split('-')[-1]),deterministic=True)
        def forbidden(*args,**kwargs):raise AssertionError('Training forbidden')
        with ExitStack() as stack:
            for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):stack.enter_context(patch.object(obj,name,forbidden))
            loader=runtime.make_loader(runtime.make_dataset(p,key,log),p,key,owner)
            for epoch in range(110):
                owner.epoch=epoch
                for batch in loader:
                    mids=[lookup[x] for x in batch['im_file']];i=len(batches)
                    if mids!=p['schedules'][key][i*6:i*6+6] or tuple(batch['img'].shape)!=(6,3,640,640):raise ValueError('Batch/order mismatch')
                    runtime.verify_supervision(batch,log[-6:]);actual+=mids
                    batches.append(dict(step=i+1,members=mids,image_tensor_sha256=tensor_hash(batch['img']),full_supervision={k:tensor_hash(batch[k]) for k in ('cls','bboxes','batch_idx')}))
        runtime.check(p,key,actual,log)
        if len(batches)!=1100:raise ValueError('Incomplete preflight')
        deps=[OUT/'design.json',Path(__file__).resolve(),Path(runtime.__file__).resolve()]
        deps += [Path(__file__).with_name(x+'.py') for x in ('closed_budget_runtime','order_retention_runtime','brightness_transfer_runtime','preflight_unified_lighting')]
        prior.frozen(dest,dict(status='actual_complete_loader_verified_no_training',key=key,cell=key,actual=actual,brightness_log=log,batch_records=batches,
            checked_draws=len(actual),checked_batches=len(batches),full_geometry_checked=True,optimizer_created=False,backward_executed=False,validation_run=False,
            inputs={str(x):prior.file_sha256(x) for x in deps}))
        print('PREFLIGHT_COMPLETE',key,flush=True)

if __name__=='__main__':run()
