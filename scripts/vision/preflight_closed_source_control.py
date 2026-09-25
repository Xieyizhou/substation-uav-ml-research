"""Traverse six actual loaders; no optimizer, backprop or validation."""
import traceback
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from collections import Counter
from scripts.vision.freeze_material_late_rehearsal import OUT,SOURCE,prior,main as feasibility
from scripts.vision.freeze_material_retention_coverage import freeze as original
from scripts.vision.brightness_transfer_runtime import make_dataset,check_log
from scripts.vision.order_retention_runtime import make_loader,check_actual
from scripts.vision.preflight_unified_lighting import tensor_hash
from scripts.vision.infer_material_member_fit import forbidden

from scripts.vision.freeze_closed_source_control import OUT,KEYS,protocol

def run():
    p=protocol();lookup={r['image_path']:r['member_id'] for r in p['pool_rows']}
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    torch.set_num_threads(4)
    for key in KEYS:
        root=OUT/'loader-checks'/key;root.mkdir(parents=True,exist_ok=True)
        done=list(root.glob('attempt-*/complete.json'))
        if done:
            if len(done)!=1:raise ValueError('Duplicate preflight')
            r=prior.read(done[0]);prior.verify(r)
            check_actual(p,key,r['actual']);check_log(p,key,r['brightness_log']);continue
        n=len(list(root.glob('attempt-*')))+1
        if n>3:raise ValueError('Attempt cap')
        attempt=root/f'attempt-{n:03}';attempt.mkdir()
        try:
            logs=[];actual=[];batches=[];owner=SimpleNamespace(epoch=0)
            init_seeds(int(key.split('-')[-1]),deterministic=True)
            with ExitStack() as stack:
                for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):
                    stack.enter_context(patch.object(obj,name,forbidden))
                loader=make_loader(make_dataset(p,key,logs),p,key,owner)
                for epoch in range(45):
                    owner.epoch=epoch
                    for batch in loader:
                        if tuple(batch['img'].shape)!=(6,3,640,640):raise ValueError('Shape drift')
                        mids=[lookup[x] for x in batch['im_file']];actual.extend(mids)
                        batches.append(dict(step=len(batches),members=mids,image_tensor_sha256=tensor_hash(batch['img']),
                            full_supervision={f:tensor_hash(batch[f]) for f in ('cls','bboxes','batch_idx')}))
                    if epoch%10==0:print(key,'verified steps',len(batches),flush=True)
            check_actual(p,key,actual);check_log(p,key,logs)
            if len(batches)!=450:raise ValueError('Incomplete batches')
            prior.frozen(attempt/'complete.json',dict(status='actual_loader_verified_no_training',cell=key,
                actual=actual,batch_records=batches,brightness_log=logs,optimizer_created=False,
                backward_executed=False,training_validation_executed=False,
                inputs={str(OUT/'protocol.json'):prior.file_sha256(OUT/'protocol.json'),str(Path(__file__).resolve()):prior.file_sha256(__file__)}))
            print('COMPLETE',key,flush=True)
        except BaseException:
            prior.frozen(attempt/'failure.json',dict(status='failed',error=traceback.format_exc(),child_processes_started=0));raise

if __name__=='__main__':run()
