from scripts.vision.il_lower_rate_control import OUT,REF,KEYS,freeze,prior
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
from scripts.vision.closed_budget_runtime import make_dataset,make_loader,check
from scripts.vision.preflight_unified_lighting import tensor_hash
from scripts.vision.infer_material_member_fit import forbidden

def preflight():
    p=freeze();dest=OUT/'loader-feasibility.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    torch.set_num_threads(4);units=[];deps=[OUT/'design.json',Path(__file__).resolve()]
    lookup={r['image_path']:r['member_id'] for r in p['pool_rows']}
    for key in KEYS:
        seed=key.split('-')[-1];cp=REF/'training'/('IL1000-'+seed)/'completion.json';c=prior.read(cp);prior.verify(c)
        xp=Path(c['exposure_path']);old=prior.read(xp);prior.verify(old);deps.extend([cp,xp])
        logs=[];actual=[];batches=[];owner=SimpleNamespace(epoch=0);init_seeds(int(seed),deterministic=True)
        with ExitStack() as stack:
            for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):stack.enter_context(patch.object(obj,name,forbidden))
            loader=make_loader(make_dataset(p,key,logs),p,key,owner)
            for epoch in range(p['training_config'][key]['epochs']):
                owner.epoch=epoch
                for batch in loader:
                    i=len(batches);expected=old['batch_records'][i]
                    members=[lookup[x] for x in batch['im_file']];sha=tensor_hash(batch['img']);labels={f:tensor_hash(batch[f]) for f in ('cls','bboxes','batch_idx')}
                    if tuple(batch['img'].shape)!=(6,3,640,640) or members!=expected['members'] or sha!=expected['image_tensor_sha256'] or labels!=expected['full_supervision']:raise ValueError('Actual tensors differ')
                    actual.extend(members);batches.append(dict(step=i,members=members,image_tensor_sha256=sha,full_supervision=labels))
        check(p,key,actual,logs)
        if logs!=old['brightness_log']:raise ValueError('Actual augmentation changed')
        if len(batches)!=1000:raise ValueError('Incomplete loader')
        units.append(dict(cell=key,actual=actual,brightness_log=logs,batch_records=batches))
        print('ACTUAL_PREFLIGHT_COMPLETE',key,len(actual),flush=True)
    return prior.frozen(dest,dict(status='three_actual_loaders_verified_no_training',units=units,optimizer_created=False,backward_executed=False,validation_run=False,inputs={str(d):prior.file_sha256(d) for d in deps}))

if __name__=='__main__':preflight()
