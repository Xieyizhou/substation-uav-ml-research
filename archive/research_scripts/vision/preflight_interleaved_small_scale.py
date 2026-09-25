"""Verify all real reordered tensors against actual historical batches."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.vision.interleaved_small_scale_control import OUT,KEYS,freeze,prior,source
from scripts.vision.closed_budget_runtime import make_dataset,make_loader,check
from scripts.vision.preflight_unified_lighting import tensor_hash
from scripts.vision.infer_material_member_fit import forbidden


def preflight():
    p=freeze(); dest=OUT/'loader-feasibility.json'
    if dest.exists():
        r=prior.read(dest); prior.verify(r); return r
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    torch.set_num_threads(4)
    deps=[OUT/'design.json',Path(__file__).resolve()]
    deps += [Path(__file__).with_name(n+'.py') for n in ('closed_budget_runtime','order_retention_runtime','brightness_transfer_runtime','closed_budget_engine_v2')]
    lookup={r['image_path']:r['member_id'] for r in p['pool_rows']}
    units=[]
    for key in KEYS:
        cp=source.OUT/'training'/key[1:]/'completion.json'; c=prior.read(cp); prior.verify(c)
        xp=Path(c['exposure_path']); old=prior.read(xp); prior.verify(old); deps += [cp,xp]
        owner=SimpleNamespace(epoch=0); logs=[]; actual=[]; batches=[]
        init_seeds(int(key.split('-')[-1]),deterministic=True)
        with ExitStack() as stack:
            for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):
                stack.enter_context(patch.object(obj,name,forbidden))
            loader=make_loader(make_dataset(p,key,logs),p,key,owner)
            for epoch in range(p['training_config'][key]['epochs']):
                owner.epoch=epoch
                for batch in loader:
                    i=len(batches); j=p['batch_permutations'][key][i]
                    members=[lookup[x] for x in batch['im_file']]
                    sha=tensor_hash(batch['img'])
                    labels={f:tensor_hash(batch[f]) for f in ('cls','bboxes','batch_idx')}
                    previous=old['batch_records'][j]
                    if tuple(batch['img'].shape)!=(6,3,640,640) or members!=p['schedules'][key][6*i:6*i+6]:
                        raise ValueError('Actual batch/member drift')
                    if sha!=previous['image_tensor_sha256'] or labels!=previous['full_supervision']:
                        raise ValueError('Historical image/full-label tensor drift')
                    actual.extend(members)
                    batches.append(dict(step=i,members=members,image_tensor_sha256=sha,full_supervision=labels))
        check(p,key,actual,logs)
        if len(batches)!=1100: raise ValueError('Incomplete loader')
        for i,record in enumerate(logs):
            j=6*p['batch_permutations'][key][i//6]+i%6
            expected=dict(old['brightness_log'][j]); expected['position']=i
            if record!=expected: raise ValueError('Brightness pixel log drift')
        units.append(dict(cell=key,actual=actual,brightness_log=logs,batch_records=batches))
        print('ACTUAL_PREFLIGHT_COMPLETE',key,len(actual),flush=True)
    return prior.frozen(dest,dict(status='six_actual_loaders_verified_no_training',units=units,
        optimizer_created=False,backward_executed=False,validation_run=False,
        inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__ == '__main__': print(preflight()['status'])
