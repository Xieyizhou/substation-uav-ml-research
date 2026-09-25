"""Actual paired loaders and fixed brightness; optimizer/backprop forbidden."""
import traceback
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.vision.freeze_gray_body_control import OUT,prior,freeze,source
from scripts.vision.brightness_transfer_runtime import make_dataset,check_log
from scripts.vision.order_retention_runtime import make_loader,check_actual
from scripts.vision.preflight_compensated_double_dose import tensor_hash
from scripts.vision.infer_material_member_fit import forbidden


def run():
    p=freeze();lookup={r['image_path']:r for r in p['pool_rows']};paths=[OUT/'protocol.json',Path(__file__).resolve()]
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    torch.set_num_threads(4)
    for seed in (7,17,27):
        root=OUT/'loader-checks'/str(seed);root.mkdir(parents=True,exist_ok=True)
        complete=list(root.glob('attempt-*/complete.json'))
        if complete:
            if len(complete)!=1:raise ValueError('Ambiguous completion')
            r=prior.read(complete[0]);prior.verify(r)
            for key,cell in r['cells'].items():check_actual(p,key,cell['actual']);check_log(p,key,cell['brightness_log'])
            paths+=complete;continue
        n=len(list(root.glob('attempt-*')))+1
        if n>3:raise ValueError('Attempt cap')
        attempt=root/f'attempt-{n:02}';attempt.mkdir()
        try:
            _,_,historical=source.contract(f'VM-{seed}')
            keys=[f'R-{seed}',f'G-{seed}'];logs={k:[] for k in keys};actual={k:[] for k in keys};batches={k:[] for k in keys}
            owner=SimpleNamespace(epoch=0);init_seeds(seed,deterministic=True);step=0
            with ExitStack() as stack:
                for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):
                    stack.enter_context(patch.object(obj,name,forbidden))
                loaders=[make_loader(make_dataset(p,k,logs[k]),p,k,owner) for k in keys]
                for epoch in range(45):
                    owner.epoch=epoch
                    for a,b in zip(*loaders,strict=True):
                        for field in ('cls','bboxes','batch_idx'):
                            if not torch.equal(a[field],b[field]):raise ValueError('Paired full supervision changed')
                        for j,(pa,pb) in enumerate(zip(a['im_file'],b['im_file'],strict=True)):
                            if pa==pb and not torch.equal(a['img'][j],b['img'][j]):raise ValueError('Untreated tensor changed')
                        for key,batch in zip(keys,(a,b),strict=True):
                            if tuple(batch['img'].shape)!=(6,3,640,640):raise ValueError('Tensor shape changed')
                            mids=[lookup[x]['member_id'] for x in batch['im_file']];actual[key]+=mids
                            batches[key].append(dict(step=step,members=mids,tensor_sha256=tensor_hash(batch['img']),
                                supervision={f:tensor_hash(batch[f]) for f in ('cls','bboxes','batch_idx')}))
                        step+=1
                    if epoch%10==0:print(seed,'loaded_steps',step,flush=True)
            if step!=450:raise ValueError('Incomplete steps')
            for k in keys:check_actual(p,k,actual[k]);check_log(p,k,logs[k])
            if logs[keys[0]]!=historical['brightness_log']:raise ValueError('Reference actual brightness/tensors differ from history')
            for i,(a,b) in enumerate(zip(logs[keys[0]],logs[keys[1]],strict=True)):
                if a['gain']!=b['gain']:raise ValueError('Gain changed')
                if i not in p['replacement_positions'][str(seed)] and a!=b:raise ValueError('Non-treatment exposure drift')
            dest=attempt/'complete.json'
            prior.frozen(dest,dict(status='actual_paired_loaders_verified',seed=seed,
                cells={k:dict(actual=actual[k],brightness_log=logs[k],batch_records=batches[k]) for k in keys},
                optimizer_created=False,backward_executed=False,training_validation_run=False,
                inputs={str(x):prior.file_sha256(x) for x in paths[:2]}));paths.append(dest)
            print('COMPLETE',seed,'5400 actual loads',flush=True)
        except BaseException:
            prior.frozen(attempt/'failure.json',dict(status='failed',reason=traceback.format_exc(),training_started=False));raise
    dest=OUT/'loader-completion.json'
    if dest.exists():prior.verify(prior.read(dest));return
    prior.frozen(dest,dict(status='six_actual_loaders_verified_final_audit_pending',training_ready=False,training_started=False,
        inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':run()
