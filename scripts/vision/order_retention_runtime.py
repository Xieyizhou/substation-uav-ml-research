"""One dataset/loader factory shared by optimizer-free preflight and training."""
from pathlib import Path
import numpy as np
from scripts.vision.exposure_order_retention import file_sha256

def overrides(seed):
    return dict(task='detect',imgsz=640,batch=6,nbs=6,epochs=45,device='cpu',workers=0,optimizer='AdamW',lr0=.001,lrf=1.,
        warmup_epochs=0,warmup_bias_lr=0,seed=seed,deterministic=True,patience=0,amp=False,
        mosaic=0,close_mosaic=0,mixup=0,copy_paste=0,degrees=0,translate=0,scale=0,shear=0,
        perspective=0,flipud=0,fliplr=0,hsv_h=0,hsv_s=0,hsv_v=0,plots=False,save=True,val=False,
        bgr=0,cutmix=0,multi_scale=0,cache=False,rect=False,fraction=1.0)

def validate_labels(dataset,lookup):
    inverse={r['image_path']:r for r in lookup.values()}
    for label in dataset.labels:
        row=inverse[label['im_file']]
        if file_sha256(row['label_path'])!=row['label_sha256']:raise ValueError('Label hash changed')
        text=Path(row['label_path']).read_text().strip()
        expected=np.array([list(map(float,s.split())) for s in text.splitlines()],dtype=np.float32).reshape(-1,5)
        actual=np.concatenate((label['cls'],label['bboxes']),axis=1)
        if expected.shape!=actual.shape or not np.allclose(expected,actual,rtol=0,atol=1e-6):raise ValueError('Full dataset labels changed')

def make_dataset(p,key):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    cfg=get_cfg(overrides=overrides(int(key.split('-')[-1])))
    data={'names':p['names'],'nc':4,'channels':3}
    return build_yolo_dataset(cfg,p['listings'][key],6,data,mode='train',rect=False,stride=32)

def make_loader(dataset,p,key,owner):
    import torch
    from torch.utils.data import Sampler,DataLoader
    rows={r['member_id']:r for r in p['pool_rows']};seq=p['schedules'][key]
    mapping={path:i for i,path in enumerate(dataset.im_files)}
    if len(mapping)!=len(dataset.im_files) or set(mapping)!={rows[m]['image_path'] for m in seq}:raise ValueError('Runtime member set changed')
    validate_labels(dataset,rows)
    class Schedule(Sampler):
        def __len__(self):return 60
        def __iter__(self):
            start=owner.epoch*60
            return iter(mapping[rows[m]['image_path']] for m in seq[start:start+60])
    return DataLoader(dataset,batch_size=6,sampler=Schedule(),num_workers=0,collate_fn=dataset.collate_fn,
        generator=torch.Generator().manual_seed(int(key.split('-')[-1])))

def check_actual(p,key,actual):
    if actual!=p['schedules'][key] or len(actual)!=2700:raise ValueError('Actual sampling mismatch')

def preflight_cell(p,key):
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(int(key.split('-')[-1]),deterministic=True)
    owner=SimpleNamespace(epoch=0);actual=[];inverse={r['image_path']:r['member_id'] for r in p['pool_rows']}
    # Fail loudly if future refactors accidentally enter any optimizer/backprop path.
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('Optimizer forbidden in preflight')), \
         patch.object(torch.Tensor,'backward',side_effect=AssertionError('Backward forbidden in preflight')):
        dataset=make_dataset(p,key);loader=make_loader(dataset,p,key,owner)
        for epoch in range(45):
            owner.epoch=epoch
            for batch in loader:
                if tuple(batch['img'].shape)!=(6,3,640,640):raise ValueError('Unexpected actual image tensor shape')
                actual.extend(inverse[path] for path in batch['im_file'])
    check_actual(p,key,actual)
    return dict(cell=key,checked_draws=2700,checked_batches=450,checked_members=len(dataset.im_files),
        actual=actual,optimizer_created=False,backward_executed=False,validation_run=False)
