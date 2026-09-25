"""Frozen per-exposure V-only augmentation; no global RNG consumption."""
import hashlib
from pathlib import Path
import cv2
import numpy as np
from scripts.vision import order_retention_runtime as base

VERSION='brightness-transfer-control-v1'

def factor(seed,member,position):
    b=hashlib.sha256(f'{VERSION}:{seed}:{member}:{position}'.encode()).digest()
    return 1.0 if b[0]%2==0 else .8+.4*int.from_bytes(b[1:9],'big')/(2**64-1)

def brighten(img,gain):
    if not .8<=gain<=1.2:raise ValueError('Brightness outside frozen bounds')
    if img.dtype!=np.uint8 or img.ndim!=3 or img.shape[-1]!=3:raise ValueError('Expected uint8 BGR')
    if gain==1:return img.copy()
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hsv[:,:,2]=cv2.LUT(hsv[:,:,2],np.clip(np.arange(256,dtype=np.float64)*gain,0,255).astype(np.uint8))
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

def pixel_hash(img):return hashlib.sha256(str(img.shape).encode()+img.tobytes()).hexdigest()

class Brightness:
    def __init__(self,p,key,log):self.p=p;self.key=key;self.log=log;self.n=0;self.idx={r['image_path']:r['member_id'] for r in p['pool_rows']}
    def __call__(self,labels):
        if self.n>=2700:raise ValueError('Unexpected extra training exposure')
        member=self.idx[labels['im_file']]
        if member!=self.p['schedules'][self.key][self.n]:raise ValueError('Brightness position/member mismatch')
        gain=self.p['brightness_factors'][self.key][self.n];before=labels['img'];after=brighten(before,gain)
        self.log.append(dict(position=self.n,member_id=member,gain=gain,before=pixel_hash(before),after=pixel_hash(after)))
        labels['img']=after;self.n+=1;return labels

def make_dataset(p,key,log):
    d=base.make_dataset(p,key)
    # Detect environment changes that could activate unplanned augmentation.
    for t in d.transforms.transforms:
        if type(t).__name__=='Albumentations' and t.transform is not None:raise ValueError('Unplanned Albumentations enabled')
    d.transforms.insert(0,Brightness(p,key,log));return d

def check_log(p,key,log):
    if len(log)!=2700:raise ValueError('Incomplete brightness exposure ledger')
    for i,r in enumerate(log):
        if r['position']!=i or r['member_id']!=p['schedules'][key][i] or r['gain']!=p['brightness_factors'][key][i]:raise ValueError('Brightness actual/plan mismatch')
        if r['gain']==1 and r['before']!=r['after']:raise ValueError('Identity brightness changed pixels')

def preflight(p,seed):
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    keys=[f'noaug-450-{seed}',f'brightness-450-{seed}'];logs={k:[] for k in keys};owner=SimpleNamespace(epoch=0)
    init_seeds(seed,deterministic=True)
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('No optimizer')),patch.object(torch.Tensor,'backward',side_effect=AssertionError('No backward')),patch.object(YOLO,'train',side_effect=AssertionError('No train')),patch.object(YOLO,'val',side_effect=AssertionError('No validation')):
        datasets=[base.make_dataset(p,keys[0])]+[make_dataset(p,k,logs[k]) for k in keys]
        loaders=[base.make_loader(d,p,k,owner) for d,k in zip(datasets,[keys[0]]+keys)]
        actual=[];batches=0
        for epoch in range(45):
            owner.epoch=epoch
            for original,off,on in zip(*loaders,strict=True):
                if original['im_file']!=off['im_file'] or off['im_file']!=on['im_file']:raise ValueError('Paired batch members changed')
                if not torch.equal(original['img'],off['img']):raise ValueError('No-augmentation baseline not byte-identical')
                for name in ('cls','bboxes','batch_idx'):
                    if not torch.equal(original[name],off[name]) or not torch.equal(off[name],on[name]):raise ValueError('Full supervision changed')
                if tuple(on['img'].shape)!=(6,3,640,640):raise ValueError('Image shape changed')
                actual.extend(off['im_file']);batches+=1
    inverse={r['image_path']:r['member_id'] for r in p['pool_rows']};draws=[inverse[x] for x in actual]
    if batches!=450:raise ValueError('Incomplete loader')
    for k in keys:base.check_actual(p,k,draws);check_log(p,k,logs[k])
    if any(a['before']!=b['before'] for a,b in zip(logs[keys[0]],logs[keys[1]],strict=True)):raise ValueError('Pre-brightness pixels not paired')
    return dict(seed=seed,batches=450,draws=draws,logs=logs,baseline_tensor_bytes_identical=True,complete_labels_identical=True,optimizer_created=False,backward_executed=False,validation_run=False)
