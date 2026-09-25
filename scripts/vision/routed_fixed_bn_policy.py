"""Freeze backbone BN buffers only; preserve affine and convolution gradients."""
import hashlib
import torch
from scripts.vision.routed_backbone_control import frozen_name

VERSION='backbone-bn-fixed-buffers-full-parameter-adaptation-v1'

def modules(model):
    return [(n,m) for n,m in model.named_modules()
            if frozen_name(n) and isinstance(m,torch.nn.BatchNorm2d)]

def apply(model):
    selected=modules(model)
    if not selected:raise ValueError('No backbone BatchNorm')
    for _,m in selected:m.eval()

def digest(model):
    h=hashlib.sha256()
    for n,m in modules(model):
        for attr in ('running_mean','running_var','num_batches_tracked'):
            t=getattr(m,attr)
            if t is None:raise ValueError('Missing running statistics')
            h.update((n+'.'+attr).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def check(model,initial):
    for name,p in model.named_parameters():
        if p.requires_grad!=('.dfl' not in name):raise ValueError('Unexpected parameter freeze '+name)
    for name,m in model.named_modules():
        if isinstance(m,torch.nn.BatchNorm2d) and m.training==frozen_name(name):
            raise ValueError('BN mode mismatch '+name)
    if digest(model)!=initial:raise ValueError('Backbone BN buffers changed')

def copy_buffers(model,ema):
    target=dict(ema.named_modules())
    with torch.no_grad():
        for name,m in modules(model):
            for attr in ('running_mean','running_var','num_batches_tracked'):
                getattr(target[name],attr).copy_(getattr(m,attr))
