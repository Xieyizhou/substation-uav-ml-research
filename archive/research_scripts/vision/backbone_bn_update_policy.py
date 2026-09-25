"""Separate frozen backbone parameters from explicitly mutable BN buffers."""
import hashlib
import torch
from scripts.vision.routed_backbone_control import frozen_name, copy_frozen_to_ema

VERSION='frozen-backbone-parameters-live-bn-copy-to-ema-v1'

def mutable_names(model):
    names=set()
    for name,module in model.named_modules():
        if frozen_name(name) and isinstance(module,torch.nn.BatchNorm2d):
            if not module.track_running_stats:raise ValueError('BN lacks tracked statistics')
            names.update(name+'.'+suffix for suffix in ('running_mean','running_var','num_batches_tracked'))
    if not names:raise ValueError('No backbone BN buffers')
    return names

def digest(model,mutable=False,half=False):
    allowed=mutable_names(model);h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if not frozen_name(name) or ((name in allowed)!=mutable):continue
        t=tensor.detach().cpu().contiguous()
        if half and t.is_floating_point():t=t.half()
        h.update(f'{name}|{t.dtype}|{tuple(t.shape)}'.encode());h.update(t.numpy().tobytes())
    return h.hexdigest()

def activate(model):
    # Call after the trainer's normal _model_train, which freezes these BNs.
    for name,module in model.named_modules():
        if frozen_name(name) and isinstance(module,torch.nn.BatchNorm2d):module.train()
    return assert_policy(model)

def assert_policy(model):
    frozen=other=0
    for name,p in model.named_parameters():
        expected=not(frozen_name(name) or '.dfl' in name)
        if p.requires_grad!=expected:raise ValueError('Parameter partition changed '+name)
        if frozen_name(name):
            frozen+=p.numel()
            if p.grad is not None:raise ValueError('Frozen parameter gradient '+name)
        elif expected:other+=p.numel()
    for name,module in model.named_modules():
        if frozen_name(name) and isinstance(module,torch.nn.BatchNorm2d) and not module.training:
            raise ValueError('Backbone BN silently in eval '+name)
    if not frozen or not other:raise ValueError('Empty partition')
    return dict(frozen_parameters=frozen,trainable_parameters=other,mutable_buffers=sorted(mutable_names(model)))

def sync_ema(model,ema):
    # Same exact-copy policy as the previous frozen arm, now including live BN.
    # Neck/head keep library EMA; no alternative EMA selected after results.
    copy_frozen_to_ema(model,ema)
    if digest(model)!=digest(ema) or digest(model,True)!=digest(ema,True):
        raise ValueError('EMA copy mismatch')
