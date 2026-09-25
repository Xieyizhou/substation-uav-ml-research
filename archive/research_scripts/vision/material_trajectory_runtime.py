"""Observational model/EMA capture; never run inference inside training."""
import copy
import hashlib
import pickle
import random
from pathlib import Path


def state_digest(model):
    import torch
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        v=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def rng_digest():
    import numpy as np
    import torch
    # CPU is a frozen constraint, not an implicit GPU RNG assertion.
    return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state(),
                                      torch.get_rng_state().numpy().tobytes()))).hexdigest()


def capture(trainer,path,step):
    import torch
    path=Path(path)
    if path.exists():raise ValueError('Refuse checkpoint overwrite')
    if next(trainer.model.parameters()).device.type!='cpu':raise ValueError('CPU-only protocol')
    if trainer.ema is None:raise ValueError('Missing EMA state')
    before=(state_digest(trainer.model),state_digest(trainer.ema.ema),rng_digest())
    # Save full precision live copies. Conversion/inference is a separate offline task.
    payload=dict(model=copy.deepcopy(trainer.model).cpu(),ema=copy.deepcopy(trainer.ema.ema).cpu(),
                 updates=trainer.ema.updates,train_args=copy.deepcopy(vars(trainer.args)),
                 optimizer_step=step,training_admitted=False,promotable=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,path)
    after=(state_digest(trainer.model),state_digest(trainer.ema.ema),rng_digest())
    if before!=after:raise ValueError('Checkpoint observation changed live state or RNG')
    return dict(step=step,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                raw_state_sha256=before[0],ema_state_sha256=before[1],rng_sha256=before[2],
                observation_state_unchanged=True,training_admitted=False,promotable=False)
