"""Enforce and audit Torch intra-op threads after Ultralytics device selection."""
from contextlib import contextmanager
from unittest.mock import patch

@contextmanager
def locked_threads(count=4):
    import torch
    from ultralytics.utils import torch_utils
    original=torch.set_num_threads
    events=[]
    def enforce(requested):
        original(count)
        actual=torch.get_num_threads()
        events.append(dict(requested=requested,actual=actual))
        if actual!=count:raise RuntimeError('CPU thread contract violated')
    with patch.object(torch_utils,'NUM_THREADS',count),patch.object(torch,'set_num_threads',enforce):
        enforce(count)
        yield events

