"""Preserve an already frozen baseline's effective within-batch order."""
from collections import Counter

def align_to_reference(requested,reference,batch_size=6):
    if batch_size<=0 or len(requested)!=len(reference) or len(reference)%batch_size:
        raise ValueError('Complete equally sized batch sequences required')
    if any(not isinstance(mid,str) or not mid for mid in requested+reference):
        raise ValueError('Stable member IDs required')
    for i in range(0,len(reference),batch_size):
        if Counter(requested[i:i+batch_size])!=Counter(reference[i:i+batch_size]):
            raise ValueError('Batch identity changed; cannot normalize a regrouping experiment')
    return list(reference)
