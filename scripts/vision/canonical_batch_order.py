"""Development sampler policy: stable member order inside each existing batch.

Apply BEFORE dataset fetching/collation and log requested and effective sequences.
This deliberately does not reorder whole batches or change member exposures.
"""
from collections import Counter

def canonicalize(draws,batch_size=6):
    if batch_size<=0 or len(draws)%batch_size:raise ValueError('Complete batches required')
    if any(not isinstance(mid,str) or not mid for mid in draws):raise ValueError('Stable nonempty member IDs required')
    result=[]
    for i in range(0,len(draws),batch_size):
        result.extend(sorted(draws[i:i+batch_size]))
    if Counter(result)!=Counter(draws):raise ValueError('Member exposures changed')
    return result
