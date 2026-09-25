"""Verify an executable batch-order-only design; this module cannot train."""
import copy
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.vision.diagnose_bn_statistics import SOURCE,prior
from scripts.vision.closed_budget_runtime import make_dataset,make_loader,check
from scripts.vision.preflight_unified_lighting import tensor_hash
from scripts.vision.infer_material_member_fit import forbidden
OUT=SOURCE/'tail-interleaving-feasibility-v1'

def permutation():
    # Stable internal order of both lists; one original tail batch every ten steps.
    old=iter(range(900));tail=iter(range(900,1000))
    return [next(tail) if (i+1)%10==0 else next(old) for i in range(1000)]

def main():
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    torch.set_num_threads(4)
    dp=SOURCE/'design.json';original=prior.read(dp);prior.verify(original)
    p=copy.deepcopy(original);perm=permutation()
    if sorted(perm)!=list(range(1000)):raise ValueError('Batch omission/repetition')
    deps=[dp,Path(__file__).resolve()];results=[]
    for seed in (7,17,27):
        oldkey=f'R1000-{seed}';key=f'I1000-{seed}'
        cp=SOURCE/'training'/oldkey/'completion.json';c=prior.read(cp);prior.verify(c)
        xp=Path(c['exposure_path']);x=prior.read(xp);prior.verify(x);deps.extend([cp,xp])
        p['schedules'][key]=[m for i in perm for m in original['schedules'][oldkey][i*6:i*6+6]]
        p['brightness_factors'][key]=[g for i in perm for g in original['brightness_factors'][oldkey][i*6:i*6+6]]
        p['listings'][key]=original['listings'][oldkey];p['training_config'][key]=original['training_config'][oldkey]
        seq=p['schedules'][key]
        if Counter(seq)!=Counter(x['actual']):raise ValueError('Exposure multiset changed')
        init_seeds(seed,deterministic=True);owner=SimpleNamespace(epoch=0);logs=[];actual=[];batches=[]
        lookup={r['image_path']:r['member_id'] for r in p['pool_rows']}
        with ExitStack() as stack:
            for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):stack.enter_context(patch.object(obj,name,forbidden))
            loader=make_loader(make_dataset(p,key,logs),p,key,owner)
            for epoch in range(p['training_config'][key]['epochs']):
                owner.epoch=epoch
                for batch in loader:
                    idx=len(batches);expected=x['batch_records'][perm[idx]]
                    members=[lookup[f] for f in batch['im_file']]
                    if members!=expected['members'] or tuple(batch['img'].shape)!=(6,3,640,640):raise ValueError('Actual members/batch changed')
                    image_hash=tensor_hash(batch['img']);labels={f:tensor_hash(batch[f]) for f in ('cls','bboxes','batch_idx')}
                    if image_hash!=expected['image_tensor_sha256'] or labels!=expected['full_supervision']:raise ValueError('Actual full tensor changed')
                    actual.extend(members);batches.append(dict(step=idx,source_batch=perm[idx],members=members,image_tensor_sha256=image_hash,full_supervision=labels))
        check(p,key,actual,logs)
        if len(batches)!=1000:raise ValueError('Incomplete traversal')
        for j,r in enumerate(logs):
            old=x['brightness_log'][perm[j//6]*6+j%6]
            if {k:v for k,v in r.items() if k!='position'}!={k:v for k,v in old.items() if k!='position'}:raise ValueError('Brightness pair changed')
        rows={r['member_id']:r for r in p['pool_rows']}
        windows=[]
        for start in range(0,1000,50):
            part=actual[start*6:(start+50)*6]
            windows.append(dict(first_step=start+1,tail_batches=sum(i>=900 for i in perm[start:start+50]),subsets=dict(Counter(rows[m]['subset'] for m in part)),class_instances={c:sum(rows[m]['class_instances'].get(c,0) for m in part) for c in ('transformer','switchgear','capacitor_bank','reactor')}))
        results.append(dict(cell=key,actual=actual,brightness_log=logs,batch_records=batches,windows=windows))
        print(key,'6000 exposures exact',flush=True)
    deps.extend(Path(__file__).with_name(n) for n in ('closed_budget_runtime.py','order_retention_runtime.py','brightness_transfer_runtime.py'))
    return prior.frozen(OUT/'loader-feasibility.json',dict(status='actual_loader_feasible_error_review_and_training_entry_pending',permutation=perm,units=results,
        optimizer_created=False,backward_executed=False,validation_run=False,training_started=False,
        constraints='Same image, complete label, brightness and intact batch multiset. Negative exposure counts preserved, positions deliberately change. No 900-step prefix reuse.',
        inputs={str(d):prior.file_sha256(d) for d in deps}))

if __name__=='__main__':print(main()['status'])
