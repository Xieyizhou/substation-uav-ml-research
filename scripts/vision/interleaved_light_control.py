"""Frozen low-light vs normal control under the same proven interleaving."""
import argparse,copy
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.vision.train_tail_interleaving import OUT as REF,SOURCE,prior
from scripts.vision.preflight_tail_interleaving import permutation
from scripts.vision.evaluate_tail_interleaving import complete as reference_complete
from scripts.vision.physical_low_light_design import freeze as source_freeze
from scripts.vision.record_physical_low_light_review import main as quality
from scripts.vision.closed_budget_runtime import make_dataset,make_loader,check
from scripts.vision.preflight_unified_lighting import tensor_hash
from scripts.vision.infer_material_member_fit import forbidden
OUT=REF/'interleaved-low-light-control-v1'
KEYS=tuple(f'IL1000-{s}' for s in (7,17,27))

def constraints(p,old,ref):
    rows={r['member_id']:r for r in p['pool_rows']};perm=permutation()
    for key in KEYS:
        seed=key.split('-')[-1];lk='L1000-'+seed;rk='I1000-'+seed
        expected=[m for i in perm for m in old['schedules'][lk][i*6:i*6+6]]
        seq=p['schedules'][key];base=ref['schedules'][rk]
        if seq!=expected or len(seq)!=6000:raise ValueError('Sequence drift')
        if p['brightness_factors'][key]!=ref['brightness_factors'][rk] or p['training_config'][key]!=ref['training_config'][rk]:raise ValueError('Non-lighting configuration drift')
        if sum(a!=b for a,b in zip(seq,base))!=300:raise ValueError('Treatment dose drift')
        changed_batches=set()
        for i,(a,b) in enumerate(zip(seq,base)):
            if a in p['held_members']:raise ValueError('Held member restored')
            if rows[a]['subset']=='hard_negative' or rows[b]['subset']=='hard_negative':
                if a!=b:raise ValueError('Negative position changed')
            if rows[a]['class_instances']!=rows[b]['class_instances'] or Path(rows[a]['label_path']).read_bytes()!=Path(rows[b]['label_path']).read_bytes():raise ValueError('Full supervision changed')
            if rows[a]['lineage_id']!=rows[b]['lineage_id']:raise ValueError('Lineage changed')
            if a!=b:
                if rows[a].get('source_member_id')!=b:raise ValueError('Not same-source counterpart')
                changed_batches.add(i//6)
        if len(changed_batches)!=50 or any(sum(seq[j]!=base[j] for j in range(i*6,i*6+6))!=6 for i in changed_batches):raise ValueError('Batch treatment split')

def freeze():
    old=source_freeze();quality()
    ref=prior.read(REF/'design.json');prior.verify(ref)
    paths=[SOURCE/'design.json',SOURCE/'quality-review.json',SOURCE/'export/manifest.json',REF/'design.json',REF/'evaluation/error-review-v1/completion.json',REF/'pool-fit-diagnosis-v1/summary.json',Path(__file__).resolve()]
    for path in paths: 
        if path.suffix=='.json':prior.verify(prior.read(path))
    dest=OUT/'design.json'
    if dest.exists():p=prior.read(dest);prior.verify(p);constraints(p,old,ref);return p
    p=copy.deepcopy(old);p.pop('identity',None);OUT.mkdir(exist_ok=True)
    for key in KEYS:
        seed=key.split('-')[-1];reference_complete('I1000-'+seed);lk='L1000-'+seed
        p['schedules'][key]=[m for i in permutation() for m in old['schedules'][lk][i*6:i*6+6]]
        p['brightness_factors'][key]=[g for i in permutation() for g in old['brightness_factors'][lk][i*6:i*6+6]]
        p['training_config'][key]=old['training_config'][lk];p['listings'][key]=old['listings'][lk]
        paths.append(REF/'training'/('I1000-'+seed)/'completion.json')
    constraints(p,old,ref)
    p.update(status='interleaved_physical_light_control_frozen',comparison='IL1000 vs I1000: 50 intact batches replaced by reviewed same-source physical low-light counterparts; identical interleaved positions, full supervision, source lineage, brightness factors, negative positions and total budget.',limitations=['8 source poses with shared assets; not independent scenes.','E31 development content remains unknown and cannot establish clear-target regression.','No new data admission, threshold changes or unsealed test.'],inputs={**old['inputs'],**{str(d):prior.file_sha256(d) for d in paths}})
    return prior.frozen(dest,p)

def preflight():
    p=freeze();dest=OUT/'loader-feasibility.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    torch.set_num_threads(4);units=[];deps=[OUT/'design.json',Path(__file__).resolve()]
    lookup={r['image_path']:r['member_id'] for r in p['pool_rows']}
    for key in KEYS:
        seed=key.split('-')[-1];cp=SOURCE/'training'/('L1000-'+seed)/'completion.json';c=prior.read(cp);prior.verify(c)
        xp=Path(c['exposure_path']);old=prior.read(xp);prior.verify(old);deps.extend([cp,xp])
        logs=[];actual=[];batches=[];owner=SimpleNamespace(epoch=0);init_seeds(int(seed),deterministic=True)
        with ExitStack() as stack:
            for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):stack.enter_context(patch.object(obj,name,forbidden))
            loader=make_loader(make_dataset(p,key,logs),p,key,owner)
            for epoch in range(p['training_config'][key]['epochs']):
                owner.epoch=epoch
                for batch in loader:
                    i=len(batches);expected=old['batch_records'][permutation()[i]]
                    members=[lookup[x] for x in batch['im_file']];sha=tensor_hash(batch['img']);labels={f:tensor_hash(batch[f]) for f in ('cls','bboxes','batch_idx')}
                    if tuple(batch['img'].shape)!=(6,3,640,640) or members!=expected['members'] or sha!=expected['image_tensor_sha256'] or labels!=expected['full_supervision']:raise ValueError('Actual tensors differ')
                    actual.extend(members);batches.append(dict(step=i,members=members,image_tensor_sha256=sha,full_supervision=labels))
        check(p,key,actual,logs)
        if len(batches)!=1000:raise ValueError('Incomplete loader')
        units.append(dict(cell=key,actual=actual,brightness_log=logs,batch_records=batches))
        print('ACTUAL_PREFLIGHT_COMPLETE',key,len(actual),flush=True)
    return prior.frozen(dest,dict(status='three_actual_loaders_verified_no_training',units=units,optimizer_created=False,backward_executed=False,validation_run=False,inputs={str(d):prior.file_sha256(d) for d in deps}))

if __name__=='__main__':preflight()
