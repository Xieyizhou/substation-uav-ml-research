"""Candidate-only loader check; class names come from the frozen design.

Supersedes the unexecuted staging helper in order_fit_closeout_batch, whose
hard-coded class order is incorrect. Its signed inventory remains unchanged.
"""
from contextlib import ExitStack
from pathlib import Path
import shutil
from unittest.mock import patch
from scripts.vision.diagnose_small_scale_order_fit import OUT, TRAIN, prior
from scripts.vision.order_fit_closeout_batch import inventory


def run():
    import torch
    import numpy as np
    from ultralytics import YOLO, __version__ as ultra_version
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from scripts.vision.order_retention_runtime import overrides, validate_labels
    from scripts.vision.preflight_unified_lighting import tensor_hash
    inv=inventory();p=prior.read(OUT/'protocol.json');design=prior.read(TRAIN/'design.json')
    prior.verify(p);prior.verify(design)
    names=design['names']
    selected={r['member_id'] for r in inv['members'] if r['stage']=='quality_supported_source_role_gate_pending'}
    members=sorted((m for m in p['members'] if m['member_id'] in selected),key=lambda m:m['member_id'])
    root=OUT/'candidate-loader-v2';root.mkdir(exist_ok=True)
    for number in range(1,4):
        attempt=root/f'attempt-{number:02d}';done=attempt/'completion.json'
        if done.exists():
            r=prior.read(done)
            try:prior.verify(r);return r
            except (ValueError,FileNotFoundError):continue
        if attempt.exists():continue
        attempt.mkdir();break
    else:raise ValueError('Three loader attempts exhausted')
    deps=[OUT/'protocol.json',OUT/'closeout-inventory-v2.json',TRAIN/'design.json',Path(__file__).resolve(),
          Path(__file__).with_name('order_retention_runtime.py'),Path(__file__).with_name('preflight_unified_lighting.py')]
    try:
        images=attempt/'images';labels=attempt/'labels';images.mkdir();labels.mkdir();rows=[]
        for i,m in enumerate(members):
            r=dict(m)
            for key,folder,suffix in [('image',images,'.png'),('label',labels,'.txt')]:
                original=Path(m[key+'_path']);dest=folder/(f'{i:04d}'+suffix)
                if prior.file_sha256(original)!=m[key+'_sha256']:raise ValueError('Source drift before export')
                shutil.copyfile(original,dest)
                if prior.file_sha256(dest)!=m[key+'_sha256']:raise ValueError('Export changed source bytes')
                r[key+'_path']=str(dest);deps.extend([original,dest])
            rows.append(r)
        torch.set_num_threads(4)
        lookup={r['member_id']:r for r in rows};inverse={r['image_path']:r for r in rows}
        config=get_cfg(overrides=overrides(7));actual=[];batches=[]
        def forbidden(*args,**kwargs):raise AssertionError('Training forbidden in candidate preflight')
        with ExitStack() as stack:
            for obj,name in [(torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')]:
                stack.enter_context(patch.object(obj,name,forbidden))
            dataset=build_yolo_dataset(config,str(images),6,{'names':names,'nc':len(names),'channels':3},mode='train',rect=False,stride=32)
            validate_labels(dataset,lookup)
            if list(dataset.im_files)!=[r['image_path'] for r in rows]:raise ValueError('Loader member order drift')
            loader=torch.utils.data.DataLoader(dataset,batch_size=6,shuffle=False,num_workers=0,collate_fn=dataset.collate_fn)
            for batch in loader:
                ids=[inverse[path]['member_id'] for path in batch['im_file']];n=len(ids)
                if tuple(batch['img'].shape)!=(n,3,640,640):raise ValueError('Bad image shape')
                for i,mid in enumerate(ids):
                    item=dataset.labels[dataset.im_files.index(lookup[mid]['image_path'])]
                    mask=batch['batch_idx']==i
                    actual_cls=batch['cls'][mask].numpy();actual_boxes=batch['bboxes'][mask].numpy()
                    # Training transforms include letterbox; normalized original boxes must
                    # match the independently loaded item, not unresized source coordinates.
                    if len(actual_cls)!=len(item['cls']):raise ValueError('Label lost during transforms')
                    from collections import Counter
                    if Counter(actual_cls.flatten())!=Counter(item['cls'].flatten()):raise ValueError('Class changed during transforms')
                    if not np.isfinite(actual_boxes).all():raise ValueError('Non-finite transformed box')
                    if Counter(names[int(c)] for c in actual_cls.flatten())!=Counter(lookup[mid]['class_instances']):
                        raise ValueError('Frozen class identity drift')
                actual.extend(ids)
                batches.append(dict(members=ids,image_tensor_sha256=tensor_hash(batch['img']),
                    full_supervision={k:tensor_hash(batch[k]) for k in ('cls','bboxes','batch_idx')}))
        if actual!=[r['member_id'] for r in rows]:raise ValueError('Incomplete loader traversal')
        return prior.frozen(done,dict(status='candidate_bytes_and_loader_verified_not_training_ready',members=rows,
            checked_members=len(rows),checked_batches=len(batches),batches=batches,class_names=names,
            optimizer_created=False,backward_executed=False,validation_run=False,training_started=False,
            scope='One traversal without brightness schedule; not final exposure/augmentation preflight.',
            supersession='Unexecuted v1 staging helper had incorrect hard-coded class names; this entry consumes frozen names and verifies per-member named instance counts.',
            environment={'torch':torch.__version__,'ultralytics':ultra_version},inputs={str(p):prior.file_sha256(p) for p in deps}))
    except BaseException as exc:
        prior.frozen(attempt/'failure.json',dict(status='candidate_loader_failed',error=repr(exc),training_started=False))
        raise


if __name__=='__main__':print(run()['status'])
