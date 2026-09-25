"""Batch inventory, conservative risk closure, and optimizer-free staging check.

Staging is not permission to train: source-role/full-scene gates remain explicit.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
import shutil
from unittest.mock import patch
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior


def close_risks(members, relations, initial):
    ids={m['member_id'] for m in members};edges=defaultdict(set)
    if not set(initial)<=ids:raise ValueError('Unknown held member')
    for m in members:
        parent=m.get('source_member_id')
        if parent in ids:edges[m['member_id']].add(parent);edges[parent].add(m['member_id'])
    for relation in relations:
        group=set(relation['members'])
        if not group<=ids:raise ValueError('Unknown relation member')
        for mid in group:edges[mid].update(group-{mid})
    closure=set(initial);pending=list(initial)
    while pending:
        for other in edges[pending.pop()]-closure:closure.add(other);pending.append(other)
    return closure


def inventory():
    dest=OUT/'closeout-inventory-v2.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    names=['protocol','closeout-inventory-v1','closeout-source-variant-quality',
           'closeout-remaining-quality','risk-derivation-closure-v1','development-exact-overlap-v1',
           'reference-fingerprint-overlap-v1','source-census','source-resolution','quality-isolation-v4']
    paths=[OUT/(n+'.json') for n in names];records=[prior.read(p) for p in paths]
    for r in records:prior.verify(r)
    p,old,reused,authored,graph,dev,protected,census,resolved,q=records
    if dev['matches'] or protected['matches']:raise ValueError('Role fingerprint collision')
    lookup={m['member_id']:m for m in p['members']};rows={m['member_id']:dict(m) for m in old['members']}
    for d in reused['members']:
        rows[d['member_id']]['quality_decision_sources'].append(str(paths[2]))
    held={m['member_id'] for m in old['members'] if m['stage']=='quarantined'}
    for d in authored['decisions']:
        row=rows[d['member_id']];row['quality_decision_sources'].append(str(paths[3]))
        if d['quality_status']=='whole_frame_held':
            held.add(d['member_id']);row['preserved_quarantine_reasons'].append(d['reason'])
    closure=close_risks(p['members'],graph['relations'],held)
    source={m['member_id']:m for m in census['members']}
    fixed={m['member_id'] for m in resolved['resolved_sources']}
    for mid,row in rows.items():
        if mid in closure:
            row['stage']='whole_frame_held'
            if mid not in held:row['preserved_quarantine_reasons'].append('Explicit derivation group contains held member')
        elif row['quality_decision_sources']:
            row['stage']='quality_supported_source_role_gate_pending'
        else:raise ValueError('Missing quality disposition: '+mid)
        row['registered_data_role']=lookup[mid]['data_role']
        row['source_pixels_verified']=source[mid]['source_pixel_status']=='pixel_equal' or mid in fixed
        if not row['source_pixels_verified']:raise ValueError('Unresolved original RGB source: '+mid)
        row['exact_development_and_reference_fingerprint_match']=False
        row['independent_scene_claim']=False
        row['training_eligible']=False
    paths.append(Path(__file__).resolve())
    return prior.frozen(dest,dict(status='all_429_quality_dispositions_accounted_source_role_gate_pending',
        members=list(rows.values()),counts=dict(Counter(r['stage'] for r in rows.values())),
        additional_derivative_holds=sorted(closure-held),
        remaining_gates=['per_member_full_scene_source_role_evidence','frozen_training_exposure_and_brightness_preflight'],
        limitation='Exact exclusion is not pose/asset independence. Dataset staging is not training readiness.',
        inputs={str(p):prior.file_sha256(p) for p in paths}))


def stage_loader():
    import torch
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from scripts.vision.order_retention_runtime import overrides,validate_labels
    from scripts.vision.preflight_unified_lighting import tensor_hash
    inv=inventory();p=prior.read(OUT/'protocol.json');prior.verify(p)
    selected={r['member_id'] for r in inv['members'] if r['stage']=='quality_supported_source_role_gate_pending'}
    members=sorted((m for m in p['members'] if m['member_id'] in selected),key=lambda m:m['member_id'])
    root=OUT/'candidate-staging-v1';root.mkdir(exist_ok=True)
    for number in range(1,4):
        attempt=root/f'attempt-{number:02d}';done=attempt/'completion.json'
        if done.exists():
            r=prior.read(done)
            try:prior.verify(r);return r
            except (ValueError,FileNotFoundError):continue
        if attempt.exists():continue
        attempt.mkdir();break
    else:raise ValueError('Three staging attempts exhausted')
    deps=[OUT/'protocol.json',OUT/'closeout-inventory-v2.json',Path(__file__).resolve(),
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
        def forbidden(*args,**kwargs):raise AssertionError('Training forbidden in staging')
        with ExitStack() as stack:
            for obj,name in [(torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')]:
                stack.enter_context(patch.object(obj,name,forbidden))
            data={'names':{0:'switchgear',1:'transformer',2:'reactor',3:'capacitor_bank'},'nc':4,'channels':3}
            dataset=build_yolo_dataset(config,str(images),6,data,mode='train',rect=False,stride=32)
            validate_labels(dataset,lookup)
            if list(dataset.im_files)!=[r['image_path'] for r in rows]:raise ValueError('Loader member order drift')
            loader=torch.utils.data.DataLoader(dataset,batch_size=6,shuffle=False,num_workers=0,collate_fn=dataset.collate_fn)
            for batch in loader:
                ids=[inverse[path]['member_id'] for path in batch['im_file']];n=len(ids)
                if tuple(batch['img'].shape)!=(n,3,640,640):raise ValueError('Bad image shape')
                for i,mid in enumerate(ids):
                    cls=batch['cls'][batch['batch_idx']==i].flatten().numpy().astype(int).tolist()
                    text=Path(lookup[mid]['label_path']).read_text().strip()
                    expected=[int(line.split()[0]) for line in text.splitlines()]
                    if Counter(cls)!=Counter(expected):raise ValueError('Instance dropped from loaded batch')
                actual.extend(ids)
                batches.append(dict(members=ids,image_tensor_sha256=tensor_hash(batch['img']),
                    full_supervision={k:tensor_hash(batch[k]) for k in ('cls','bboxes','batch_idx')}))
        if actual!=[r['member_id'] for r in rows]:raise ValueError('Incomplete loader traversal')
        return prior.frozen(done,dict(status='candidate_bytes_and_loader_verified_not_training_ready',members=rows,
            checked_members=len(rows),checked_batches=len(batches),batches=batches,
            optimizer_created=False,backward_executed=False,validation_run=False,training_started=False,
            scope='One traversal of candidates; not a frozen exposure/brightness schedule preflight.',
            environment={'torch':torch.__version__},inputs={str(p):prior.file_sha256(p) for p in deps}))
    except BaseException as exc:
        prior.frozen(attempt/'failure.json',dict(status='staging_failed',error=repr(exc),training_started=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage-loader',action='store_true');args=parser.parse_args()
    r=stage_loader() if args.stage_loader else inventory();print(r.get('counts',r['status']))
