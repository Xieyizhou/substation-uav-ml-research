"""Bounded contrast-only transfer experiment, default genuine preflight only."""
import argparse
import copy
import hashlib
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision import clear_context_training as runtime
from scripts.vision import clear_context_lr_control as previous

SOURCE=previous.OUT
OUT=SOURCE.parent/'contrast-transfer-control-v1'
KEYS=tuple(f'contrast-transfer-480-{s}' for s in (7,17,27))
checked=previous.checked
VERSION='full-frame-channel-mean-contrast-v1'

def coefficients(seed):
    positions=sorted(range(2880),key=lambda i:hashlib.sha256(f'{VERSION}|{seed}|{i}'.encode()).hexdigest())
    values=[None]*2880
    for j,i in enumerate(positions):values[i]=(.75,1.,1.25)[j%3]
    return values

def transform(images,values):
    import torch
    if images.dtype!=torch.uint8 or images.ndim!=4 or len(values)!=len(images):raise ValueError('Invalid contrast input')
    if any(v not in (.75,1.,1.25) for v in values):raise ValueError('Unknown contrast coefficient')
    x=images.float();mean=x.mean(dim=(2,3),keepdim=True)
    factors=torch.tensor(values,dtype=x.dtype,device=x.device).reshape(-1,1,1,1)
    raw=(x-mean)*factors+mean
    clipped=((raw<0)|(raw>255)).sum(dim=(1,2,3)).tolist()
    return raw.round().clamp(0,255).to(torch.uint8),clipped

def validate_pair(old,new):
    for field in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config'):
        if new[field]!=old[field]:raise ValueError('Non-contrast change: '+field)
    config=copy.deepcopy(old['configuration']);config['augmentation']=VERSION
    if new['configuration']!=config:raise ValueError('Unexpected training configuration')
    for seed,key,prior in zip((7,17,27),KEYS,previous.KEYS):
        for field in ('schedules','exposures','windows','listings'):
            if new[field][key]!=old[field][prior]:raise ValueError('Exposure or label listing drift')
        if new['contrast'][key]!=coefficients(seed):raise ValueError('Contrast array drift')

def freeze():
    path=OUT/'protocol.json';old=checked(SOURCE/'protocol.json')
    if path.exists():
        p=checked(path);validate_pair(old,p);return p
    OUT.mkdir(parents=True,exist_ok=True)
    p=copy.deepcopy(old)
    for f in ('identity','inputs'):p.pop(f,None)
    for f in ('schedules','exposures','windows','listings'):p[f]={k:copy.deepcopy(old[f][o]) for k,o in zip(KEYS,previous.KEYS)}
    p['contrast']={k:coefficients(s) for k,s in zip(KEYS,(7,17,27))}
    p['configuration']['augmentation']=VERSION
    deps=[SOURCE/'protocol.json',SOURCE/'material-fit-v1/summary.json',SOURCE/'material-fit-v1/residual-review-v1/review.json',SOURCE/'verification-v1/completion.json',Path(__file__),Path(runtime.__file__),Path(previous.__file__)]
    for key in previous.KEYS:
        cp=SOURCE/'training'/key/'completion.json';c=checked(cp);e=checked(c['exposure_path'])
        if c['optimizer_steps']!=480 or e['actual']!=old['schedules'][key]:raise ValueError('Direct control invalid')
        deps.extend([cp,Path(c['exposure_path']),Path(c['weights']),SOURCE/'training'/key/'lr-verification.json',SOURCE/'actual-preflight'/f'{key}.json'])
        ec=SOURCE/'evaluation-v1/units'/key/'completion.json';ev=checked(ec);deps.extend([ec,Path(ev['result'])])
    for d in deps:
        if d.suffix=='.json':checked(d)
    for row in old['pool_rows']:
        for kind in ('image','label'):
            f=Path(row[kind+'_path'])
            if file_sha256(f)!=row[kind+'_sha256']:raise ValueError('Member changed')
            deps.append(f)
    p.update(status='frozen_contrast_only_preflight_pending',design='Same 392 members, 2880 draws, order, labels, 480 batches, LR 0.00025 and independent v2.11 initialization. Only deterministic whole-frame contrast differs.',
        rationale='Material training recall is 308/311,307/311,311/311 while material development mean recall is 27.22%. Test bounded photometric condition robustness, not longer fitting.',
        limits='Contrast around each RGB channel spatial mean is not physical material rendering. It affects equipment, background and negatives jointly and may clip highlights/shadows; clipping is logged. No claim of unique cause, source independence or structural recognition. No parameter sweep or checkpoint selection.',
        contrast_definition='At raw uint8 loader output before division by 255: round(clamp(mean_channel + coefficient*(pixel-mean_channel),0,255)). Coefficients 0.75,1,1.25 each 960 positions per seed; hash-frozen, no geometry/label changes.',
        direct_control=str(SOURCE),training_admitted=False,promotable=False,inputs={str(d.resolve()):file_sha256(d) for d in deps})
    validate_pair(old,p);return write_record(path,p)

def loader(p,key,owner):
    dataset,raw=runtime.baseline.loader(p,key,owner)
    ep=OUT/'actual-preflight'/f'{key}.json'
    expected=checked(ep)['tensor_records'] if ep.exists() else None
    prior=previous.KEYS[KEYS.index(key)]
    reference=checked(SOURCE/'actual-preflight'/f'{prior}.json')['tensor_records']
    records=runtime.OBSERVED.setdefault(key,[])
    class Wrapped:
        def __len__(self):return len(raw)
        def __getattr__(self,name):return getattr(raw,name)
        def __iter__(self):
            for j,b in enumerate(raw):
                pos=owner.epoch*60+j*6
                before={k:runtime.tensor_hash(b[k]) for k in ('img','cls','bboxes','batch_idx')}
                ref=reference[pos//6]
                if before!=ref['tensors'] or list(b['im_file'])!=ref['images']:raise ValueError('Raw tensor/complete label drift from direct control')
                factors=p['contrast'][key][pos:pos+6]
                b=dict(b);b['img'],clipped=transform(b['img'],factors)
                record=dict(position=pos,images=list(b['im_file']),raw_tensors=before,coefficients=factors,clipped_channels=clipped,
                    tensors={k:runtime.tensor_hash(b[k]) for k in ('img','cls','bboxes','batch_idx')})
                if any(record['tensors'][k]!=before[k] for k in ('cls','bboxes','batch_idx')):raise ValueError('Labels changed')
                if expected is not None and record!=expected[pos//6]:raise ValueError('Augmented tensor drift')
                records.append(record);yield b
    return dataset,Wrapped()

def worker(key):
    from ultralytics import YOLO
    import yaml
    original=YOLO.train
    def train(model,*args,**kwargs):return original(model,*args,**previous.training_override(kwargs))
    with patch.object(YOLO,'train',train):result=runtime.worker(key)
    args_path=Path(result['weights']).parent.parent/'args.yaml';config=yaml.safe_load(args_path.read_text())
    if config['lr0']!=.00025 or config['epochs']!=48 or config['lrf']!=1.:raise ValueError('Actual configuration drift')
    return result

def summarize():
    records=[];comparisons=[];deps={};queue=[]
    for seed,key,prior in zip((7,17,27),KEYS,previous.KEYS):
        cp=OUT/'evaluation-v1/units'/key/'completion.json';c=checked(cp);r=checked(c['result']);records.append(r)
        op=SOURCE/'evaluation-v1/units'/prior/'completion.json';o=checked(op);old=checked(o['result']);by={(x['pair_id'],x['variant']):x for x in old['rows']}
        for row in r['rows']:comparisons.append(dict(seed=seed,pair_id=row['pair_id'],variant=row['variant'],instances=runtime.evaluation.compare_truth(by[row['pair_id'],row['variant']],row)))
        for row in r['negative_rows']:
            for pred in row['predictions']:queue.append(dict(seed=seed,view_id=row['view_id'],image_sha256=row['image_sha256'],prediction=pred,review_status='pending'))
        for f in (cp,op,Path(c['result']),Path(o['result'])):deps[str(f.resolve())]=file_sha256(f)
    return write_record(OUT/'evaluation-v1/summary.json',dict(status='numerical_complete_visual_review_and_retention_gates_pending',group=runtime.evaluation.aggregate(records),
        instance_comparisons=comparisons,negative_fp_review_queue=queue,matching_conflicts={k:r['matching_conflicts'] for k,r in zip(KEYS,records)},
        selected_candidate=None,training_admitted=False,promotable=False,inputs=deps))

def configure():
    runtime.OUT=OUT;runtime.KEYS=KEYS;runtime.freeze=freeze;runtime.loader=loader;runtime.MODULE='scripts.vision.contrast_transfer_control'
    runtime.TESTS=runtime.TESTS+('tests.test_contrast_transfer_control',);runtime.summarize=summarize

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);ap.add_argument('--eval-worker',choices=KEYS);a=ap.parse_args();configure()
    if a.worker:worker(a.worker)
    elif a.eval_worker:runtime.eval_worker(a.eval_worker)
    elif a.train:runtime.train()
    else:runtime.preflight();print('PREFLIGHT_ONLY_NO_TRAINING')
