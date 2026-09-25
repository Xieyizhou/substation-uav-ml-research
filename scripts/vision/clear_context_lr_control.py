"""Independent LR-only control; default preflight, explicit train then evaluate."""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision import clear_context_training as runtime
from scripts.vision.freeze_clear_context_training import OUT as PREVIOUS, checked

OUT=PREVIOUS.parent/'lr-stability-control-v1'
KEYS=tuple(f'clear-context-lr025-480-{s}' for s in (7,17,27))
LR=.00025


def validate_pair(old,new):
    for field in ('pool_rows','names','initialization','evaluation','environment','held_members'):
        if old[field]!=new[field]:raise ValueError('Non-LR change: '+field)
    for seed,key in zip((7,17,27),KEYS):
        previous=f'clear-context-480-{seed}'
        for field in ('schedules','exposures','windows'):
            if new[field][key]!=old[field][previous]:raise ValueError('Exposure change: '+field)
    expected=copy.deepcopy(old['training_config']);expected['lr0']=LR
    if new['training_config']!=expected:raise ValueError('Training config not LR-only')
    expected=copy.deepcopy(old['configuration']);expected['lr']=LR
    if new['configuration']!=expected:raise ValueError('Configuration not LR-only')


def freeze():
    old_path=PREVIOUS/'protocol.json'; old=checked(old_path);dest=OUT/'protocol.json'
    if dest.exists():
        current=checked(dest);validate_pair(old,current);return current
    OUT.mkdir(parents=True,exist_ok=True)
    deps=[old_path,Path(__file__),Path(runtime.__file__)]
    for rel in ('evaluation-v1/error-review-v1/review.json','evaluation-v1/positive-review-v1/review.json','new-member-fit-v1.json'):
        p=PREVIOUS/rel;checked(p);deps.append(p)
    new=copy.deepcopy(old)
    for field in ('identity','inputs','replacement_audit'):new.pop(field,None)
    for field in ('schedules','exposures','windows','listings'):new[field]={}
    for seed,key in zip((7,17,27),KEYS):
        previous=f'clear-context-480-{seed}'
        cp=PREVIOUS/'training'/previous/'completion.json';c=checked(cp);ep=Path(c['exposure_path']);e=checked(ep)
        if c['optimizer_steps']!=480 or e['actual']!=old['schedules'][previous]:raise ValueError('Invalid direct control')
        deps.extend([cp,ep,Path(c['weights'])])
        for field in ('schedules','exposures','windows'):new[field][key]=copy.deepcopy(old[field][previous])
        # A read-only historical listing is valid because membership and pixels are identical.
        new['listings'][key]=old['listings'][previous];deps.append(Path(new['listings'][key]))
        for row in old['pool_rows']:
            for field,h in [('image_path','image_sha256'),('label_path','label_sha256')]:
                if file_sha256(row[field])!=row[h]:raise ValueError('Stale training member')
    new['training_config']['lr0']=LR;new['configuration']['lr']=LR
    new.update(status='frozen_lr_only_preflight_pending',design='Only constant AdamW learning rate changes from 0.0005 to 0.00025. Same initialization, samples, order, batches, exposure and 480 steps; all three seeds retained.',
        rationale='Seed 17 misses all five newly exposed switchgear instances despite actual exposure; other seeds fit them. Test update-size sensitivity, not a proven mechanism or sufficient budget claim.',
        limits='Lower learning rate can also slow fitting. One bounded comparison, no seed/checkpoint selection, no claim of unique cause or independent-scene generalization.',
        inputs={str(p.resolve()):file_sha256(p) for p in deps})
    validate_pair(old,new)
    return write_record(dest,new)


def training_override(kwargs):
    if kwargs.get('lr0')!=.0005 or kwargs.get('lrf')!=1.0 or kwargs.get('epochs')!=48:
        raise ValueError('Historical trainer configuration drift')
    result=dict(kwargs);result['lr0']=LR;return result


def worker(key):
    from ultralytics import YOLO
    import yaml
    original=YOLO.train
    def train(model,*args,**kwargs):return original(model,*args,**training_override(kwargs))
    with patch.object(YOLO,'train',train):result=runtime.worker(key)
    args_path=Path(result['weights']).parent.parent/'args.yaml'
    config=yaml.safe_load(args_path.read_text())
    if config['lr0']!=LR or config['lrf']!=1.0 or config['epochs']!=48:raise ValueError('Actual LR mismatch')
    path=OUT/'training'/key/'lr-verification.json'
    if not path.exists():write_record(path,dict(status='actual_lr_verified',lr0=LR,training_admitted=False,promotable=False,
        inputs={str(p.resolve()):file_sha256(p) for p in (args_path,OUT/'training'/key/'completion.json',Path(__file__))}))
    return result


def summarize():
    records=[];comparisons=[];deps={};queue=[]
    for seed,key in zip((7,17,27),KEYS):
        cp=OUT/'evaluation-v1/units'/key/'completion.json';c=checked(cp);r=checked(c['result']);records.append(r)
        op=PREVIOUS/'evaluation-v1/units'/f'clear-context-480-{seed}'/'completion.json';o=checked(op);old=checked(o['result'])
        by={(x['pair_id'],x['variant']):x for x in old['rows']}
        for row in r['rows']:
            comparisons.append(dict(seed=seed,pair_id=row['pair_id'],variant=row['variant'],instances=runtime.evaluation.compare_truth(by[row['pair_id'],row['variant']],row)))
        for row in r['negative_rows']:
            for pred in row['predictions']:queue.append(dict(seed=seed,view_id=row['view_id'],image_sha256=row['image_sha256'],prediction=pred,review_status='pending'))
        for path in (cp,op,Path(c['result']),Path(o['result']),OUT/'training'/key/'lr-verification.json'):checked(path);deps[str(path.resolve())]=file_sha256(path)
    return write_record(OUT/'evaluation-v1/summary.json',dict(status='numerical_complete_visual_review_and_retention_gates_pending',group=runtime.evaluation.aggregate(records),
        instance_comparisons=comparisons,negative_fp_review_queue=queue,matching_conflicts={k:r['matching_conflicts'] for k,r in zip(KEYS,records)},
        selected_candidate=None,training_admitted=False,promotable=False,inputs=deps))


def configure():
    runtime.OUT=OUT;runtime.KEYS=KEYS;runtime.freeze=freeze;runtime.MODULE='scripts.vision.clear_context_lr_control'
    runtime.TESTS=runtime.TESTS+('tests.test_clear_context_lr_control',)
    runtime.summarize=summarize


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--train',action='store_true');parser.add_argument('--worker',choices=KEYS);parser.add_argument('--eval-worker',choices=KEYS)
    args=parser.parse_args();configure()
    if args.worker:worker(args.worker)
    elif args.eval_worker:runtime.eval_worker(args.eval_worker)
    elif args.train:runtime.train()
    else:runtime.preflight();print('PREFLIGHT_ONLY_NO_TRAINING')
