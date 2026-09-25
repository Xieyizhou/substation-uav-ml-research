"""Freeze and actually preflight a learning-rate-only control. Never train."""
import argparse
import copy
import traceback
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.vision import train_material_retention_coverage as reference
from scripts.vision.preflight_unified_lighting import tensor_hash
from scripts.vision.brightness_transfer_runtime import make_dataset, check_log
from scripts.vision.order_retention_runtime import make_loader, check_actual
from scripts.vision.infer_material_member_fit import forbidden

prior=reference.prior
OUT=reference.OUT.parent/'material-fixed-sequence-lr-v1'
KEYS=tuple(f'{arm}-{seed}' for seed in (7,17,27) for arm in ('R','Q'))


def check_pair(p,seed):
    a,b=f'R-{seed}',f'Q-{seed}'
    for name in ('schedules','brightness_factors','listings'):
        if p[name][a]!=p[name][b]: raise ValueError('Non-LR input changed: '+name)
    ac,bc=p['training_config'][a],p['training_config'][b]
    if ac['lr0']!=.0005 or bc['lr0']!=.00025: raise ValueError('Wrong LR')
    if {k:v for k,v in ac.items() if k!='lr0'}!={k:v for k,v in bc.items() if k!='lr0'}:
        raise ValueError('Non-LR configuration changed')


def freeze():
    old,_,_=reference.contract('T-7')
    p={k:copy.deepcopy(old[k]) for k in ('pool_rows','names','initialization','evaluation','held_members')}
    p.update(schedules={},brightness_factors={},training_config={},listings={})
    paths=[reference.OUT/'protocol.json',reference.OUT/'decision-protocol.json',Path(__file__).resolve()]
    import torch,ultralytics
    p['environment']=dict(torch=torch.__version__,ultralytics=ultralytics.__version__)
    for seed in (7,17,27):
        key=f'T-{seed}';reference.complete(key)
        for arm,lr in (('R',.0005),('Q',.00025)):
            new=f'{arm}-{seed}'
            for name in ('schedules','brightness_factors','training_config','listings'):
                p[name][new]=copy.deepcopy(old[name][key])
            p['training_config'][new]['lr0']=lr
        check_pair(p,seed)
        cp=reference.OUT/'training'/key/'completion.json';c=prior.read(cp)
        paths.extend((cp,Path(c['exposure_path']),Path(c['weights'])))
    p.update(status='frozen_actual_preflight_pending',reference_policy='reuse_valid_T_only_no_silent_retrain',
        new_training_cells=['Q-7','Q-17','Q-27'],reference_cells=['T-7','T-17','T-27'],
        checkpoint_selection='terminal_last_only',new_training_started=False,
        acceptance_policy=prior.read(reference.OUT/'decision-protocol.json'),
        inputs={str(x):prior.file_sha256(x) for x in paths})
    OUT.mkdir(exist_ok=True)
    dest=OUT/'protocol.json'
    if dest.exists():
        r=prior.read(dest);prior.verify(r)
        for k in p:
            if r[k]!=p[k]: raise ValueError('Frozen protocol drift: '+k)
        return r
    return prior.frozen(dest,p)


def preflight():
    p=freeze()
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    torch.set_num_threads(4)
    lookup={r['image_path']:r['member_id'] for r in p['pool_rows']}
    for key in KEYS:
        seed=int(key.split('-')[-1]);_,_,expected=reference.contract(f'T-{seed}')
        root=OUT/'loader-checks'/key;root.mkdir(parents=True,exist_ok=True)
        done=list(root.glob('attempt-*/complete.json'))
        if done:
            if len(done)!=1: raise ValueError('Duplicate preflight')
            r=prior.read(done[0]);prior.verify(r)
            if any(r[k]!=expected[k] for k in expected): raise ValueError('Historical actual input drift')
            continue
        if any((x/'failure.json').exists() and prior.read(x/'failure.json')['semantic_stop'] for x in root.glob('attempt-*')):
            raise ValueError('Unresolved semantic preflight failure')
        n=len(list(root.glob('attempt-*')))+1
        if n>3: raise ValueError('Attempt budget exhausted')
        attempt=root/f'attempt-{n:03}';attempt.mkdir()
        try:
            logs=[];actual=[];batches=[];owner=SimpleNamespace(epoch=0)
            init_seeds(seed,deterministic=True)
            with ExitStack() as stack:
                for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):
                    stack.enter_context(patch.object(obj,name,forbidden))
                loader=make_loader(make_dataset(p,key,logs),p,key,owner)
                for epoch in range(45):
                    owner.epoch=epoch
                    for batch in loader:
                        if tuple(batch['img'].shape)!=(6,3,640,640): raise ValueError('Batch shape drift')
                        mids=[lookup[x] for x in batch['im_file']];actual.extend(mids)
                        row=dict(step=len(batches),members=mids,image_tensor_sha256=tensor_hash(batch['img']),
                            full_supervision={f:tensor_hash(batch[f]) for f in ('cls','bboxes','batch_idx')})
                        if row!=expected['batch_records'][len(batches)]: raise ValueError('Actual tensor or full labels differ from T')
                        batches.append(row)
                    if epoch%10==0: print(key,'verified',len(batches),'steps',flush=True)
            check_actual(p,key,actual);check_log(p,key,logs)
            if len(batches)!=450 or logs!=expected['brightness_log']: raise ValueError('Incomplete/changed brightness')
            prior.frozen(attempt/'complete.json',dict(status='actual_loader_verified_no_training',cell=key,
                actual=actual,batch_records=batches,brightness_log=logs,optimizer_created=False,
                backward_executed=False,training_validation_executed=False,
                inputs={str(OUT/'protocol.json'):prior.file_sha256(OUT/'protocol.json')}))
            print('COMPLETE',key,flush=True)
        except BaseException as ex:
            prior.frozen(attempt/'failure.json',dict(semantic_stop=isinstance(ex,ValueError),error=traceback.format_exc(),child_processes_started=0))
            raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    if args.preflight:preflight()
    else:freeze();print('FROZEN_ONLY_NO_TRAINING')


if __name__=='__main__':main()
