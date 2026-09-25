"""Two real loaders per seed, complete traversal; never trains."""
import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import traceback
from scripts.vision.material_transfer_controls import OUT as DESIGN,prior
from scripts.vision.material_control_feasibility import RUN
from scripts.vision.brightness_transfer_runtime import make_dataset,check_log
from scripts.vision.order_retention_runtime import make_loader,check_actual
from scripts.vision.preflight_unified_lighting import tensor_hash
from scripts.vision.frozen_multiscale_runtime import preprocess
OUT=DESIGN.parent/'material-multiscale-loader-v1'

def forbidden(*a,**k):raise AssertionError('Optimizer/backward/training/validation forbidden')

def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():prior.verify(prior.read(dest));return prior.read(dest)
    dp=DESIGN/'completion.json';d=prior.read(dp);prior.verify(d);old=prior.read(RUN/'protocol.json');prior.verify(old)
    design=d['next_design']
    paths=[dp,RUN/'protocol.json',Path(__file__).resolve(),prior.ROOT/'scripts/vision/frozen_multiscale_runtime.py']
    for name in ('brightness_transfer_runtime.py','order_retention_runtime.py','preflight_unified_lighting.py'):
        paths.append(prior.ROOT/'scripts/vision'/name)
    for seed in (7,17,27):
        key=f'R-clean-{seed}'
        if design['member_sequences'][str(seed)]!=old['schedules'][key] or design['brightness_factors'][str(seed)]!=old['brightness_factors'][key]:raise ValueError('Design sequence/factor mismatch')
        if len(design['scale_by_step'][str(seed)])!=450 or Counter(design['scale_by_step'][str(seed)])!={320:150,640:150,960:150}:raise ValueError('Scale quota mismatch')
        if any(x in old['held_members'] for x in design['member_sequences'][str(seed)]):raise ValueError('Held exposure')
    import inspect,torch,ultralytics
    from ultralytics.models.yolo.detect import DetectionTrainer
    if dict(torch=torch.__version__,ultralytics=ultralytics.__version__)!=old['environment']:raise ValueError('Historical environment mismatch')
    paths.append(Path(inspect.getfile(DetectionTrainer)))
    OUT.mkdir(exist_ok=True)
    return prior.frozen(dest,dict(status='frozen_loader_preflight_only',design=design,source_protocol=str(RUN/'protocol.json'),environment=old['environment'],
        interpolation='Actual DetectionTrainer normalization with multi_scale=0, then frozen square bilinear align_corners=False resize from 640; not native-960 detail.',
        training_authorized=False,training_ready=False,inputs={str(x):prior.file_sha256(x) for x in paths}))

def validate(r,p,source):
    prior.verify(r)
    if r['status']!='paired_loaders_complete' or r['protocol_identity']!=p['identity']:raise ValueError('Invalid completion')
    seed=r['seed'];key=f'R-clean-{seed}'
    if set(r['arms'])!={'fixed','multiscale'}:raise ValueError('Missing arm')
    for arm,cell in r['arms'].items():
        check_actual(source,key,cell['actual']);check_log(source,key,cell['brightness_log'])
        expected=[640]*450 if arm=='fixed' else p['design']['scale_by_step'][str(seed)]
        if [b['size'] for b in cell['batches']]!=expected:raise ValueError('Missing/changed effective sizes')
        if len(cell['batches'])!=450:raise ValueError('Incomplete batch ledger')
        for i,b in enumerate(cell['batches']):
            if b['members']!=source['schedules'][key][i*6:i*6+6]:raise ValueError('Changed batch order')
    if r['arms']['fixed']['brightness_log']!=r['arms']['multiscale']['brightness_log']:raise ValueError('Brightness pixels changed across arms')

def run():
    p=freeze();source=prior.read(p['source_protocol']);prior.verify(source)
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    torch.set_num_threads(4);lookup={r['image_path']:r['member_id'] for r in source['pool_rows']}
    paths=[OUT/'protocol.json'];(OUT/'checks').mkdir(exist_ok=True)
    for seed in (7,17,27):
        folder=OUT/'checks'/f'seed-{seed}';folder.mkdir(exist_ok=True)
        completed=list(folder.glob('attempt-*/complete.json'))
        if completed:
            if len(completed)!=1:raise ValueError('Duplicate completion')
            validate(prior.read(completed[0]),p,source);paths+=completed;print('REUSE',seed,flush=True);continue
        if any(prior.read(x)['status']=='semantic_failure' for x in folder.glob('attempt-*/failure.json')):raise ValueError('Unresolved semantic failure')
        number=len(list(folder.glob('attempt-*')))+1
        if number>3:raise ValueError('Attempt budget exhausted')
        attempt=folder/f'attempt-{number:03}';attempt.mkdir();key=f'R-clean-{seed}'
        historical=list((RUN/'loader-checks'/f'seed-{seed}').glob('attempt-*/complete.json'))
        if len(historical)!=1:raise ValueError('Missing historical fixed loader identity')
        h=prior.read(historical[0]);prior.verify(h);expected=h['cells'][key]
        arms={a:dict(actual=[],brightness_log=[],batches=[]) for a in ('fixed','multiscale')}
        owner=SimpleNamespace(epoch=0);processor=SimpleNamespace(device=torch.device('cpu'),args=SimpleNamespace(multi_scale=0),stride=32)
        try:
            init_seeds(seed,deterministic=True)
            with ExitStack() as stack:
                for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):stack.enter_context(patch.object(obj,name,forbidden))
                loaders=[make_loader(make_dataset(source,key,arms[a]['brightness_log']),source,key,owner) for a in arms]
                step=0
                for epoch in range(45):
                    owner.epoch=epoch
                    for left,right in zip(*loaders,strict=True):
                        if left['im_file']!=right['im_file'] or not torch.equal(left['img'],right['img']):raise ValueError('Base batch mismatch')
                        record=expected['batch_records'][step]
                        if tensor_hash(left['img'])!=record['image_tensor_sha256']:raise ValueError('Historical fixed image tensor differs')
                        for field in ('cls','bboxes','batch_idx'):
                            if not torch.equal(left[field],right[field]) or tensor_hash(left[field])!=record['full_supervision'][field]:raise ValueError('Full label tensors differ')
                        for arm,batch in zip(arms,(left,right)):
                            size=640 if arm=='fixed' else p['design']['scale_by_step'][str(seed)][step]
                            out=preprocess(processor,batch,size);members=[lookup[x] for x in batch['im_file']]
                            arms[arm]['actual']+=members
                            arms[arm]['batches'].append(dict(step=step,size=size,members=members,source640_sha256=tensor_hash(batch['img']),
                                effective_tensor_sha256=tensor_hash(out['img']),effective_shape=list(out['img'].shape),
                                full_supervision={f:tensor_hash(out[f]) for f in ('cls','bboxes','batch_idx')}))
                        step+=1
                        if step%150==0:print('PROGRESS',seed,step,'/450 paired batches',flush=True)
            if step!=450:raise ValueError('Incomplete traversal')
            if arms['fixed']['brightness_log']!=expected['brightness_log']:raise ValueError('Historical brightness log differs')
            r=prior.frozen(attempt/'complete.json',dict(status='paired_loaders_complete',seed=seed,protocol_identity=p['identity'],arms=arms,
                historical_fixed_tensors_equal=True,complete_labels_equal=True,optimizer_created=False,backward_executed=False,training_validation_run=False,
                inputs={str(x):prior.file_sha256(x) for x in (OUT/'protocol.json',historical[0])}))
            validate(r,p,source);paths.append(attempt/'complete.json');print('COMPLETE',seed,'5400 actual loads',flush=True)
        except BaseException as exc:
            prior.frozen(attempt/'failure.json',dict(status='semantic_failure' if isinstance(exc,ValueError) else 'technical_or_cancelled_failure',reason=traceback.format_exc(),child_processes_started=0,training_started=False));raise
    prior.frozen(OUT/'loader-completion.json',dict(status='six_loaders_passed_final_regression_pending',training_started=False,training_ready=False,
        actual_image_loads=16200,actual_batches=2700,inputs={str(x):prior.file_sha256(x) for x in paths}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check-loaders',action='store_true');a=ap.parse_args()
    if a.check_loaders:run()
    else:freeze();print('STATIC_PREFLIGHT_ONLY_NO_TRAINING')
