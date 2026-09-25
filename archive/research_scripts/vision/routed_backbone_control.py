"""A bounded frozen-backbone control against unchanged routed training."""
import argparse
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.vision import material_routed_contrast_control as routed
from scripts.vision import routed_reflection_control as reflection
from scripts.vision import contrast_transfer_control as augmentation
from scripts.vision import contrast_cpu4_control as cpu
from scripts.vision import clear_context_training as runtime
from scripts.vision.record_contrast_cpu4_review import validate_positive
from scripts.vision.audit_reviewed_order_results import validate_review
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

SOURCE=routed.OUT
SOURCE_KEYS=tuple(routed.KEYS)
OUT=SOURCE.parent/'routed-backbone-retention-control-v1'
KEYS=tuple(f'routed-backbone-480-{s}' for s in (7,17,27))
FROZEN=list(range(11))
VERSION='fixed-v211-backbone-0-through-10-including-bn-v1'
POLICY=dict(version=VERSION,indices=FROZEN,batchnorm='eval_running_stats_fixed',
            ema='copy_frozen_state_exactly_after_each_update',other_parameters='train_except_existing_dfl')
checked=routed.checked

def frozen_name(name):
    return any(name.startswith(f'model.{i}.') for i in FROZEN)

def state_digest(model, frozen=True, half=False):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        if frozen_name(name)!=frozen:continue
        t=t.detach().cpu().contiguous()
        if half and t.is_floating_point():t=t.half()
        h.update(f'{name}|{t.dtype}|{tuple(t.shape)}'.encode());h.update(t.numpy().tobytes())
    return h.hexdigest()

def assert_policy(model):
    import torch
    frozen=other=0
    for name,p in model.named_parameters():
        expected=not (frozen_name(name) or '.dfl' in name)
        if p.requires_grad!=expected:raise ValueError('Parameter freeze mismatch '+name)
        if frozen_name(name):
            frozen+=p.numel()
            if p.grad is not None:raise ValueError('Frozen parameter has gradient '+name)
        elif expected:other+=p.numel()
    for name,m in model.named_modules():
        if frozen_name(name) and isinstance(m,torch.nn.BatchNorm2d) and m.training:
            raise ValueError('Frozen BatchNorm in training mode '+name)
    if not frozen or not other:raise ValueError('Empty frozen or trainable partition')
    return dict(frozen_parameters=frozen,trainable_parameters=other)

def topology(model):
    expected=['Conv','Conv','C3k2','Conv','C3k2','Conv','C3k2','Conv','C3k2','SPPF','C2PSA']
    actual=[type(m).__name__ for m in model.model]
    if len(model.yaml['backbone'])!=11 or actual[:11]!=expected or len(actual)!=24 or actual[-1]!='Detect':
        raise ValueError('Unexpected v2.11 topology')
    return [dict(index=i,type=type(m).__name__,parameters=sum(p.numel() for p in m.parameters()),frozen=i in FROZEN) for i,m in enumerate(model.model)]

def copy_frozen_to_ema(model,ema):
    import torch
    a=model.state_dict();b=ema.state_dict()
    with torch.no_grad():
        for name in a:
            if frozen_name(name):b[name].copy_(a[name])

def validate(old,p):
    for name in ('pool_rows','names','initialization','evaluation','environment','held_members'):
        if old[name]!=p[name]:raise ValueError('Unexpected change '+name)
    cfg=copy.deepcopy(old['training_config']);cfg['freeze']=FROZEN
    config=copy.deepcopy(old['configuration']);config['parameter_update_policy']=VERSION
    if p['training_config']!=cfg or p['configuration']!=config or p['backbone_policy']!=POLICY:
        raise ValueError('Non-backbone training change')
    for k,o in zip(KEYS,SOURCE_KEYS):
        for f in ('schedules','exposures','windows','listings','contrast'):
            if p[f][k]!=old[f][o]:raise ValueError('Exposure, batch, label or transform change '+f)

def review_gate():
    a=reflection.OUT/'audit-v1'
    paths=[a/'evidence.json',a/'review.json',a/'material-evidence.json',reflection.OUT/'evaluation-v1/error-review-v1/evidence.json',a/'completion.json',a/'review-author-receipt.json',a/'evidence-build-receipt.json']
    e,r,m,n,c,_,_=map(checked,paths)
    validate_positive(e,r['positive_decisions']);validate_positive(m,r['material_decisions']);validate_review(n,r['negative_decisions'])
    if c['status']!='review_complete_not_candidate_passed' or not c['integrity']['integrity_passed'] or c['integrity']['pinned_files_verified']!=40:
        raise ValueError('Previous review incomplete')
    return paths

def freeze():
    old=checked(SOURCE/'protocol.json');reviews=review_gate();dest=OUT/'protocol.json'
    if dest.exists():
        p=checked(dest);validate(old,p);return p
    OUT.mkdir(parents=True,exist_ok=True);p=copy.deepcopy(old)
    for f in ('identity','inputs','routing_ledger'):p.pop(f,None)
    for f in ('schedules','exposures','windows','listings','contrast'):
        p[f]={k:copy.deepcopy(old[f][o]) for k,o in zip(KEYS,SOURCE_KEYS)}
    p['training_config']['freeze']=FROZEN;p['configuration']['parameter_update_policy']=VERSION;p['backbone_policy']=copy.deepcopy(POLICY)
    from ultralytics.engine import trainer as engine
    paths=[SOURCE/'protocol.json',Path(__file__),Path(augmentation.__file__),Path(cpu.__file__),Path(runtime.__file__),Path(engine.__file__),Path(runtime.trainer.__file__),
           OUT/'research-plan-zh.md',Path('tests/test_routed_backbone_control.py'),*reviews]
    for key in SOURCE_KEYS:
        cp=SOURCE/'training'/key/'completion.json';c=checked(cp);ex=checked(c['exposure_path'])
        if c['optimizer_steps']!=480 or ex['actual']!=old['schedules'][key]:raise ValueError('Direct control invalid')
        paths += [cp,Path(c['exposure_path']),Path(c['weights']),SOURCE/'training'/key/'tensor-verification.json',SOURCE/'training'/key/'thread-verification.json',SOURCE/'actual-preflight'/f'{key}.json']
        ep=SOURCE/'evaluation-v1/units'/key/'completion.json';ec=checked(ep);paths += [ep,Path(ec['result'])]
    deps=dict(old['inputs'])
    for q in paths:
        if q.suffix=='.json':checked(q)
        deps[str(q.resolve())]=file_sha256(q)
    p.update(status='frozen_backbone_control_preflight_pending',arm='frozen_backbone',direct_control=str(SOURCE),
        design='Independent v2.11 initialization, identical routed input tensors, full labels, 2880 draws, 480 batches/steps, LR .00025. Freeze actual backbone modules 0..10 including BatchNorm buffers; train neck/head except default DFL. Preserve frozen EMA state exactly.',
        rationale='Routed material fitting is near saturated; grayscale and reflection have not resolved development transfer and retention. Test whether restricting feature updates improves this tradeoff, not an established forgetting mechanism.',
        limits='Frozen features may also prevent useful adaptation. One strategy comparison including frozen BN statistics, not isolation of weights versus normalization. No augmentation stacking, new collection, protected labels, sealed tests, independent-scene or structure-only claim.',
        training_admitted=False,promotable=False,inputs=deps)
    validate(old,p);return write_record(dest,p)

def loader(p,key,owner):
    dataset,raw=augmentation.loader(p,key,owner)
    reference=checked(SOURCE/'actual-preflight'/f'{SOURCE_KEYS[KEYS.index(key)]}.json')['tensor_records']
    class Wrapped:
        def __len__(self):return len(raw)
        def __getattr__(self,n):return getattr(raw,n)
        def __iter__(self):
            for j,b in enumerate(raw):
                pos=owner.epoch*10+j
                if runtime.OBSERVED[key][-1]!=reference[pos]:raise ValueError('Input tensor differs from direct routed control')
                yield b
    return dataset,Wrapped()

def model_preflight(p):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    path=OUT/'model-preflight.json'
    if path.exists():return checked(path)
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('No optimizer')),patch.object(torch.Tensor,'backward',side_effect=AssertionError('No backward')),patch.object(YOLO,'train',side_effect=AssertionError('No training')),patch.object(YOLO,'val',side_effect=AssertionError('No validation')):
        model=YOLO(p['initialization']['path']).model.float();layers=topology(model)
        initial=state_digest(model)
        for name,param in model.named_parameters():param.requires_grad_(not(frozen_name(name) or '.dfl' in name))
        owner=SimpleNamespace(model=model,freeze_layer_names=[f'model.{i}.' for i in FROZEN]+['.dfl'])
        DetectionTrainer._model_train(owner);partition=assert_policy(model)
        # Pure forward on a real preflight tensor, not a training/validation loop.
        key=KEYS[0];o=SimpleNamespace(epoch=0);runtime.OBSERVED[key]=[]
        _,batches=loader(p,key,o);batch=next(iter(batches))
        with torch.no_grad():model(batch['img'].float()/255.)
        assert_policy(model)
        if state_digest(model)!=initial:raise ValueError('Backbone changed during forward')
    return write_record(path,dict(status='actual_model_freeze_and_bn_forward_verified_without_optimizer',layers=layers,partition=partition,
        frozen_initial_sha256=initial,checkpoint_half_frozen_sha256=state_digest(model,half=True),optimizer_created=False,backward_executed=False,validation_run=False,
        training_admitted=False,promotable=False,inputs={str(q.resolve()):file_sha256(q) for q in [OUT/'protocol.json',OUT/'actual-preflight'/f'{KEYS[0]}.json']}))

def prepare():
    p,ready=runtime.preflight();m=model_preflight(p);path=OUT/'pretraining-completion.json'
    deps=[OUT/'entry-ready.json',OUT/'model-preflight.json',OUT/'protocol.json']
    if path.exists():checked(path)
    else:write_record(path,dict(status='ready_for_training_not_started',actual_draws=8640,model_freeze_verified=True,
        training_admitted=False,promotable=False,inputs={str(q.resolve()):file_sha256(q) for q in deps}))
    return p,m

def verified_unit(key):
    c=checked(OUT/'training'/key/'completion.json');v=checked(OUT/'training'/key/'backbone-verification.json')
    for name in ('tensor-verification.json','thread-verification.json'):checked(OUT/'training'/key/name)
    if c['optimizer_steps']!=480 or len(v['steps'])!=480 or not v['nonbackbone_changed'] or not v['terminal_frozen_half_equal']:
        raise ValueError('Incomplete backbone unit')
    return c

def worker(key):
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    checked(OUT/'pretraining-completion.json');runtime.launch_gate();p=freeze();m=checked(OUT/'model-preflight.json')
    root=OUT/'training'/key
    if (root/'completion.json').exists():return verified_unit(key)
    setup=DetectionTrainer._setup_train;step=DetectionTrainer.optimizer_step;train=YOLO.train
    observations=[];snap={}
    def setup_checked(t):
        setup(t);topology(t.model);t._model_train();partition=assert_policy(t.model)
        if state_digest(t.model)!=m['frozen_initial_sha256']:raise ValueError('Initialization differs in backbone')
        snap.update(partition=partition,other_initial=state_digest(t.model,False),trainer=t)
    def step_checked(t,*args,**kwargs):
        assert_policy(t.model)
        result=step(t,*args,**kwargs)
        if state_digest(t.model)!=m['frozen_initial_sha256']:raise ValueError('Backbone parameter/buffer changed')
        copy_frozen_to_ema(t.model,t.ema.ema)
        if state_digest(t.ema.ema)!=m['frozen_initial_sha256']:raise ValueError('Frozen EMA drift')
        observations.append(dict(step=len(observations)+1,backbone_unchanged=True,bn_fixed=True,actual_threads=torch.get_num_threads(),ema_frozen_unchanged=True))
        if len(observations)==10:
            if state_digest(t.model,False)==snap['other_initial']:raise ValueError('No neck/head update')
            write_record(Path(t.save_dir).parent/'first-ten-steps.json',dict(status='ten_real_updates_backbone_fixed_neck_head_changed',steps=observations.copy(),partition=snap['partition'],
                frozen_sha256=m['frozen_initial_sha256'],training_admitted=False,promotable=False,
                inputs={str(q.resolve()):file_sha256(q) for q in [OUT/'protocol.json',OUT/'model-preflight.json',Path(t.save_dir)/'args.yaml']}))
        return result
    def train_frozen(model,*args,**kwargs):
        if kwargs.get('freeze') not in (None,0):raise ValueError('Unexpected existing freeze')
        kwargs['freeze']=list(FROZEN)
        return train(model,*args,**kwargs)
    with patch.object(YOLO,'train',train_frozen),patch.object(DetectionTrainer,'_setup_train',setup_checked),patch.object(DetectionTrainer,'optimizer_step',step_checked):
        result=cpu.worker(key)
    a=yaml.safe_load((Path(result['weights']).parent.parent/'args.yaml').read_text())
    if a['freeze']!=FROZEN or len(observations)!=480:raise ValueError('Actual freeze/steps mismatch')
    terminal=YOLO(result['weights']).model
    equal=state_digest(terminal,half=True)==m['checkpoint_half_frozen_sha256']
    changed=state_digest(snap['trainer'].model,False)!=snap['other_initial']
    if not equal or not changed:raise ValueError('Terminal model did not preserve intended update partition')
    paths=[root/'completion.json',root/'tensor-verification.json',root/'thread-verification.json',OUT/'model-preflight.json']
    write_record(root/'backbone-verification.json',dict(status='all_480_steps_and_terminal_backbone_verified',steps=observations,nonbackbone_changed=changed,terminal_frozen_half_equal=equal,
        partition=snap['partition'],training_admitted=False,promotable=False,inputs={str(q.resolve()):file_sha256(q) for q in paths}))
    return result

def summarize():
    records=[];transitions=[];queue=[];deps={}
    for seed,key,oldkey in zip((7,17,27),KEYS,SOURCE_KEYS):
        verified_unit(key)
        cp=OUT/'evaluation-v1/units'/key/'completion.json';c=checked(cp);r=checked(c['result']);records.append(r)
        op=SOURCE/'evaluation-v1/units'/oldkey/'completion.json';o=checked(op);old=checked(o['result']);by={(x['pair_id'],x['variant']):x for x in old['rows']}
        for row in r['rows']:transitions.append(dict(seed=seed,reference='routed',pair_id=row['pair_id'],variant=row['variant'],instances=runtime.evaluation.compare_truth(by[row['pair_id'],row['variant']],row)))
        for row in r['negative_rows']:
            for pred in row['predictions']:queue.append(dict(seed=seed,view_id=row['view_id'],image_sha256=row['image_sha256'],prediction=pred,review_status='pending'))
        for q in (cp,op,Path(c['result']),Path(o['result']),OUT/'training'/key/'backbone-verification.json'):deps[str(q.resolve())]=file_sha256(q)
    from scripts.vision.exposure_order_retention import PRIOR
    from scripts.vision.finalize_exposure_diagnosis import aggregate,policy_checks
    pp=PRIOR/'protocol.json';p=checked(pp);hp=Path(p['evaluation']['historical_reference']);h=checked(hp)
    refs=[PRIOR/f'evaluation-retained_reference-450-{s}.json' for s in (7,17,27)]
    group=runtime.evaluation.aggregate(records);gates=policy_checks(group,aggregate([checked(q) for q in refs]),h['historical_A'],p)
    for item in gates['checks']:
        if item.get('reference')=='same_budget_R':item['reference']='fixed_retained_reference_450_not_same_budget'
    for q in (pp,hp,*refs):deps[str(q.resolve())]=file_sha256(q)
    return write_record(OUT/'evaluation-v1/summary.json',dict(status='numerical_complete_visual_review_pending',group=group,gates=gates,instance_comparisons=transitions,
        negative_fp_review_queue=queue,matching_conflicts={k:r['matching_conflicts'] for k,r in zip(KEYS,records)},selected_candidate=None,training_admitted=False,promotable=False,inputs=deps))

def configure():
    cpu.OUT=OUT;cpu.KEYS=KEYS;augmentation.OUT=OUT;augmentation.KEYS=KEYS
    runtime.OUT=OUT;runtime.KEYS=KEYS;runtime.freeze=freeze;runtime.loader=loader;runtime.summarize=summarize
    runtime.MODULE='scripts.vision.routed_backbone_control'
    runtime.TESTS=tuple(dict.fromkeys(runtime.TESTS+('tests.test_routed_backbone_control','tests.test_contrast_transfer_control','tests.test_locked_cpu_threads','tests.test_contrast_cpu4_control','tests.test_contrast_cpu4_review')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);ap.add_argument('--eval-worker',choices=KEYS);a=ap.parse_args();configure()
    if a.worker:worker(a.worker)
    elif a.eval_worker:verified_unit(a.eval_worker);runtime.eval_worker(a.eval_worker)
    else:
        prepare()
        if a.train:runtime.train()
        else:print('PREFLIGHT_ONLY_NO_TRAINING')
