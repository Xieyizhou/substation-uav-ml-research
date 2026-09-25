"""Frozen backbone weights with live BN: independent, bounded training arm."""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.vision import routed_backbone_control as old
from scripts.vision import backbone_bn_update_policy as policy
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

SOURCE=old.OUT
SOURCE_KEYS=old.KEYS
OUT=SOURCE.parent/'routed-backbone-bn-control-v1'
KEYS=tuple(f'routed-backbone-bn-480-{s}' for s in (7,17,27))
runtime=old.runtime
checked=old.checked

def validate(source,p):
    for f in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config'):
        if source[f]!=p[f]:raise ValueError('Non-BN change '+f)
    c=copy.deepcopy(source['configuration']);c['parameter_update_policy']=policy.VERSION
    if p['configuration']!=c:raise ValueError('Configuration changed')
    for f in ('schedules','exposures','windows','listings','contrast'):
        if p[f]!={k:source[f][o] for k,o in zip(KEYS,SOURCE_KEYS)}:raise ValueError('Input changed '+f)
    if p['backbone_policy']!=dict(version=policy.VERSION,indices=old.FROZEN,batchnorm='train_running_stats_update_affine_fixed',ema='copy_entire_backbone_exactly_after_each_update',other_parameters='train_except_existing_dfl'):raise ValueError('BN policy changed')

def freeze():
    source=checked(SOURCE/'protocol.json')
    fit=SOURCE.parent/'routed-backbone-fit-diagnosis-v1'
    c=checked(fit/'completion.json')
    if c['status']!='backbone_result_and_followup_fit_research_complete':raise ValueError('Previous research incomplete')
    for k in SOURCE_KEYS:old.verified_unit(k)
    path=OUT/'protocol.json'
    if path.exists():p=checked(path);validate(source,p);return p
    OUT.mkdir(parents=True,exist_ok=True);p=copy.deepcopy(source)
    for f in ('identity','inputs'):p.pop(f,None)
    for f in ('schedules','exposures','windows','listings','contrast'):p[f]={k:copy.deepcopy(source[f][o]) for k,o in zip(KEYS,SOURCE_KEYS)}
    p['configuration']['parameter_update_policy']=policy.VERSION
    p['backbone_policy']=dict(version=policy.VERSION,indices=old.FROZEN,batchnorm='train_running_stats_update_affine_fixed',ema='copy_entire_backbone_exactly_after_each_update',other_parameters='train_except_existing_dfl')
    paths=[SOURCE/'protocol.json',fit/'completion.json',fit/'research-conclusion-zh.md',Path(__file__),Path(policy.__file__),Path('tests/test_backbone_bn_update_policy.py'),OUT/'research-plan-zh.md']
    for k in SOURCE_KEYS:
        paths += [SOURCE/'training'/k/n for n in ('completion.json','backbone-verification.json','tensor-verification.json','thread-verification.json')]
    deps=dict(source['inputs']);deps.update({str(q.resolve()):file_sha256(q) for q in paths})
    p.update(status='bn_control_frozen_preflight_pending',arm='frozen_parameters_live_bn',direct_control=str(SOURCE),design='Only activate backbone BN training; same frozen affine/other parameters and exact backbone-to-EMA copy policy. Same input tensors and 480-step budget.',rationale='Separate BN training behavior from complete backbone freeze after all-seed fitting and development degradation.',limits='BN train mode changes both batch normalization during training and running statistics. Not a pure inference-statistics intervention or unique-cause claim.',inputs=deps,training_admitted=False,promotable=False)
    validate(source,p);return write_record(path,p)

def loader(p,key,owner):
    dataset,raw=old.augmentation.loader(p,key,owner)
    expected=checked(SOURCE/'actual-preflight'/f'{SOURCE_KEYS[KEYS.index(key)]}.json')['tensor_records']
    class Wrapped:
        def __len__(self):return len(raw)
        def __getattr__(self,n):return getattr(raw,n)
        def __iter__(self):
            for j,b in enumerate(raw):
                if runtime.OBSERVED[key][-1]!=expected[owner.epoch*10+j]:raise ValueError('Actual input changed')
                yield b
    return dataset,Wrapped()

def prepare():
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    p,_=runtime.preflight();path=OUT/'model-preflight.json'
    if path.exists():return p,checked(path)
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('No optimizer')),patch.object(torch.Tensor,'backward',side_effect=AssertionError('No backward')),patch.object(YOLO,'train',side_effect=AssertionError('No training')),patch.object(YOLO,'val',side_effect=AssertionError('No validation')):
        model=YOLO(p['initialization']['path']).model.float();old.topology(model)
        for n,v in model.named_parameters():v.requires_grad_(not(old.frozen_name(n) or '.dfl' in n))
        owner=SimpleNamespace(model=model,freeze_layer_names=[f'model.{i}.' for i in old.FROZEN]+['.dfl'])
        DetectionTrainer._model_train(owner);partition=policy.activate(model)
        fixed=policy.digest(model);before=policy.digest(model,True);half=policy.digest(model,half=True)
        key=KEYS[0];runtime.OBSERVED[key]=[];_,batches=loader(p,key,SimpleNamespace(epoch=0));b=next(iter(batches))
        with torch.no_grad():model(b['img'].float()/255.)
        if policy.digest(model)!=fixed or policy.digest(model,True)==before:raise ValueError('Forward policy failed')
        policy.assert_policy(model)
    return p,write_record(path,dict(status='real_forward_bn_update_parameters_fixed_verified',fixed_sha256=fixed,fixed_half_sha256=half,initial_bn_sha256=before,partition=partition,actual_draws_verified=8640,optimizer_created=False,backward_executed=False,validation_run=False,training_admitted=False,promotable=False,inputs={str(q.resolve()):file_sha256(q) for q in (OUT/'protocol.json',OUT/'entry-ready.json')}))

def verified_unit(key):
    c=checked(OUT/'training'/key/'completion.json');v=checked(OUT/'training'/key/'bn-verification.json')
    for n in ('tensor-verification.json','thread-verification.json'):checked(OUT/'training'/key/n)
    if c['optimizer_steps']!=480 or len(v['steps'])!=480 or not v['terminal_fixed_equal']:raise ValueError('Incomplete BN unit')
    return c

def worker(key):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    p=freeze();m=checked(OUT/'model-preflight.json');runtime.launch_gate()
    root=OUT/'training'/key
    if (root/'completion.json').exists():return verified_unit(key)
    train_mode=DetectionTrainer._model_train;setup=DetectionTrainer._setup_train;step=DetectionTrainer.optimizer_step;train=YOLO.train
    observations=[];snap={}
    def mode(t):train_mode(t);policy.activate(t.model)
    def setup_checked(t):
        setup(t);old.topology(t.model);policy.assert_policy(t.model)
        if policy.digest(t.model)!=m['fixed_sha256'] or policy.digest(t.model,True)!=m['initial_bn_sha256']:raise ValueError('Initial state changed')
        snap.update(other=old.state_digest(t.model,False),previous_bn=policy.digest(t.model,True))
    def step_checked(t,*a,**kw):
        policy.assert_policy(t.model);r=step(t,*a,**kw)
        if policy.digest(t.model)!=m['fixed_sha256']:raise ValueError('Frozen parameter drift')
        bn=policy.digest(t.model,True)
        if bn==snap['previous_bn']:raise ValueError('BN not updated')
        snap['previous_bn']=bn;policy.sync_ema(t.model,t.ema.ema)
        observations.append(dict(step=len(observations)+1,fixed_sha256=policy.digest(t.model),bn_sha256=bn,actual_threads=torch.get_num_threads(),ema_exact=True))
        if len(observations)==10:
            if old.state_digest(t.model,False)==snap['other']:raise ValueError('Head did not update')
            write_record(Path(t.save_dir).parent/'first-ten-steps.json',dict(status='ten_real_updates_fixed_parameters_live_bn_verified',steps=observations.copy(),training_admitted=False,promotable=False,inputs={str(q.resolve()):file_sha256(q) for q in (OUT/'protocol.json',OUT/'model-preflight.json',Path(t.save_dir)/'args.yaml')}))
        return r
    def train_fixed(model,*a,**kw):
        if kw.get('freeze') not in (None,0):raise ValueError('Unexpected freeze')
        kw['freeze']=old.FROZEN[:];return train(model,*a,**kw)
    with patch.object(DetectionTrainer,'_model_train',mode),patch.object(DetectionTrainer,'_setup_train',setup_checked),patch.object(DetectionTrainer,'optimizer_step',step_checked),patch.object(YOLO,'train',train_fixed):result=old.cpu.worker(key)
    terminal=YOLO(result['weights']).model
    equal=policy.digest(terminal,half=True)==m['fixed_half_sha256']
    if not equal or len(observations)!=480:raise ValueError('Terminal policy failed')
    paths=[root/n for n in ('completion.json','tensor-verification.json','thread-verification.json')]+[OUT/'model-preflight.json']
    write_record(root/'bn-verification.json',dict(status='all_steps_verified',steps=observations,terminal_fixed_equal=equal,training_admitted=False,promotable=False,inputs={str(q.resolve()):file_sha256(q) for q in paths}))
    return result

def summarize():
    # Reuse fixed numerical gates and comparison, with explicit current identities.
    with patch.object(old,'OUT',OUT),patch.object(old,'KEYS',KEYS),patch.object(old,'SOURCE',SOURCE),patch.object(old,'SOURCE_KEYS',SOURCE_KEYS),patch.object(old,'verified_unit',verified_unit):
        # Original summarizer binds a backbone-verification file; supply no alias.
        return summary_impl()

def summary_impl():
    records=[];deps={};comparisons=[]
    for k,o in zip(KEYS,SOURCE_KEYS):
        verified_unit(k)
        cp=OUT/'evaluation-v1/units'/k/'completion.json';c=checked(cp);r=checked(c['result']);records.append(r)
        op=SOURCE/'evaluation-v1/units'/o/'completion.json';oc=checked(op);prior=checked(oc['result']);by={(x['pair_id'],x['variant']):x for x in prior['rows']}
        comparisons.extend(dict(key=k,pair_id=x['pair_id'],variant=x['variant'],instances=runtime.evaluation.compare_truth(by[x['pair_id'],x['variant']],x)) for x in r['rows'])
        for q in (cp,op,Path(c['result']),Path(oc['result']),OUT/'training'/k/'bn-verification.json'):deps[str(q.resolve())]=file_sha256(q)
    return write_record(OUT/'evaluation-v1/summary.json',dict(status='numerical_complete_review_and_retention_gates_pending',group=runtime.evaluation.aggregate(records),instance_comparisons=comparisons,matching_conflicts={k:r['matching_conflicts'] for k,r in zip(KEYS,records)},selected_candidate=None,training_admitted=False,promotable=False,inputs=deps))

def configure():
    old.cpu.OUT=OUT;old.cpu.KEYS=KEYS;old.augmentation.OUT=OUT;old.augmentation.KEYS=KEYS
    runtime.OUT=OUT;runtime.KEYS=KEYS;runtime.freeze=freeze;runtime.loader=loader;runtime.summarize=summary_impl;runtime.MODULE='scripts.vision.routed_backbone_bn_control'
    runtime.TESTS=tuple(dict.fromkeys(runtime.TESTS+('tests.test_backbone_bn_update_policy','tests.test_routed_backbone_control','tests.test_contrast_transfer_control','tests.test_locked_cpu_threads','tests.test_contrast_cpu4_control')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);ap.add_argument('--eval-worker',choices=KEYS);a=ap.parse_args();configure()
    if a.worker:worker(a.worker)
    elif a.eval_worker:verified_unit(a.eval_worker);runtime.eval_worker(a.eval_worker)
    else:
        prepare()
        if a.train:runtime.train()
        else:print('PREFLIGHT_ONLY_NO_TRAINING')
