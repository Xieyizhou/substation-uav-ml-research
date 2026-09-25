"""Original routed tensors with predeclared late LR decay; explicit training."""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch
from scripts.vision import material_routed_contrast_control as routed
from scripts.vision import routed_small_backbone_control as reporting
from scripts.vision import routed_late_decay_policy as policy
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

SOURCE=routed.OUT;SOURCE_KEYS=tuple(routed.KEYS)
OUT=SOURCE.parent/'routed-late-decay-control-v1'
KEYS=tuple(f'routed-late-decay-480-{s}' for s in (7,17,27))
runtime=routed.runtime;checked=routed.checked

def validate(s,p):
    for f in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config'):
        if p[f]!=s[f]:raise ValueError('Non-schedule change '+f)
    cfg=copy.deepcopy(s['configuration']);cfg['learning_rate_policy']=policy.VERSION
    if p['configuration']!=cfg or p['learning_rates']!=policy.sequence():raise ValueError('LR policy drift')
    for f in ('schedules','exposures','windows','listings','contrast'):
        if p[f]!={k:s[f][o] for k,o in zip(KEYS,SOURCE_KEYS)}:raise ValueError('Exposure drift')

def freeze():
    s=checked(SOURCE/'protocol.json');dest=OUT/'protocol.json'
    audit=SOURCE.parent/'routed-fixed-bn-control-v1/audit-v1/completion.json'
    if checked(audit)['status']!='fixed_bn_review_complete_candidate_failed':raise ValueError('Prior audit incomplete')
    if dest.exists():p=checked(dest);validate(s,p);return p
    OUT.mkdir(parents=True,exist_ok=True);p=copy.deepcopy(s)
    for f in ('identity','inputs','routing_ledger'):p.pop(f,None)
    for f in ('schedules','exposures','windows','listings','contrast'):
        p[f]={k:copy.deepcopy(s[f][o]) for k,o in zip(KEYS,SOURCE_KEYS)}
    p['configuration']['learning_rate_policy']=policy.VERSION;p['learning_rates']=policy.sequence()
    from ultralytics.engine import trainer as engine
    paths=[SOURCE/'protocol.json',SOURCE/'audit-v1/completion.json',audit,OUT/'design-zh.md',Path(__file__),Path(policy.__file__),Path(engine.__file__),Path(reporting.__file__),Path('tests/test_routed_late_decay_policy.py')]
    for key in SOURCE_KEYS:
        cp=SOURCE/'training'/key/'completion.json';c=checked(cp);ex=checked(c['exposure_path'])
        if c['optimizer_steps']!=480 or ex['actual']!=s['schedules'][key]:raise ValueError('Reference invalid')
        paths.extend([cp,Path(c['weights']),Path(c['exposure_path']),SOURCE/'actual-preflight'/f'{key}.json'])
    deps=dict(s['inputs']);deps.update({str(q.resolve()):file_sha256(q) for q in paths})
    p.update(status='late_decay_frozen_preflight_pending',arm='routed_late_decay',direct_control=str(SOURCE),
        design='Same routed tensors, labels, batches, normal BN and 480 steps. First 240 steps LR 0.00025; remaining 24 ten-step epochs linearly descend to 0.000125.',
        limits='Tests a late-update strategy including lower cumulative LR, not a pure order effect. No seed or endpoint selection. No new data or sealed evaluation.',
        training_admitted=False,promotable=False,inputs=deps)
    validate(s,p);return write_record(dest,p)

def loader(p,key,owner):
    dataset,raw=routed.transform_runtime.loader(p,key,owner)
    ref=checked(SOURCE/'actual-preflight'/f'{SOURCE_KEYS[KEYS.index(key)]}.json')['tensor_records']
    class Wrapped:
        def __len__(self):return len(raw)
        def __getattr__(self,n):return getattr(raw,n)
        def __iter__(self):
            for j,b in enumerate(raw):
                if runtime.OBSERVED[key][-1]!=ref[owner.epoch*10+j]:raise ValueError('Routed tensor drift')
                yield b
    return dataset,Wrapped()

def verified_unit(k):
    c=checked(OUT/'training'/k/'completion.json');v=checked(OUT/'training'/k/'lr-verification.json')
    t=checked(OUT/'training'/k/'tensor-verification.json');checked(OUT/'training'/k/'thread-verification.json')
    if c['optimizer_steps']!=480 or v['learning_rates']!=policy.sequence() or t['records']!=checked(OUT/'actual-preflight'/f'{k}.json')['tensor_records']:raise ValueError('Incomplete unit')
    return c

def worker(k):
    import torch
    from ultralytics.models.yolo.detect import DetectionTrainer
    freeze();runtime.launch_gate()
    if (OUT/'training'/k/'completion.json').exists():return verified_unit(k)
    original=DetectionTrainer.optimizer_step;rates=[]
    def setup(t):
        t.lf=policy.factor
        t.scheduler=torch.optim.lr_scheduler.LambdaLR(t.optimizer,lr_lambda=t.lf)
    def step(t,*a,**kw):
        expected=policy.sequence()[len(rates)]
        if any(g['lr']!=expected for g in t.optimizer.param_groups):raise ValueError('Actual LR differs from frozen sequence')
        result=original(t,*a,**kw);rates.append(expected)
        if len(rates)==10:
            write_record(Path(t.save_dir).parent/'first-ten-steps.json',dict(status='real_late_decay_training_started',learning_rates=rates.copy(),training_admitted=False,promotable=False,inputs={str((OUT/'protocol.json').resolve()):file_sha256(OUT/'protocol.json')}))
        return result
    with patch.object(DetectionTrainer,'_setup_scheduler',setup),patch.object(DetectionTrainer,'optimizer_step',step):result=routed.previous.worker(k)
    if rates!=policy.sequence():raise ValueError('Incomplete LR observations')
    write_record(OUT/'training'/k/'lr-verification.json',dict(status='all_480_learning_rates_verified',learning_rates=rates,training_admitted=False,promotable=False,inputs={str((OUT/'training'/k/'completion.json').resolve()):file_sha256(OUT/'training'/k/'completion.json')}))
    return result

def summarize():
    with patch.object(reporting,'OUT',OUT),patch.object(reporting,'KEYS',KEYS),patch.object(reporting,'SOURCE',SOURCE),patch.object(reporting,'SOURCE_KEYS',SOURCE_KEYS),patch.object(reporting,'verified_unit',verified_unit):return reporting.summarize()

def configure():
    routed.previous.OUT=OUT;routed.previous.KEYS=KEYS;routed.transform_runtime.OUT=OUT;routed.transform_runtime.KEYS=KEYS
    runtime.OUT=OUT;runtime.KEYS=KEYS;runtime.freeze=freeze;runtime.loader=loader;runtime.summarize=summarize;runtime.MODULE='scripts.vision.routed_late_decay_control'
    runtime.TESTS=tuple(dict.fromkeys(runtime.TESTS+('tests.test_routed_late_decay_policy','tests.test_contrast_transfer_control','tests.test_locked_cpu_threads','tests.test_contrast_cpu4_control')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);ap.add_argument('--eval-worker',choices=KEYS);a=ap.parse_args();configure()
    if a.worker:worker(a.worker)
    elif a.eval_worker:verified_unit(a.eval_worker);runtime.eval_worker(a.eval_worker)
    elif a.train:runtime.train()
    else:runtime.launch_gate(create=True);print('PREFLIGHT_ONLY_NO_TRAINING')
