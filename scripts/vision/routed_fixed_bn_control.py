"""Full adaptation with fixed backbone BN buffers; explicit training only."""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch
from scripts.vision import material_routed_contrast_control as routed
from scripts.vision import routed_small_backbone_control as reporting
from scripts.vision import routed_fixed_bn_policy as policy
from scripts.vision import routed_backbone_control as backbone
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

SOURCE=routed.OUT;SOURCE_KEYS=tuple(routed.KEYS)
OUT=SOURCE.parent/'routed-fixed-bn-control-v1'
KEYS=tuple(f'routed-fixed-bn-480-{s}' for s in (7,17,27))
runtime=routed.runtime;checked=routed.checked

def validate(s,p):
    for f in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config'):
        if s[f]!=p[f]:raise ValueError('Non-BN change '+f)
    cfg=copy.deepcopy(s['configuration']);cfg['normalization_policy']=policy.VERSION
    if p['configuration']!=cfg:raise ValueError('Configuration change')
    for f in ('schedules','exposures','windows','listings','contrast'):
        if p[f]!={k:s[f][o] for k,o in zip(KEYS,SOURCE_KEYS)}:raise ValueError('Exposure change')

def freeze():
    s=checked(SOURCE/'protocol.json');dest=OUT/'protocol.json'
    audit=SOURCE.parent/'routed-scale-transfer-control-v1/audit-v2/completion.json'
    a=checked(audit)
    if a['status']!='scale_error_review_and_fit_diagnosis_complete_candidate_failed':raise ValueError('Prior audit incomplete')
    if dest.exists():p=checked(dest);validate(s,p);return p
    OUT.mkdir(parents=True,exist_ok=True);p=copy.deepcopy(s)
    for f in ('identity','inputs','routing_ledger'):p.pop(f,None)
    for f in ('schedules','exposures','windows','listings','contrast'):
        p[f]={k:copy.deepcopy(s[f][o]) for k,o in zip(KEYS,SOURCE_KEYS)}
    p['configuration']['normalization_policy']=policy.VERSION
    from ultralytics.engine import trainer as engine
    paths=[SOURCE/'protocol.json',SOURCE/'audit-v1/completion.json',audit,Path(__file__),Path(policy.__file__),Path(engine.__file__),Path(reporting.__file__),Path('tests/test_routed_fixed_bn_policy.py')]
    for key in SOURCE_KEYS:
        cp=SOURCE/'training'/key/'completion.json';c=checked(cp);ex=checked(c['exposure_path'])
        if c['optimizer_steps']!=480 or ex['actual']!=s['schedules'][key]:raise ValueError('Reference invalid')
        paths.extend([cp,Path(c['weights']),Path(c['exposure_path']),SOURCE/'actual-preflight'/f'{key}.json'])
    deps=dict(s['inputs']);deps.update({str(q.resolve()):file_sha256(q) for q in paths})
    p.update(status='fixed_bn_frozen_preflight_pending',arm='fixed_backbone_bn_full_adaptation',
        design='Original routed tensors, exposures, LR and 480 steps. Only backbone BN uses initialized fixed running statistics in forward; all usual parameters including BN affine remain trainable. Neck/head BN updates normally. Copy only fixed buffers to EMA.',
        rationale='Prior full backbone freezing confounded normalization and feature adaptation. This tests fixed normalization while preserving learning, not an established cause.',
        limits='Changes both batch-stat forward normalization and running-buffer updates; cannot separate their contributions. No scale/grayscale/reflection stacking, new data, sealed testing or promotion. Fixed statistics may impair adaptation.',
        direct_control=str(SOURCE),training_admitted=False,promotable=False,inputs=deps)
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
    c=checked(OUT/'training'/k/'completion.json');v=checked(OUT/'training'/k/'bn-verification.json')
    t=checked(OUT/'training'/k/'tensor-verification.json');checked(OUT/'training'/k/'thread-verification.json')
    if c['optimizer_steps']!=480 or len(v['steps'])!=480 or t['records']!=checked(OUT/'actual-preflight'/f'{k}.json')['tensor_records']:raise ValueError('Incomplete unit')
    return c

def worker(k):
    from ultralytics.models.yolo.detect import DetectionTrainer
    freeze();runtime.launch_gate()
    if (OUT/'training'/k/'completion.json').exists():return verified_unit(k)
    setup=DetectionTrainer._setup_train;mode=DetectionTrainer._model_train;step=DetectionTrainer.optimizer_step
    snap={};steps=[]
    def setup_checked(t,*a,**kw):
        result=setup(t,*a,**kw);backbone.topology(t.model)
        snap['buffers']=policy.digest(t.model);snap['weights']=backbone.state_digest(t.model)
        policy.apply(t.model);policy.copy_buffers(t.model,t.ema.ema)
        return result
    def mode_checked(t,*a,**kw):
        result=mode(t,*a,**kw);policy.apply(t.model);return result
    def step_checked(t,*a,**kw):
        policy.check(t.model,snap['buffers']);result=step(t,*a,**kw)
        policy.check(t.model,snap['buffers']);policy.copy_buffers(t.model,t.ema.ema)
        if policy.digest(t.ema.ema)!=snap['buffers']:raise ValueError('EMA fixed buffers drift')
        steps.append(dict(step=len(steps)+1,buffers=snap['buffers']))
        if len(steps)==10:
            if backbone.state_digest(t.model)==snap['weights']:raise ValueError('No feature adaptation')
            write_record(Path(t.save_dir).parent/'first-ten-steps.json',dict(status='real_fixed_bn_steps_and_feature_adaptation_verified',steps=steps.copy(),training_admitted=False,promotable=False,inputs={str((OUT/'protocol.json').resolve()):file_sha256(OUT/'protocol.json')}))
        return result
    with patch.object(DetectionTrainer,'_setup_train',setup_checked),patch.object(DetectionTrainer,'_model_train',mode_checked),patch.object(DetectionTrainer,'optimizer_step',step_checked):
        result=routed.previous.worker(k)
    if len(steps)!=480:raise ValueError('Missing steps')
    write_record(OUT/'training'/k/'bn-verification.json',dict(status='fixed_buffers_full_adaptation_verified',steps=steps,training_admitted=False,promotable=False,inputs={str((OUT/'training'/k/'completion.json').resolve()):file_sha256(OUT/'training'/k/'completion.json')}))
    return result

def summarize():
    with patch.object(reporting,'OUT',OUT),patch.object(reporting,'KEYS',KEYS),patch.object(reporting,'SOURCE',SOURCE),patch.object(reporting,'SOURCE_KEYS',SOURCE_KEYS),patch.object(reporting,'verified_unit',verified_unit):return reporting.summarize()

def configure():
    routed.previous.OUT=OUT;routed.previous.KEYS=KEYS;routed.transform_runtime.OUT=OUT;routed.transform_runtime.KEYS=KEYS
    runtime.OUT=OUT;runtime.KEYS=KEYS;runtime.freeze=freeze;runtime.loader=loader;runtime.summarize=summarize;runtime.MODULE='scripts.vision.routed_fixed_bn_control'
    runtime.TESTS=tuple(dict.fromkeys(runtime.TESTS+('tests.test_routed_fixed_bn_policy','tests.test_contrast_transfer_control','tests.test_locked_cpu_threads','tests.test_contrast_cpu4_control')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);ap.add_argument('--eval-worker',choices=KEYS);a=ap.parse_args();configure()
    if a.worker:worker(a.worker)
    elif a.eval_worker:verified_unit(a.eval_worker);runtime.eval_worker(a.eval_worker)
    elif a.train:runtime.train()
    else:runtime.launch_gate(create=True);print('PREFLIGHT_ONLY_NO_TRAINING')
