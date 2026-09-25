"""Material-routed control with a fixed 0.1 backbone AdamW LR multiplier."""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch
from scripts.vision import routed_backbone_bn_control as reusable
from scripts.vision import material_routed_contrast_control as routed
from scripts.vision import routed_backbone_control as backbone
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

SOURCE=routed.OUT;SOURCE_KEYS=tuple(routed.KEYS)
OUT=SOURCE.parent/'routed-small-backbone-control-v1'
KEYS=tuple(f'routed-small-backbone-480-{s}' for s in (7,17,27))
runtime=routed.runtime;checked=routed.checked
VERSION='backbone-lr-multiplier-0.1-normal-bn-v1'

def split_groups(model,groups):
    names={id(p):n for n,p in model.named_parameters()};seen=[];out=[]
    for group in groups:
        if group['lr']!=.00025:raise ValueError('Unexpected base LR')
        for is_backbone in (True,False):
            params=[]
            for p in group['params']:
                if id(p) not in names:raise ValueError('Unknown parameter')
                if backbone.frozen_name(names[id(p)])==is_backbone:params.append(p);seen.append(id(p))
            if params:
                g={k:v for k,v in group.items() if k!='params'}
                g.update(params=params,lr=.000025 if is_backbone else .00025,region='backbone' if is_backbone else 'neck_head')
                if 'initial_lr' in g:g['initial_lr']=g['lr']
                out.append(g)
    if len(seen)!=len(set(seen)) or set(seen)!=set(names):raise ValueError('Parameter coverage mismatch')
    return out

def assert_optimizer(model,opt):
    import torch
    names={id(p):n for n,p in model.named_parameters()};seen=[]
    for g in opt.param_groups:
        expected=.000025 if g['region']=='backbone' else .00025
        if abs(g['lr']-expected)>1e-12:raise ValueError('Actual group LR drift')
        for p in g['params']:
            n=names[id(p)];seen.append(id(p))
            if backbone.frozen_name(n)!=(g['region']=='backbone'):raise ValueError('Misrouted parameter')
            if p.requires_grad!=('.dfl' not in n):raise ValueError('Unexpected frozen parameter')
    if len(seen)!=len(set(seen)) or set(seen)!=set(names):raise ValueError('Missing parameter')
    if any(not m.training for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)):raise ValueError('BN mode changed')

def validate(s,p):
    for f in ('pool_rows','names','initialization','evaluation','environment','held_members','training_config'):
        if s[f]!=p[f]:raise ValueError('Non-LR change '+f)
    expected=copy.deepcopy(s['configuration']);expected['parameter_update_policy']=VERSION
    if p['configuration']!=expected:raise ValueError('Configuration drift')
    for f in ('schedules','exposures','windows','listings','contrast'):
        if p[f]!={k:s[f][o] for k,o in zip(KEYS,SOURCE_KEYS)}:raise ValueError('Input drift '+f)

def freeze():
    s=checked(SOURCE/'protocol.json');path=OUT/'protocol.json'
    checked(SOURCE/'audit-v1/completion.json')
    if path.exists():p=checked(path);validate(s,p);return p
    OUT.mkdir(parents=True,exist_ok=True);p=copy.deepcopy(s)
    for f in ('identity','inputs','routing_ledger'):p.pop(f,None)
    for f in ('schedules','exposures','windows','listings','contrast'):p[f]={k:copy.deepcopy(s[f][o]) for k,o in zip(KEYS,SOURCE_KEYS)}
    p['configuration']['parameter_update_policy']=VERSION
    paths=[SOURCE/'protocol.json',SOURCE/'audit-v1/completion.json',Path(__file__),Path('tests/test_routed_small_backbone.py'),OUT/'research-plan-zh.md']
    for k in SOURCE_KEYS:
        for n in ('completion.json','tensor-verification.json','thread-verification.json'):
            q=SOURCE/'training'/k/n;checked(q);paths.append(q)
        c=checked(SOURCE/'training'/k/'completion.json');ex=checked(c['exposure_path'])
        if c['optimizer_steps']!=480 or ex['actual']!=s['schedules'][k]:raise ValueError('Reference exposure mismatch')
        paths += [Path(c['weights']),Path(c['exposure_path']),SOURCE/'actual-preflight'/f'{k}.json']
    deps=dict(s['inputs']);deps.update({str(q.resolve()):file_sha256(q) for q in paths})
    p.update(status='small_backbone_lr_frozen_preflight_pending',arm='small_backbone_lr',direct_control=str(SOURCE),design='Same routed inputs and independent v2.11 initialization; backbone LR 0.000025 versus neck/head 0.00025, 480 steps, normal BN and default EMA.',rationale='Test bounded backbone adaptation rather than full freezing; multiplier is prospective heuristic, not optimum.',limits='AdamW LR also scales decoupled weight decay; this is a regional update-rate strategy, not pure gradient magnitude causality. No new data or sealed tests.',training_admitted=False,promotable=False,inputs=deps)
    validate(s,p);return write_record(path,p)

def loader(p,key,owner):
    dataset,raw=routed.transform_runtime.loader(p,key,owner)
    expected=checked(SOURCE/'actual-preflight'/f'{SOURCE_KEYS[KEYS.index(key)]}.json')['tensor_records']
    class Wrapped:
        def __len__(self):return len(raw)
        def __getattr__(self,n):return getattr(raw,n)
        def __iter__(self):
            for j,b in enumerate(raw):
                if runtime.OBSERVED[key][-1]!=expected[owner.epoch*10+j]:raise ValueError('Tensor differs from routed control')
                yield b
    return dataset,Wrapped()

def verified_unit(k):
    c=checked(OUT/'training'/k/'completion.json');v=checked(OUT/'training'/k/'regional-lr-verification.json')
    for n in ('tensor-verification.json','thread-verification.json'):checked(OUT/'training'/k/n)
    if c['optimizer_steps']!=480 or len(v['steps'])!=480:raise ValueError('Incomplete training')
    return c

def worker(k):
    import torch
    from ultralytics.models.yolo.detect import DetectionTrainer
    freeze();runtime.launch_gate()
    if (OUT/'training'/k/'completion.json').exists():return verified_unit(k)
    build=DetectionTrainer.build_optimizer;step=DetectionTrainer.optimizer_step;observations=[];snap={}
    def build_checked(t,model,*a,**kw):
        opt=build(t,model,*a,**kw)
        if opt.state:raise ValueError('Optimizer already used')
        opt.param_groups[:]=split_groups(model,opt.param_groups)
        snap['initial']=backbone.state_digest(model)
        return opt
    def step_checked(t,*a,**kw):
        assert_optimizer(t.model,t.optimizer);r=step(t,*a,**kw)
        observations.append(dict(step=len(observations)+1,lrs=[g['lr'] for g in t.optimizer.param_groups],regions=[g['region'] for g in t.optimizer.param_groups],threads=torch.get_num_threads()))
        if len(observations)==10:
            if backbone.state_digest(t.model)==snap['initial']:raise ValueError('No backbone adaptation')
            write_record(Path(t.save_dir).parent/'first-ten-steps.json',dict(status='real_regional_lr_steps_verified',steps=observations.copy(),training_admitted=False,promotable=False,inputs={str((OUT/'protocol.json').resolve()):file_sha256(OUT/'protocol.json')}))
        return r
    with patch.object(DetectionTrainer,'build_optimizer',build_checked),patch.object(DetectionTrainer,'optimizer_step',step_checked):r=routed.previous.worker(k)
    if len(observations)!=480:raise ValueError('Missing step records')
    write_record(OUT/'training'/k/'regional-lr-verification.json',dict(status='all_regional_lr_steps_verified',steps=observations,training_admitted=False,promotable=False,inputs={str((OUT/'training'/k/'completion.json').resolve()):file_sha256(OUT/'training'/k/'completion.json')}))
    return r

def summarize():
    # Existing generic paired evaluator; comparison identities explicitly rebound.
    with patch.object(reusable,'OUT',OUT),patch.object(reusable,'KEYS',KEYS),patch.object(reusable,'SOURCE',SOURCE),patch.object(reusable,'SOURCE_KEYS',SOURCE_KEYS),patch.object(reusable,'verified_unit',verified_unit):
        records=[];deps={};comparisons=[]
        for k,o in zip(KEYS,SOURCE_KEYS):
            verified_unit(k);cp=OUT/'evaluation-v1/units'/k/'completion.json';c=checked(cp);r=checked(c['result']);records.append(r)
            op=SOURCE/'evaluation-v1/units'/o/'completion.json';oc=checked(op);ref=checked(oc['result']);by={(x['pair_id'],x['variant']):x for x in ref['rows']}
            comparisons.extend(dict(key=k,pair_id=x['pair_id'],variant=x['variant'],instances=runtime.evaluation.compare_truth(by[x['pair_id'],x['variant']],x)) for x in r['rows'])
            for q in (cp,op,Path(c['result']),Path(oc['result'])):deps[str(q.resolve())]=file_sha256(q)
        from scripts.vision.exposure_order_retention import PRIOR
        from scripts.vision.finalize_exposure_diagnosis import aggregate,policy_checks
        pp=PRIOR/'protocol.json';p=checked(pp);hp=Path(p['evaluation']['historical_reference']);h=checked(hp)
        refs=[PRIOR/f'evaluation-retained_reference-450-{s}.json' for s in (7,17,27)]
        group=runtime.evaluation.aggregate(records);gates=policy_checks(group,aggregate([checked(q) for q in refs]),h['historical_A'],p)
        for item in gates['checks']:
            if item.get('reference')=='same_budget_R':item['reference']='fixed_retained_reference_450_not_same_budget'
        for q in (pp,hp,*refs):deps[str(q.resolve())]=file_sha256(q)
        return write_record(OUT/'evaluation-v1/summary.json',dict(status='numerical_complete_visual_review_pending',group=group,gates=gates,instance_comparisons=comparisons,matching_conflicts={k:r['matching_conflicts'] for k,r in zip(KEYS,records)},selected_candidate=None,training_admitted=False,promotable=False,inputs=deps))

def configure():
    routed.previous.OUT=OUT;routed.previous.KEYS=KEYS;routed.transform_runtime.OUT=OUT;routed.transform_runtime.KEYS=KEYS
    runtime.OUT=OUT;runtime.KEYS=KEYS;runtime.freeze=freeze;runtime.loader=loader;runtime.summarize=summarize;runtime.MODULE='scripts.vision.routed_small_backbone_control'
    runtime.TESTS=tuple(dict.fromkeys(runtime.TESTS+('tests.test_routed_small_backbone','tests.test_contrast_transfer_control','tests.test_locked_cpu_threads','tests.test_contrast_cpu4_control')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);ap.add_argument('--eval-worker',choices=KEYS);a=ap.parse_args();configure()
    if a.worker:worker(a.worker)
    elif a.eval_worker:verified_unit(a.eval_worker);runtime.eval_worker(a.eval_worker)
    elif a.train:runtime.train()
    else:runtime.preflight();print('PREFLIGHT_ONLY_NO_TRAINING')
