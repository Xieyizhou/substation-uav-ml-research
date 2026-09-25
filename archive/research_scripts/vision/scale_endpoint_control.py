"""Explicit endpoint-scale experiment. Default only freezes/verifies, never trains."""
import argparse,csv,fcntl,subprocess,sys,traceback
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.vision import train_frozen_multiscale as old
from scripts.vision.preflight_frozen_multiscale import forbidden
from scripts.vision.brightness_transfer_runtime import make_dataset,check_log
from scripts.vision.order_retention_runtime import make_loader,check_actual
from scripts.vision.preflight_unified_lighting import tensor_hash
from scripts.vision.frozen_multiscale_runtime import preprocess
from scripts.vision.brightness_lr_retention import check_curve

prior=old.prior
OUT=old.OUT.parent/'scale-endpoint-control-v1'
KEYS=tuple(f'{a}-{s}' for s in (7,17,27) for a in ('small','large'))

def sizes(sequence,arm):
    if Counter(sequence)!={320:150,640:150,960:150} or arm not in ('small','large'):raise ValueError('Invalid source scale plan')
    return [640 if s==640 else (320 if arm=='small' else 960) for s in sequence]

def freeze():
    path=OUT/'protocol.json'
    if path.exists():p=prior.read(path);prior.verify(p);return p
    paths=[Path(__file__).resolve(),old.OUT/'protocol.json',old.OUT/'training/completion.json',old.OUT/'evaluation/summary.json']
    p,source,_=old.contract('fixed-7')
    for key in old.KEYS:
        old.complete(key)
        paths.extend([old.OUT/'training'/key/'completion.json',old.OUT/'evaluation'/f'{key}.json'])
    for path0 in paths[1:]:prior.verify(prior.read(path0))
    for name in ('train_frozen_multiscale.py','frozen_multiscale_runtime.py','brightness_transfer_runtime.py','order_retention_runtime.py','preflight_unified_lighting.py'):
        paths.append(prior.ROOT/'scripts/vision'/name)
    OUT.mkdir(exist_ok=True)
    return prior.frozen(path,dict(status='frozen',source_protocol=p['source_protocol'],environment=source['environment'],
        sizes={k:sizes(p['design']['scale_by_step'][k.rsplit('-',1)[1]],k.rsplit('-',1)[0]) for k in KEYS},
        reference='Verified fixed-640 seeds 7/17/27 from material-multiscale-loader-v1; mixed-scale family secondary context only',
        dose='300 non-640 batches,150 640; same non-640 positions in both arms; original mixed positions retained',
        interpretation='Endpoint strategies at fixed non640 dose; not isolated causal attribution of each mixed-scale batch. All resized from640.',
        max_attempts=3,selected_candidate=None,inputs={str(x):prior.file_sha256(x) for x in paths}))

def inputs(key):
    if key not in KEYS:raise ValueError('Unknown cell')
    p=freeze();seed=int(key.rsplit('-',1)[1]);_,source,base=old.contract(f'fixed-{seed}')
    old.complete(f'fixed-{seed}')
    return p,source,base,seed

def verify_check(key):
    p,source,base,seed=inputs(key)
    path=OUT/'checks'/key/'completion.json';r=prior.read(path);prior.verify(r)
    if r['protocol_identity']!=p['identity'] or r['status']!='real_loader_verified':raise ValueError('Preflight incomplete')
    c=r['expected'];check_actual(source,f'R-clean-{seed}',c['actual']);check_log(source,f'R-clean-{seed}',c['brightness_log'])
    if c['actual']!=base['actual'] or c['brightness_log']!=base['brightness_log'] or len(c['batches'])!=450:raise ValueError('Exposure/brightness drift')
    for i,(a,b) in enumerate(zip(c['batches'],base['batches'],strict=True)):
        if a['size']!=p['sizes'][key][i] or a['effective_shape']!=[6,3,a['size'],a['size']]:raise ValueError('Scale drift')
        if any(a[f]!=b[f] for f in ('step','members','source640_sha256','full_supervision')):raise ValueError('Labels/member/base drift')
        if a['size']==640 and a['effective_tensor_sha256']!=b['effective_tensor_sha256']:raise ValueError('640 drift')
    return c

def attempt_folder(root):
    root.mkdir(parents=True,exist_ok=True)
    if any(prior.read(x)['status']=='semantic_failure' for x in root.glob('attempt-*/failure.json')):raise ValueError('Resolve semantic failure first')
    n=len(list(root.glob('attempt-*')))+1
    if n>3:raise ValueError('Attempt cap reached')
    path=root/f'attempt-{n:03}';path.mkdir();return path

def failure(path,ex):
    prior.frozen(path/'failure.json',dict(status='semantic_failure' if isinstance(ex,ValueError) else 'technical_failure',error=traceback.format_exc()))

def preflight(key):
    root=OUT/'checks'/key
    if (root/'completion.json').exists():verify_check(key);return
    p,source,base,seed=inputs(key);attempt=attempt_folder(root)
    try:
        import torch
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import init_seeds
        torch.set_num_threads(4);init_seeds(seed,deterministic=True)
        logs=[];actual=[];batches=[];lookup={r['image_path']:r['member_id'] for r in source['pool_rows']}
        owner=SimpleNamespace(epoch=0,device=torch.device('cpu'),args=SimpleNamespace(multi_scale=0),stride=32)
        with ExitStack() as stack:
            for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):stack.enter_context(patch.object(obj,name,forbidden))
            loader=make_loader(make_dataset(source,f'R-clean-{seed}',logs),source,f'R-clean-{seed}',owner)
            step=0
            for epoch in range(45):
                owner.epoch=epoch
                for batch in loader:
                    b=base['batches'][step];members=[lookup[x] for x in batch['im_file']]
                    if members!=b['members'] or tensor_hash(batch['img'])!=b['source640_sha256']:raise ValueError('Actual source drift')
                    if any(tensor_hash(batch[f])!=sha for f,sha in b['full_supervision'].items()):raise ValueError('Full label drift')
                    size=p['sizes'][key][step];out=preprocess(owner,batch,size)
                    batches.append({**b,'size':size,'effective_shape':list(out['img'].shape),'effective_tensor_sha256':tensor_hash(out['img'])})
                    actual.extend(members);step+=1
        expected=dict(actual=actual,brightness_log=logs,batches=batches)
        if step!=450:raise ValueError('Incomplete loader')
        prior.frozen(root/'completion.json',dict(status='real_loader_verified',protocol_identity=p['identity'],expected=expected,
            optimizer_created=False,backward_executed=False,training_validation_run=False,
            inputs={str(OUT/'protocol.json'):prior.file_sha256(OUT/'protocol.json')}))
        verify_check(key);print('PREFLIGHT',key,'2700 loads',flush=True)
    except BaseException as ex:failure(attempt,ex);raise

def complete(key):
    expected=verify_check(key);cp=OUT/'training'/key/'completion.json';r=prior.read(cp);prior.verify(r)
    if r['status']!='complete' or r['cell']!=key or r['optimizer_steps']!=450 or r['checkpoint_selection']!='terminal_last_only':raise ValueError('Incomplete training')
    x=prior.read(r['exposure_path']);prior.verify(x)
    if x['actual']!=expected['actual'] or x['batch_records']!=expected['batches'] or x['brightness_log']!=expected['brightness_log']:raise ValueError('Training exposure mismatch')
    if prior.file_sha256(r['weights'])!=r['weights_sha256']:raise ValueError('Changed weight')
    return r

def train(key):
    p,source,_,seed=inputs(key);expected=verify_check(key)
    prior.verify(prior.read(OUT/'authorization.json'))
    root=OUT/'training'/key
    if (root/'completion.json').exists():complete(key);return
    attempt=attempt_folder(root)
    try:
        import torch,yaml
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect import DetectionTrainer
        torch.set_num_threads(4);logs=[];sk=f'R-clean-{seed}'
        gate=old.BatchGate(expected,{r['image_path']:r['member_id'] for r in source['pool_rows']})
        class Trainer(DetectionTrainer):
            step_count=0
            def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
                if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
                if dataset_path!=source['listings'][sk] or batch_size!=6:raise ValueError('Loader drift')
                return make_loader(make_dataset(source,sk,logs),source,sk,self)
            def preprocess_batch(self,batch):return gate.apply(self,batch)
            def optimizer_step(self):
                if self.step_count>=450:raise ValueError('Extra optimizer step')
                result=super().optimizer_step();self.step_count+=1;return result
        data=attempt/'dataset.yaml';data.write_text(yaml.safe_dump(dict(path=str(root),train=source['listings'][sk],val=source['listings'][sk],names=source['names'])))
        model=YOLO(source['initialization']['path'])
        model.train(trainer=Trainer,data=str(data),project=str(attempt),name='run',**source['training_config'][sk])
        gate.finish(logs)
        if model.trainer.step_count!=450:raise ValueError('Wrong optimizer count')
        args=attempt/'run/args.yaml';curve=attempt/'run/results.csv'
        check_curve(yaml.safe_load(args.read_text()),list(csv.DictReader(curve.open())),seed,.0005)
        xp=attempt/'exposure.json';prior.frozen(xp,dict(actual=gate.actual,batch_records=gate.records,brightness_log=logs))
        weight=attempt/'run/weights/last.pt'
        paths=[OUT/'protocol.json',OUT/'authorization.json',OUT/'checks'/key/'completion.json',xp,weight,args,curve,Path(__file__).resolve()]
        prior.frozen(root/'completion.json',dict(status='complete',cell=key,optimizer_steps=450,checkpoint_selection='terminal_last_only',
            exposure_path=str(xp),weights=str(weight),weights_sha256=prior.file_sha256(weight),training_member_validation_role='training_fit_only',
            inputs={str(x):prior.file_sha256(x) for x in paths}))
        complete(key)
    except BaseException as ex:failure(attempt,ex);raise

def run():
    freeze()
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for key in KEYS:preflight(key)
        tests=subprocess.run([sys.executable,'-m','unittest','tests.test_scale_endpoint_control','tests.test_multiscale_training_entry','tests.test_frozen_multiscale_runtime'],capture_output=True,text=True)
        if tests.returncode:raise ValueError(tests.stdout+tests.stderr)
        auth=OUT/'authorization.json'
        if not auth.exists():prior.frozen(auth,dict(status='explicit_run_authorized_after_preflight',regressions=tests.stdout+tests.stderr,inputs={str(OUT/'protocol.json'):prior.file_sha256(OUT/'protocol.json'),**{str(OUT/'checks'/k/'completion.json'):prior.file_sha256(OUT/'checks'/k/'completion.json') for k in KEYS}}))
        else:prior.verify(prior.read(auth))
        for key in KEYS:
            if (OUT/'training'/key/'completion.json').exists():complete(key);continue
            folder=attempt_folder(OUT/'workers'/key);proc=None
            try:
                with (folder/'log.txt').open('x') as log:
                    proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.scale_endpoint_control','--worker',key],cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    print('TRAINING',key,proc.pid,flush=True);proc.wait(timeout=21600)
                    if proc.returncode:raise RuntimeError('Worker failed '+str(folder))
                complete(key);print('TRAINED',key,flush=True)
            except BaseException as ex:
                if proc:old.cleanup(proc)
                failure(folder,ex);raise
        from scripts.vision.evaluate_scale_endpoints import run as evaluate
        evaluate()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--worker',choices=KEYS);a=ap.parse_args()
    if a.worker:train(a.worker)
    elif a.run:run()
    else:freeze();print('FROZEN_ONLY_NO_TRAINING')

if __name__=='__main__':main()
