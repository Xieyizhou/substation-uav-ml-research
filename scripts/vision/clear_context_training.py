"""Explicit 3-seed fixed-budget data control and automatic numerical evaluation."""
import argparse
import fcntl
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision.freeze_clear_context_training import OUT,DATA,CONTROL,KEYS,SEEDS,freeze,checked
from scripts.vision import reviewed_negative_order_control as baseline
from scripts.vision import train_reactor_visibility_expansion as trainer
from scripts.vision import evaluate_reviewed_negative_order as evaluation
from scripts.vision.verify_experiment_baseline import verify as integrity,ROOT

MODULE='scripts.vision.clear_context_training'
OBSERVED={}
TESTS=('tests.test_clear_context_increment','tests.test_clear_context_training','tests.test_canonical_gates',
       'tests.test_canonical_recovery','tests.test_canonical_recovery_increment','tests.test_reviewed_order_evaluation','tests.test_reviewed_order_audit')


def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def loader(p,key,owner):
    dataset,raw=baseline.loader(p,key,owner)
    path=OUT/'actual-preflight'/f'{key}.json'
    expected=checked(path)['tensor_records'] if path.exists() else None
    records=OBSERVED.setdefault(key,[])
    class Wrapped:
        def __len__(self):return len(raw)
        def __getattr__(self,name):return getattr(raw,name)
        def __iter__(self):
            for j,b in enumerate(raw):
                pos=owner.epoch*60+j*6
                record=dict(position=pos,images=list(b['im_file']),tensors={k:tensor_hash(b[k]) for k in ('img','cls','bboxes','batch_idx')})
                if expected is not None and record!=expected[pos//6]:raise ValueError('Actual loader tensors differ from preflight')
                records.append(record);yield b
    return dataset,Wrapped()


def preflight():
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4);p=freeze();root=OUT/'actual-preflight';root.mkdir(exist_ok=True)
    inverse={r['image_path']:r['member_id'] for r in p['pool_rows']}
    for key in KEYS:
        path=root/f'{key}.json'
        if path.exists():r=checked(path)
        else:
            owner=SimpleNamespace(epoch=0);actual=[];OBSERVED[key]=[]
            with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('Optimizer forbidden')),patch.object(torch.Tensor,'backward',side_effect=AssertionError('Backward forbidden')),patch.object(YOLO,'train',side_effect=AssertionError('Training forbidden')),patch.object(YOLO,'val',side_effect=AssertionError('Validation forbidden')):
                dataset,batches=loader(p,key,owner)
                for epoch in range(48):
                    owner.epoch=epoch
                    for b in batches:
                        if tuple(b['img'].shape)!=(6,3,640,640):raise ValueError('Unexpected input shape')
                        actual.extend(inverse[x] for x in b['im_file'])
            if actual!=p['schedules'][key] or len(OBSERVED[key])!=480:raise ValueError('Actual sequence mismatch')
            r=write_record(path,dict(status='actual_complete_loader_verified_no_training',actual=actual,tensor_records=OBSERVED[key],checked_draws=2880,
                optimizer_created=False,backward_executed=False,validation_run=False,training_admitted=False,promotable=False,
                inputs={str((OUT/'protocol.json').resolve()):file_sha256(OUT/'protocol.json')}))
        if r['actual']!=p['schedules'][key] or len(r['tensor_records'])!=480:raise ValueError('Invalid cached preflight')
        print('PREFLIGHT_COMPLETE',key,2880,flush=True)
    path=OUT/'entry-ready.json'
    if not path.exists():write_record(path,dict(status='ready_for_training_not_started',cells=list(KEYS),actual_draws_verified=8640,
        optimizer_created=False,backward_executed=False,training_admitted=False,promotable=False,
        inputs={str(x.resolve()):file_sha256(x) for x in [OUT/'protocol.json']+[root/f'{k}.json' for k in KEYS]}))
    return p,checked(path)


def launch_gate(create=False):
    path=OUT/'launch-readiness.json'
    if not path.exists():
        if not create:raise ValueError('Launch gate missing')
        preflight();contract=ROOT/'config/perception/visual_experiment_baseline_v1.json';result=integrity(contract)
        if not result['integrity_passed'] or result['pinned_files_verified']!=40:raise ValueError('Fixed-40 integrity failed')
        command=[sys.executable,'-m','unittest',*TESTS,'-q'];tests=subprocess.run(command,text=True,capture_output=True,timeout=180)
        if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
        _,_,deps=evaluation.ev._load_inputs()
        for x in [OUT/'protocol.json',OUT/'entry-ready.json',Path(__file__),contract]+[ROOT/('tests/'+t.split('.')[-1]+'.py') for t in TESTS]:deps[str(x.resolve())]=file_sha256(x)
        write_record(path,dict(status='verified_for_training_and_automatic_evaluation',test_command=command,test_returncode=0,test_output=tests.stdout+tests.stderr,
            integrity=result,training_admitted=False,promotable=False,inputs=deps))
    return checked(path)


def worker(key):
    launch_gate();p,ready=preflight();OBSERVED[key]=[]
    root=OUT/'training'/key;cp=root/'completion.json';tp=root/'tensor-verification.json'
    if cp.exists():
        c=checked(cp);e=checked(c['exposure_path']);t=checked(tp)
        if c['optimizer_steps']!=480 or e['actual']!=p['schedules'][key] or t['records']!=checked(OUT/'actual-preflight'/f'{key}.json')['tensor_records']:raise ValueError('Completed unit invalid')
        return c
    trainer.OUT=OUT.resolve();trainer.KEYS=KEYS;trainer.STEPS=480;trainer._loader=loader;trainer._ready=lambda:(p,ready)
    c=trainer._worker(key)
    expected=checked(OUT/'actual-preflight'/f'{key}.json')['tensor_records']
    if OBSERVED[key]!=expected:raise ValueError('Training tensor ledger incomplete')
    write_record(tp,dict(status='actual_tensors_match_preflight',records=OBSERVED[key],training_admitted=False,promotable=False,
        inputs={str(x.resolve()):file_sha256(x) for x in [cp,OUT/'actual-preflight'/f'{key}.json']}))
    return c


def parallel(flag,folder):
    folder.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(KEYS),2):
        jobs=[]
        try:
            for key in KEYS[start:start+2]:
                n=len(list(folder.glob(f'{key}-attempt-*.log')))+1
                if n>3:raise ValueError('Technical attempt cap')
                log=(folder/f'{key}-attempt-{n:03}.log').open('x')
                proc=subprocess.Popen([sys.executable,'-u','-m',MODULE,flag,key],stdout=log,stderr=subprocess.STDOUT)
                jobs.append((proc,log,key));print('STARTED',flag,key,proc.pid,flush=True)
            for proc,_,key in jobs:
                proc.wait(timeout=14400)
                if proc.returncode:raise RuntimeError('Unit failed '+key)
        finally:
            for proc,log,_ in jobs:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                log.close()


def eval_worker(key):
    launch_gate();evaluation.TRAIN=OUT;evaluation.OUT=OUT/'evaluation-v1';evaluation.KEYS=KEYS
    return evaluation.worker(key)


def summarize():
    records=[];deps={};comparisons=[];queue=[]
    for seed,key in zip(SEEDS,KEYS):
        cp=OUT/'evaluation-v1/units'/key/'completion.json';c=checked(cp);r=checked(c['result']);records.append(r)
        ref=checked(CONTROL/'evaluation-v1/units'/f'reviewed-interleaved-480-{seed}'/'completion.json');old=checked(ref['result'])
        if len(r['rows'])!=48 or len(old['rows'])!=48:raise ValueError('Incomplete paired evaluation')
        for x,y in zip(old['rows'],r['rows']):comparisons.append(dict(seed=seed,pair_id=x['pair_id'],variant=x['variant'],instances=evaluation.compare_truth(x,y)))
        for row in r['negative_rows']:
            for i,pred in enumerate(row['predictions']):queue.append(dict(seed=seed,view_id=row['view_id'],image_sha256=row['image_sha256'],prediction_index=i,prediction=pred,review_status='pending'))
        for path in (cp,Path(c['result']),Path(ref['result'])):deps[str(path.resolve())]=file_sha256(path)
    return write_record(OUT/'evaluation-v1/summary.json',dict(status='numerical_complete_visual_review_and_retention_gates_pending',group=evaluation.aggregate(records),
        instance_comparisons=comparisons,negative_fp_review_queue=queue,matching_conflicts={k:r['matching_conflicts'] for k,r in zip(KEYS,records)},selected_candidate=None,
        training_admitted=False,promotable=False,inputs=deps))


def train():
    launch_gate(create=True);root=OUT/'runner';root.mkdir(exist_ok=True)
    with (root/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);parallel('--worker',root)
        paths=[OUT/'training'/k/name for k in KEYS for name in ('completion.json','tensor-verification.json')]
        for path in paths:checked(path)
        cp=root/'completion.json'
        if not cp.exists():write_record(cp,dict(status='three_units_complete_evaluation_queued',training_admitted=False,promotable=False,inputs={str(x.resolve()):file_sha256(x) for x in paths}))
        parallel('--eval-worker',OUT/'evaluation-v1')
        if not (OUT/'evaluation-v1/summary.json').exists():summarize()
        print('TRAINING_AND_NUMERICAL_EVALUATION_COMPLETE_REVIEW_PENDING',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--freeze',action='store_true');ap.add_argument('--worker',choices=KEYS);ap.add_argument('--eval-worker',choices=KEYS);a=ap.parse_args()
    if a.worker:worker(a.worker)
    elif a.eval_worker:eval_worker(a.eval_worker)
    elif a.train:train()
    elif a.freeze:print(freeze()['status'])
    else:preflight();print('PREFLIGHT_ONLY_NO_TRAINING')
