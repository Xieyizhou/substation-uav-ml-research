"""Fixed development inference for six reviewed-order endpoints; never trains."""
import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
from pathlib import Path
import subprocess
import sys
import traceback
from unittest.mock import patch
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import read_record, write_record
from scripts.vision import evaluate_reactor_visibility_expansion as ev
from scripts.vision.reviewed_negative_order_control import OUT as TRAIN, KEYS, verify
from scripts.vision.finalize_exposure_diagnosis import aggregate

OUT=TRAIN/'evaluation-v1'

def checked(path):
    r=read_record(path);verify(r);return r

def inputs():
    p=checked(TRAIN/'protocol.json');checked(TRAIN/'runner/completion.json')
    paired,negatives,deps=ev._load_inputs()
    for path in (Path(__file__),TRAIN/'protocol.json',Path(ev.score.__code__.co_filename),Path(aggregate.__code__.co_filename)):
        deps[str(path.resolve())]=file_sha256(path)
    for key in KEYS:
        c=checked(TRAIN/'training'/key/'completion.json');e=checked(c['exposure_path'])
        if c['optimizer_steps']!=480 or e['actual']!=p['schedules'][key] or Path(c['weights']).name!='last.pt':raise ValueError('Endpoint mismatch')
        if file_sha256(c['weights'])!=c['weights_sha256']:raise ValueError('Weight changed')
        if c['training_admitted'] or c['promotable']:raise ValueError('Forbidden admission')
    return paired,negatives,deps

def summary_with_undefined_precision(rows):
    from scripts.vision.exposure_metrics import summary
    result=summary(rows)
    for name in result['per_class']:
        if not any(p['class_name']==name for row in rows for p in row['predictions']):result['per_class'][name]['matched_precision']=None
    return result

def worker(key):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    paired,negatives,deps=inputs();root=OUT/'units'/key;root.mkdir(parents=True,exist_ok=True)
    with (root/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        done=root/'completion.json'
        if done.exists():
            c=checked(done);r=checked(c['result']);return r
        c=checked(TRAIN/'training'/key/'completion.json')
        for n in range(len(list(root.glob('attempt-*')))+1,4):
            folder=root/f'attempt-{n:03}';folder.mkdir()
            try:
                ev.TRAIN=TRAIN;ev.OUT=folder;ev.summary=summary_with_undefined_precision
                with ExitStack() as stack:
                    for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(YOLO,'train'),(YOLO,'val')):
                        stack.enter_context(patch.object(obj,name,side_effect=AssertionError('Training forbidden in evaluation')))
                    r=ev._evaluate(key,Path(c['weights']),paired,negatives,deps)
                checked(folder/f'{key}.json')
                if len(r['rows'])!=48 or len(r['negative_rows'])!=48:raise ValueError('Incomplete evaluation')
                write_record(done,dict(status='complete',result=str((folder/f'{key}.json').resolve()),training_admitted=False,promotable=False,
                    inputs={str((folder/f'{key}.json').resolve()):file_sha256(folder/f'{key}.json')}))
                print('EVALUATION_COMPLETE',key,r['negative_summary'],flush=True);return r
            except BaseException as exc:
                write_record(folder/'failure.json',dict(status='failed',error=traceback.format_exc(),training_admitted=False,promotable=False))
                if isinstance(exc,(ValueError,AssertionError,KeyboardInterrupt,SystemExit)) or n==3:raise
        raise RuntimeError('Attempt cap exhausted')

def compare_truth(a,b):
    if (a['pair_id'],a['view_id'],a['variant'],a['image_sha256'],a['truth'])!=(b['pair_id'],b['view_id'],b['variant'],b['image_sha256'],b['truth']):
        raise ValueError('Paired truth or input differs')
    ma={m['truth_index'] for m in a['matches']};mb={m['truth_index'] for m in b['matches']}
    return [dict(truth=t,state='persistent_hit' if i in ma&mb else 'gain' if i in mb else 'loss' if i in ma else 'persistent_miss') for i,t in enumerate(a['truth'])]

def summarize():
    records={};deps={str(Path(__file__).resolve()):file_sha256(Path(__file__))}
    for key in KEYS:
        cp=OUT/'units'/key/'completion.json';c=checked(cp);r=checked(c['result']);records[key]=r
        deps[str(cp.resolve())]=file_sha256(cp);deps[c['result']]=file_sha256(c['result'])
    groups={arm:aggregate([records[f'{arm}-480-{s}'] for s in (7,17,27)]) for arm in ('reviewed-tail','reviewed-interleaved')}
    comparisons=[];errors=[]
    for seed in (7,17,27):
        a,b=(records[f'{arm}-480-{seed}'] for arm in ('reviewed-tail','reviewed-interleaved'))
        for x,y in zip(a['rows'],b['rows']):
            comparisons.append(dict(seed=seed,pair_id=x['pair_id'],variant=x['variant'],instances=compare_truth(x,y)))
    for key,r in records.items():
        for row in r['negative_rows']:
            for index,pred in enumerate(row['predictions']):errors.append(dict(cell=key,view_id=row['view_id'],variant=row['variant'],image_sha256=row['image_sha256'],prediction_index=index,prediction=pred,review_status='pending'))
    result=dict(status='numerical_complete_visual_review_and_retention_gates_pending',groups=groups,instance_comparisons=comparisons,negative_fp_review_queue=errors,
        matching_conflicts={k:r['matching_conflicts'] for k,r in records.items()},selected_candidate=None,
        limits=['Viewed development set: 48 paired images in 12 pose groups and 48 no-target images. Seeds do not add independent data.',
                'Only batch order differs between the two new arms. Historical data composition differs.',
                'Low-confidence diagnostics are bounded by NMS/max_det. Undefined precision retained as null.',
                'No semantic review decisions or candidate approval are generated by this summary.'],
        training_admitted=False,promotable=False,inputs=deps)
    dest=OUT/'summary.json'
    if dest.exists():return checked(dest)
    return write_record(dest,result)

def run():
    OUT.mkdir(exist_ok=True);inputs()
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for start in range(0,len(KEYS),2):
            jobs=[]
            try:
                for key in KEYS[start:start+2]:
                    log=(OUT/f'{key}.log').open('a')
                    proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.evaluate_reviewed_negative_order','--worker',key],stdout=log,stderr=subprocess.STDOUT)
                    jobs.append((proc,log,key));print('EVALUATION_STARTED',key,proc.pid,flush=True)
                for proc,_,key in jobs:
                    proc.wait(timeout=3600)
                    if proc.returncode:raise RuntimeError('Evaluation failed '+key)
            finally:
                for proc,log,_ in jobs:
                    if proc.poll() is None:
                        proc.terminate()
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    log.close()
        print(summarize()['status'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluate',action='store_true');p.add_argument('--worker',choices=KEYS);p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.worker:worker(a.worker)
    elif a.evaluate:run()
    elif a.summarize:print(summarize()['status'])
    else:inputs();print('PREFLIGHT_ONLY_NO_INFERENCE')
