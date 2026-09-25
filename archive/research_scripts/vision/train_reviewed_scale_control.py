"""Explicit scale-control training after complete actual preflight."""
import argparse
import fcntl
import signal
import subprocess
import sys
import traceback
from pathlib import Path
from scripts.vision.freeze_reviewed_scale_control import OUT,KEYS,freeze,prior,contract as source_contract
from scripts.vision import reviewed_scale_runtime as runtime
from scripts.vision.train_closed_source_control import cleanup

TESTS=('test_reviewed_scale_runtime','test_reviewed_scale_control','test_reviewed_scale_probe','test_reviewed_endpoint_review','test_exposure_diagnosis','test_order_diagnosis')
def contract():
    p=freeze();_,_,_,baseline=source_contract();units=[];deps=[OUT/'design.json',Path(__file__).resolve(),Path(runtime.__file__).resolve()]
    for k in KEYS:
        path=OUT/'actual-preflight'/f'{k}.json';r=prior.read(path);prior.verify(r)
        if r['key']!=k or not r['full_geometry_checked'] or r['checked_draws']!=6600 or r['checked_batches']!=1100 or len(r['batch_records'])!=1100:raise ValueError('Incomplete real scale preflight')
        if any(r[x] for x in ('optimizer_created','backward_executed','validation_run')):raise ValueError('Training occurred during preflight')
        runtime.check(p,k,r['actual'],r['brightness_log']);units.append(r);deps.append(path)
    deps += [Path(__file__).with_name(x+'.py') for x in ('closed_budget_engine_v2','train_small_scale','closed_budget_runtime','order_retention_runtime','brightness_transfer_runtime')]
    return p,dict(units=units),deps,baseline

def ready():
    path=OUT/'entry-ready.json';r=prior.read(path);prior.verify(r)
    if r['status']!='ready_for_training_not_started' or r['cells']!=list(KEYS) or r['actual_draws_verified']!=39600:raise ValueError('Invalid readiness')
    if not r['relevant_regressions_passed'] or not r['exact_image_and_label_tensors_frozen']:raise ValueError('Missing regression/tensor gate')
    if any(r[x] for x in ('optimizer_created','backward_executed','training_admitted','promotable')):raise ValueError('Invalid readiness flags')
    return path

def prepare():
    _,_,deps,b=contract()
    if (OUT/'entry-ready.json').exists():return ready()
    r=subprocess.run([sys.executable,'-m','unittest',*('tests.'+x for x in TESTS)],cwd=prior.ROOT,capture_output=True,text=True)
    if r.returncode:raise RuntimeError(r.stdout+r.stderr)
    deps += [prior.ROOT/'tests'/f'{x}.py' for x in TESTS]
    prior.frozen(OUT/'entry-ready.json',dict(status='ready_for_training_not_started',cells=list(KEYS),actual_draws_verified=39600,
        relevant_regressions_passed=True,regression_output=r.stdout+r.stderr,exact_image_and_label_tensors_frozen=True,baseline=b,
        optimizer_created=False,backward_executed=False,inputs={str(x):prior.file_sha256(x) for x in deps}))
    return ready()

def complete(k):
    p=freeze();path=OUT/'training'/k/'completion.json';r=prior.read(path);prior.verify(r)
    x=prior.read(r['exposure_path']);prior.verify(x);runtime.check(p,k,x['actual'],x['brightness_log'])
    if r['optimizer_steps']!=1100 or x['optimizer_steps']!=1100 or prior.file_sha256(r['weights'])!=r['weights_sha256']:raise ValueError('Invalid completion')
    return r

def stop(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')

def worker(k):
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if (OUT/'training'/k/'completion.json').exists():return complete(k)
    from scripts.vision import train_small_scale as engine
    from scripts.vision import closed_budget_engine_v2 as core
    # Per-process adapters; the historical modules and artifacts are not edited.
    core.make_dataset=runtime.make_dataset;core.make_loader=runtime.make_loader;core.check=runtime.check
    engine.OUT=OUT;engine.KEYS=KEYS;engine.contract=contract;engine.ready=ready
    return engine.worker(k)

def run():
    contract();ready();root=OUT/'training-runner';root.mkdir(exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        for start in range(0,len(KEYS),2):
            jobs=[]
            try:
                for k in KEYS[start:start+2]:
                    if (OUT/'training'/k/'completion.json').exists():complete(k);continue
                    folder=root/k;folder.mkdir(exist_ok=True);n=len(list(folder.glob('attempt-*')))+1
                    if n>3:raise ValueError('Technical attempt cap')
                    attempt=folder/f'attempt-{n:03}';attempt.mkdir();log=(attempt/'log.txt').open('x')
                    proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.train_reviewed_scale_control','--train','--worker',k],cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    jobs.append((proc,log,attempt));print('TRAINING_STARTED',k,proc.pid,'THREADS=4',flush=True)
                for proc,_,_ in jobs:
                    proc.wait(timeout=14400)
                    if proc.returncode:raise RuntimeError('Worker failed')
            except BaseException:
                error=traceback.format_exc()
                for proc,_,attempt in jobs:
                    cleanup(proc);prior.frozen(attempt/'failure.json',dict(error=error,process_cleanup_complete=proc.poll() is not None))
                raise
            finally:
                for proc,log,_ in jobs:cleanup(proc);log.close()
        for k in KEYS:complete(k)
        deps=[OUT/'training'/k/'completion.json' for k in KEYS]
        prior.frozen(root/'completion.json',dict(status='six_units_complete_evaluation_pending',inputs={str(x):prior.file_sha256(x) for x in deps}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);a=ap.parse_args()
    if a.worker and not a.train:ap.error('--train required')
    if a.prepare:prepare();print('READY',flush=True)
    if a.worker:worker(a.worker)
    elif a.train:run()
    elif not a.prepare:contract();print('PREFLIGHT_ONLY_NO_TRAINING')
