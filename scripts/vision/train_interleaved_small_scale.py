"""Explicit order-only training, reusing verified engine without history edits."""
import argparse
import fcntl
import platform
import signal
import subprocess
import sys
import traceback
from pathlib import Path
from scripts.vision.interleaved_small_scale_control import OUT,KEYS,freeze,prior
from scripts.vision.closed_budget_runtime import check
from scripts.vision.exposure_order_retention import baseline_verify
from scripts.vision.train_closed_source_control import cleanup


def contract():
    p=freeze(); fp=OUT/'loader-feasibility.json'; f=prior.read(fp); prior.verify(f)
    if any(f[k] for k in ('optimizer_created','backward_executed','validation_run')):
        raise ValueError('Preflight executed forbidden operation')
    if [u['cell'] for u in f['units']]!=list(KEYS): raise ValueError('Missing preflight units')
    for u in f['units']:
        check(p,u['cell'],u['actual'],u['brightness_log'])
        if len(u['batch_records'])!=1100: raise ValueError('Incomplete actual loader')
    import torch,ultralytics
    if p['environment']!=dict(python=platform.python_version(),torch=torch.__version__,ultralytics=ultralytics.__version__):
        raise ValueError('Environment drift')
    init=Path(p['initialization']['path'])
    if prior.file_sha256(init)!=p['initialization']['sha256']: raise ValueError('Initialization drift')
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40: raise ValueError('Baseline drift')
    deps=[OUT/'design.json',fp,init,Path(__file__).resolve(),Path(__file__).with_name('closed_budget_engine_v2.py'),
          Path(__file__).with_name('train_small_scale.py')]
    return p,f,deps,b


def ready():
    path=OUT/'entry-ready.json'; r=prior.read(path); prior.verify(r)
    if r['status']!='ready_for_training_not_started' or r['cells']!=list(KEYS) or r['actual_draws_verified']!=39600:
        raise ValueError('Invalid readiness')
    if not r['relevant_regressions_passed'] or not r['exact_image_and_label_tensors_frozen']:
        raise ValueError('Missing regression/tensor gate')
    if any(r[k] for k in ('optimizer_created','backward_executed','training_admitted','promotable')):
        raise ValueError('Invalid readiness flags')
    return path


def complete(key):
    if key not in KEYS: raise ValueError('Unknown cell')
    p=freeze(); cp=OUT/'training'/key/'completion.json'; c=prior.read(cp); prior.verify(c)
    x=prior.read(c['exposure_path']); prior.verify(x)
    check(p,key,x['actual'],x['brightness_log'])
    if c['optimizer_steps']!=1100 or x['optimizer_steps']!=1100: raise ValueError('Wrong endpoint')
    if prior.file_sha256(c['weights'])!=c['weights_sha256']: raise ValueError('Weight drift')
    return c


def worker(key):
    # Only worker implementation is reused; its contract and destinations are local
    # process bindings, not changes to the historical module or artifacts.
    from scripts.vision import train_small_scale as engine
    engine.OUT=OUT; engine.KEYS=KEYS; engine.contract=contract; engine.ready=ready
    def stop(signum,frame): raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    if (OUT/'training'/key/'completion.json').exists(): return complete(key)
    return engine.worker(key)


def run():
    contract(); ready(); root=OUT/'training-runner'; root.mkdir(exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def stop(signum,frame): raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
        for start in range(0,len(KEYS),2):
            jobs=[]
            try:
                for key in KEYS[start:start+2]:
                    if (OUT/'training'/key/'completion.json').exists(): complete(key); continue
                    folder=root/key; folder.mkdir(exist_ok=True)
                    n=len(list(folder.glob('attempt-*')))+1
                    if n>3: raise ValueError('Launch attempt cap')
                    attempt=folder/f'attempt-{n:03}'; attempt.mkdir()
                    log=(attempt/'log.txt').open('x')
                    proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.train_interleaved_small_scale','--train','--worker',key],
                        cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    jobs.append((proc,log,key,attempt))
                    print('TRAINING_STARTED',key,proc.pid,'THREADS=4',flush=True)
                for proc,_,key,_ in jobs:
                    proc.wait(timeout=7200)
                    if proc.returncode: raise RuntimeError('Worker failed '+key)
                    complete(key); print('TRAINING_VERIFIED',key,flush=True)
            except BaseException:
                error=traceback.format_exc()
                for proc,_,key,attempt in jobs:
                    cleanup(proc)
                    prior.frozen(attempt/'failure.json',dict(cell=key,error=error,process_cleanup_complete=proc.poll() is not None))
                raise
            finally:
                for proc,log,_,_ in jobs: cleanup(proc); log.close()
        deps=[OUT/'training'/k/'completion.json' for k in KEYS]+[Path(__file__).resolve()]
        prior.frozen(root/'completion.json',dict(status='six_units_complete_evaluation_pending',inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--train',action='store_true'); ap.add_argument('--worker',choices=KEYS)
    ap.add_argument('--then-evaluate',action='store_true'); a=ap.parse_args()
    if a.worker and not a.train: ap.error('Explicit --train required')
    if a.then_evaluate and (not a.train or a.worker): ap.error('--then-evaluate requires full --train')
    if a.worker: worker(a.worker)
    elif a.train:
        run()
        if a.then_evaluate:
            from scripts.vision.evaluate_interleaved_small_scale import run as evaluate_all
            evaluate_all()
    else: contract(); print('PREFLIGHT_ONLY_NO_TRAINING')
