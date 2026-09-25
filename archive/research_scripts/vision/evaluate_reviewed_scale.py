"""Scale-control evaluation adapter; fixed protocol, no automatic reviews."""
import argparse
import fcntl
import signal
import subprocess
import sys
import traceback
from pathlib import Path
from scripts.vision.train_reviewed_scale_control import OUT,KEYS,complete,contract,ready,prior
from scripts.vision.freeze_reviewed_scale_control import freeze,REFERENCE
from scripts.vision import evaluate_order_fit_reviewed as base
from scripts.vision.train_closed_source_control import cleanup

def bind():
    base.DEST=OUT;base.TRAIN=REFERENCE;base.KEYS=KEYS;base.freeze=freeze;base.complete=complete
    return base

def evaluate(key):
    r=bind().evaluate(key);dest=OUT/'evaluation'/f'{key}-scale-binding.json'
    if dest.exists():prior.verify(prior.read(dest))
    else:
        deps=[Path(__file__).resolve(),Path(base.__file__).resolve(),OUT/'evaluation'/f'{key}.json',OUT/'design.json']
        prior.frozen(dest,dict(status='scale_evaluation_adapter_bound',inputs={str(x):prior.file_sha256(x) for x in deps}))
    return r

def finish():
    for key in KEYS:evaluate(key)
    r=bind().finish();dest=OUT/'evaluation/scale-completion.json'
    if dest.exists():prior.verify(prior.read(dest))
    else:
        deps=[OUT/'evaluation/summary.json',Path(__file__).resolve()]+[OUT/'evaluation'/f'{k}-scale-binding.json' for k in KEYS]
        prior.frozen(dest,dict(status=r['status'],review_complete=False,selected_candidate=None,inputs={str(x):prior.file_sha256(x) for x in deps}))
    return r

def run():
    contract();ready();root=OUT/'evaluation-runner';root.mkdir(exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def stop(signum,frame):raise KeyboardInterrupt(signum)
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        for k in KEYS:complete(k)
        for start in range(0,len(KEYS),2):
            jobs=[]
            try:
                for k in KEYS[start:start+2]:
                    if (OUT/'evaluation'/f'{k}.json').exists():evaluate(k);continue
                    folder=root/k;folder.mkdir(exist_ok=True);n=len(list(folder.glob('attempt-*')))+1
                    if n>3:raise ValueError('Inference attempt cap')
                    attempt=folder/f'attempt-{n:03}';attempt.mkdir();log=(attempt/'log.txt').open('x')
                    proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.evaluate_reviewed_scale','--evaluate','--cell',k],cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    jobs.append((proc,log,attempt));print('EVALUATION_STARTED',k,proc.pid,flush=True)
                for proc,_,_ in jobs:
                    proc.wait(timeout=1800)
                    if proc.returncode:raise RuntimeError('Evaluation worker failed')
            except BaseException:
                error=traceback.format_exc()
                for proc,_,attempt in jobs:
                    cleanup(proc);prior.frozen(attempt/'failure.json',dict(error=error,process_cleanup_complete=proc.poll() is not None))
                raise
            finally:
                for proc,log,_ in jobs:cleanup(proc);log.close()
        print(finish()['status'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--evaluate',action='store_true');ap.add_argument('--cell',choices=KEYS);a=ap.parse_args()
    if a.cell and not a.evaluate:ap.error('--evaluate required')
    if a.cell:evaluate(a.cell)
    elif a.evaluate:run()
    else:contract();ready();print('PREFLIGHT_ONLY_NO_INFERENCE')
