"""Explicit bounded dual-four-thread training coordinator; no scheduled wakeups."""
import argparse,fcntl,signal,subprocess,sys,time,traceback
from pathlib import Path
from scripts.vision.train_tail_interleaving import OUT,prior,contract
from scripts.vision.train_closed_source_control import cleanup

def run():
    contract()
    ready=OUT/'direction-completion.json';prior.verify(prior.read(ready))
    root=OUT/'training-runner';root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dest=root/'completion.json'
        if dest.exists():prior.verify(prior.read(dest));return
        def stop(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        started=time.monotonic();deps=[ready,Path(__file__).resolve()]
        for group in ((7,17),(27,)):
            jobs=[]
            try:
                for seed in group:
                    cp=OUT/'training'/f'I1000-{seed}'/'completion.json';deps.append(cp)
                    if cp.exists():prior.verify(prior.read(cp));continue
                    folder=root/f'I1000-{seed}';folder.mkdir(exist_ok=True)
                    n=len(list(folder.glob('attempt-*')))+1
                    if n>3:raise ValueError('Launch attempt cap')
                    attempt=folder/f'attempt-{n:03}';attempt.mkdir();log=(attempt/'log.txt').open('x')
                    proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.train_tail_interleaving','--train','--seed',str(seed)],cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    jobs.append((proc,log,attempt,cp));print('TRAINING_STARTED',seed,proc.pid,'THREADS=4',flush=True)
                for proc,_,_,cp in jobs:
                    proc.wait(timeout=7200)
                    if proc.returncode:raise RuntimeError(f'Training exit {proc.returncode}: {cp}')
                    c=prior.read(cp);prior.verify(c)
                    if c['optimizer_steps']!=1000 or prior.file_sha256(c['weights'])!=c['weights_sha256']:raise ValueError('Endpoint invalid')
                    print('TRAINING_VERIFIED',c['cell'],flush=True)
            except BaseException:
                for proc,_,attempt,_ in jobs:
                    cleanup(proc)
                    prior.frozen(attempt/'failure.json',dict(error=traceback.format_exc(),process_cleanup_complete=proc.poll() is not None))
                raise
            finally:
                for proc,log,_,_ in jobs:cleanup(proc);log.close()
        prior.frozen(dest,dict(status='three_seed_training_complete_not_evaluated',threads_per_worker=4,max_parallel=2,wall_seconds=time.monotonic()-started,inputs={str(p):prior.file_sha256(p) for p in deps}))
        print('ALL_TRAINING_COMPLETE_EVALUATION_NOT_STARTED',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');a=ap.parse_args()
    if a.train:run()
    else:contract();print('PREFLIGHT_ONLY')
