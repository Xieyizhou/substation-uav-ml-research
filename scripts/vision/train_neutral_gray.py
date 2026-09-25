"""Explicit gray-material training; default contract only, CPU 4 threads per worker."""
import argparse,fcntl,signal,subprocess,sys,time,traceback
from pathlib import Path
from scripts.vision.neutral_gray_control import OUT,REF,KEYS,freeze,prior
from scripts.vision import train_tail_interleaving as engine
from scripts.vision.train_closed_source_control import cleanup

def contract():
    p=freeze();fp=OUT/'loader-feasibility.json';f=prior.read(fp);prior.verify(f)
    if prior.file_sha256(p['initialization']['path'])!=p['initialization']['sha256']:raise ValueError('Initialization drift')
    if f['optimizer_created'] or f['backward_executed'] or f['validation_run']:raise ValueError('Invalid no-training preflight')
    if [u['cell'] for u in f['units']]!=list(KEYS):raise ValueError('Preflight units drift')
    for u in f['units']:
        engine.check(p,u['cell'],u['actual'],u['brightness_log'])
        if len(u['batch_records'])!=1000:raise ValueError('Missing batches')
    import platform,torch,ultralytics
    if p['environment']!=dict(python=platform.python_version(),torch=torch.__version__,ultralytics=ultralytics.__version__):raise ValueError('Environment drift')
    b=engine.baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline drift')
    deps=[OUT/'design.json',OUT/'quality-review.json',fp,Path(p['initialization']['path']),Path(__file__).resolve(),Path(engine.__file__).resolve(),Path(__file__).with_name('closed_budget_engine_v2.py')]
    return p,f,deps,b

def verify_ready():
    r=prior.read(OUT/'entry-ready.json');prior.verify(r)
    if r['status']!='ready_for_training_not_started' or r['cells']!=list(KEYS) or r['actual_draws_verified']!=18000:raise ValueError('Incomplete entry receipt')
    if not r['relevant_regressions_passed'] or not r['exact_image_and_label_tensors_frozen']:raise ValueError('Missing readiness gate')
    if r['optimizer_created'] or r['backward_executed'] or r['training_admitted'] or r['promotable']:raise ValueError('Invalid readiness flags')
    return r

def worker(key):
    verify_ready()
    engine.OUT=OUT;engine.contract=contract
    return engine.run(key)

def run():
    contract();verify_ready();root=OUT/'training-runner';root.mkdir(exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def stop(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        started=time.monotonic()
        for group in (KEYS[:2],KEYS[2:]):
            jobs=[]
            try:
                for key in group:
                    cp=OUT/'training'/key/'completion.json'
                    if cp.exists():prior.verify(prior.read(cp));continue
                    folder=root/key;folder.mkdir(exist_ok=True);n=len(list(folder.glob('attempt-*')))+1
                    if n>3:raise ValueError('Launch cap')
                    attempt=folder/f'attempt-{n:03}';attempt.mkdir();log=(attempt/'log.txt').open('x')
                    proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.train_neutral_gray','--worker',key,'--train'],cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    jobs.append((proc,log,attempt,key));print('TRAINING_STARTED',key,proc.pid,'THREADS=4',flush=True)
                for proc,_,_,key in jobs:
                    proc.wait(timeout=7200)
                    if proc.returncode:raise RuntimeError('Worker failed '+key)
                    c=prior.read(OUT/'training'/key/'completion.json');prior.verify(c)
                    if c['optimizer_steps']!=1000 or prior.file_sha256(c['weights'])!=c['weights_sha256']:raise ValueError('Invalid endpoint')
                    print('TRAINING_VERIFIED',key,flush=True)
            except BaseException:
                for proc,_,attempt,key in jobs:
                    cleanup(proc);prior.frozen(attempt/'failure.json',dict(cell=key,error=traceback.format_exc(),process_cleanup_complete=proc.poll() is not None))
                raise
            finally:
                for proc,log,_,_ in jobs:cleanup(proc);log.close()
        deps=[OUT/'training'/k/'completion.json' for k in KEYS]+[Path(__file__).resolve()]
        prior.frozen(root/'completion.json',dict(status='three_seed_training_complete_evaluation_pending',wall_seconds=time.monotonic()-started,inputs={str(d):prior.file_sha256(d) for d in deps}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);a=ap.parse_args()
    if a.worker and not a.train:ap.error('Explicit --train required')
    if a.worker:worker(a.worker)
    elif a.train:run()
    else:contract();print('READY_NO_TRAINING_STARTED')
