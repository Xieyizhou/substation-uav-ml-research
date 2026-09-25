"""Explicit isolated three-seed control, using the frozen shared training adapter."""
import argparse,fcntl,os,signal,subprocess,sys,traceback
from pathlib import Path
from scripts.vision.lineage_capped_control import OUT,ROOT,KEYS,ready,read,verify,frozen,file_sha256
from scripts.vision import train_whole_image_hold as adapter

def configured():
    # Process-local dependency injection; historical source and entry behavior stay unchanged.
    adapter.OUT=OUT;adapter.KEYS=KEYS;adapter.ready=ready

def completed(key,p):
    adapter.complete(OUT/'training'/key/'completion.json',p,key)
    v=read(OUT/'training'/key/'adapter-binding.json');verify(v)
    if v['protocol_identity']!=p['identity']:raise ValueError('Adapter binding changed')

def worker(key):
    configured();p=ready();(OUT/'export').mkdir(exist_ok=True)
    adapter.worker(key)
    adapter.complete(OUT/'training'/key/'completion.json',p,key)
    paths=[Path(__file__),Path(adapter.__file__),ROOT/'scripts/vision/lineage_capped_control.py',ROOT/'scripts/vision/order_retention_runtime.py',OUT/'training'/key/'completion.json']
    dest=OUT/'training'/key/'adapter-binding.json'
    if dest.exists():verify(read(dest))
    else:frozen(dest,dict(status='adapter_and_weights_bound',protocol_identity=p['identity'],inputs={str(x):file_sha256(x) for x in paths}))

def run():
    p=ready()
    with (OUT/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for key in KEYS:
            if (OUT/'training'/key/'adapter-binding.json').exists():completed(key,p);continue
            root=OUT/'workers'/key;root.mkdir(parents=True,exist_ok=True);n=len(list(root.glob('attempt-*')))+1
            if n>3:raise ValueError('Worker attempt budget exhausted')
            attempt=root/f'attempt-{n:03}';attempt.mkdir();proc=None
            try:
                with (attempt/'log.txt').open('x') as log:
                    proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.train_lineage_capped_control','--train','--worker',key],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    frozen(attempt/'launch.json',dict(pid=proc.pid,cell=key,inputs={str(OUT/'ready.json'):file_sha256(OUT/'ready.json'),str(Path(__file__)):file_sha256(Path(__file__))}))
                    proc.wait(timeout=21600)
                    if proc.returncode:raise RuntimeError('Worker failed '+key)
                completed(key,p)
                frozen(attempt/'completion.json',dict(status='complete',process_cleanup_confirmed=True,inputs={str(attempt/'log.txt'):file_sha256(attempt/'log.txt')}))
            except BaseException:
                if proc is not None and proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
                frozen(attempt/'failure.json',dict(error=traceback.format_exc(),process_cleanup_confirmed=proc is None or proc.poll() is not None));raise
            print('TRAINED',key,flush=True)
        frozen(OUT/'training/completion.json',dict(status='three_trained_evaluation_pending',inputs={str(OUT/'training'/k/'adapter-binding.json'):file_sha256(OUT/'training'/k/'adapter-binding.json') for k in KEYS}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);a=ap.parse_args()
    if a.worker and not a.train:ap.error('worker requires explicit train')
    if a.train:
        if a.worker:worker(a.worker)
        else:run()
    else:print('PREFLIGHT_ONLY_NO_TRAINING')
