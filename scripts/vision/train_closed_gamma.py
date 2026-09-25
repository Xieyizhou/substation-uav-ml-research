"""Explicit complete-budget training; consumes signed preflight and CPU policy."""
import argparse,fcntl,subprocess,sys,time,traceback,math,csv
from pathlib import Path
from scripts.vision.closed_gamma_design import OUT,KEYS,freeze,prior
from scripts.vision.preflight_closed_gamma import receipt
from scripts.vision.closed_gamma_engine import execute
from scripts.vision.closed_gamma_runtime import check
from scripts.vision.benchmark_closed_training_cpu_v2 import OUT as BENCH,CONFIGS
from scripts.vision.train_closed_source_control import cleanup

def ready():
    p=freeze();r=prior.read(OUT/'entry-ready.json');prior.verify(r)
    from scripts.vision.review_closed_gamma import main as review
    review()
    for mode in ('identity','gamma'):
        prior.verify(prior.read(OUT/'entry-probes'/mode/'completion.json'))
    if r['status']!='ready_for_training_not_started' or r['cells']!=list(KEYS):raise ValueError('Not ready')
    b=prior.read(BENCH/'completion.json');prior.verify(b)
    if r['cpu_selection']!=b['selected']:raise ValueError('CPU policy drift')
    import platform,torch,ultralytics
    if p['environment']!=dict(python=platform.python_version(),torch=torch.__version__,ultralytics=ultralytics.__version__):raise ValueError('Environment drift')
    for key in KEYS:receipt(key,p)
    return p,CONFIGS[b['selected']]

def complete(key):
    p,cpu=ready();r=prior.read(OUT/'training'/key/'completion.json');prior.verify(r)
    expected,_=receipt(key,p);x=prior.read(r['exposure_path']);prior.verify(x)
    if r['optimizer_steps']!=len(p['schedules'][key])//6 or r['design_identity']!=p['identity'] or r['cell']!=key:raise ValueError('Completion identity')
    check(p,key,x['actual'],x['brightness_log'],x['gamma_log'])
    if x['batch_records']!=expected['batch_records'] or x['brightness_log']!=expected['brightness_log'] or x['gamma_log']!=expected['gamma_log']:raise ValueError('Actual batch mismatch')
    if prior.file_sha256(r['weights'])!=r['weights_sha256'] or x['effective_threads']!=cpu[0]:raise ValueError('Endpoint/thread drift')

def worker(key):
    p,(threads,_)=ready();prior.verify(prior.read(OUT/'training-authorization.json'))
    root=OUT/'training'/key;root.mkdir(parents=True,exist_ok=True)
    if (root/'completion.json').exists():complete(key);return
    if any(prior.read(f)['semantic'] for f in root.glob('attempt-*/failure.json')):raise ValueError('Semantic failure requires investigation')
    n=len(list(root.glob('attempt-*')))+1
    if n>3:raise ValueError('Training attempt cap')
    attempt=root/f'attempt-{n:03}';attempt.mkdir()
    try:
        expected,ep=receipt(key,p);x=execute(p,key,attempt,threads,expected)
        import yaml
        args=attempt/'run/args.yaml';curve=attempt/'run/results.csv';a=yaml.safe_load(args.read_text())
        if any(a.get(k)!=v for k,v in p['training_config'][key].items()):raise ValueError('Actual config differs')
        with curve.open() as stream:rows=list(csv.DictReader(stream))
        if len(rows)!=p['training_config'][key]['epochs']:raise ValueError('Incomplete loss curve')
        for row in rows:
            lr=[float(v) for k,v in row.items() if k.startswith('lr/')]
            if len(lr)!=3 or any(abs(v-.0005)>1e-12 for v in lr):raise ValueError('Learning rate drift')
            if any(not math.isfinite(float(v)) for k,v in row.items() if k.startswith('train/')):raise ValueError('Nonfinite curve')
        xp=attempt/'exposure.json';prior.frozen(xp,dict(**x,inputs={str(ep):prior.file_sha256(ep),str(OUT/'entry-ready.json'):prior.file_sha256(OUT/'entry-ready.json')}))
        weight=attempt/'run/weights/last.pt';deps=[OUT/'entry-ready.json',OUT/'training-authorization.json',xp,weight,args,curve,Path(__file__).resolve()]
        prior.frozen(root/'completion.json',dict(status='trained_not_evaluated',cell=key,design_identity=p['identity'],optimizer_steps=x['optimizer_steps'],weights=str(weight),weights_sha256=prior.file_sha256(weight),exposure_path=str(xp),
            checkpoint_selection='terminal_last_only',training_validation_role='fit_only_not_acceptance',inputs={str(d):prior.file_sha256(d) for d in deps}))
        complete(key)
    except BaseException as exc:
        prior.frozen(attempt/'failure.json',dict(semantic=isinstance(exc,ValueError),error=traceback.format_exc()));raise

def run():
    p,(threads,parallel)=ready()
    with (OUT/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        auth=OUT/'training-authorization.json'
        if auth.exists():prior.verify(prior.read(auth))
        else:prior.frozen(auth,dict(status='explicit_goal_authorizes_three_gamma_training_units',cells=list(KEYS),inputs={str(OUT/'entry-ready.json'):prior.file_sha256(OUT/'entry-ready.json')}))
        for i in range(0,len(KEYS),parallel):
            jobs=[]
            try:
                for key in KEYS[i:i+parallel]:
                    if (OUT/'training'/key/'completion.json').exists():complete(key);continue
                    root=OUT/'workers'/key;root.mkdir(parents=True,exist_ok=True);n=len(list(root.glob('attempt-*')))+1
                    if n>3:raise ValueError('Worker attempt cap')
                    attempt=root/f'attempt-{n:03}';attempt.mkdir();log=(attempt/'log.txt').open('x')
                    proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.train_closed_gamma','--worker',key,'--train'],cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    jobs.append((proc,log,attempt,key));print('TRAINING_STARTED',key,proc.pid,'THREADS',threads,flush=True)
                for proc,_,_,key in jobs:
                    proc.wait(timeout=21600)
                    if proc.returncode:raise RuntimeError('Training failed '+key)
                    complete(key);print('TRAINED_VERIFIED',key,flush=True)
            except BaseException:
                for proc,_,attempt,key in jobs:
                    cleanup(proc);prior.frozen(attempt/'failure.json',dict(cell=key,error=traceback.format_exc(),cleanup_confirmed=proc.poll() is not None))
                raise
            finally:
                for proc,log,_,_ in jobs:cleanup(proc);log.close()
        dest=OUT/'training/completion.json'
        if not dest.exists():prior.frozen(dest,dict(status='all_three_trained_evaluation_pending',cells=list(KEYS),inputs={str(OUT/'training'/k/'completion.json'):prior.file_sha256(OUT/'training'/k/'completion.json') for k in KEYS}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);a=ap.parse_args()
    if a.worker and not a.train:ap.error('Worker requires explicit training')
    if a.worker:worker(a.worker)
    elif a.train:run()
    else:ready();print('READY_NO_TRAINING_STARTED')
