"""Explicit three-seed training, audited against real brightness preflight."""
import argparse,fcntl,os,signal,subprocess,sys,traceback
from pathlib import Path
from scripts.vision.prepare_brightness_transfer import OUT,SOURCE,ROOT,KEYS,ready,read,verify,frozen,file_sha256
from scripts.vision import train_whole_image_hold as adapter
from scripts.vision.brightness_transfer_runtime import make_dataset,check_log
from scripts.vision.order_retention_runtime import overrides
from scripts.vision.prepare_whole_image_hold import baseline_verify

def pretrain():
    p=ready();verify(read(OUT/'ready.json'));path=OUT/'training-gate.json'
    if path.exists():verify(read(path));return p
    import torch,ultralytics,yaml
    paths=[OUT/'ready.json',OUT/'protocol.json',Path(__file__),ROOT/'scripts/vision/brightness_transfer_runtime.py',Path(adapter.__file__)]
    for seed in (7,17,27):
        c=read(p['baselines'][str(seed)]['training_receipt']);args=Path(c['weights']).parents[1]/'args.yaml';a=yaml.safe_load(args.read_text())
        if any(a.get(k)!=v for k,v in overrides(seed).items()):raise ValueError('Historical control config drift')
        ck=torch.load(c['weights'],map_location='cpu',weights_only=False)
        if ck.get('version')!=ultralytics.__version__:raise ValueError('Historical Ultralytics incompatible')
        logs=[]
        for lp in (SOURCE/'workers'/f'lineage-capped-450-{seed}').glob('attempt-*/log.txt'):
            lines=lp.read_text().splitlines()
            if any(f'Ultralytics {ultralytics.__version__}' in l and f'torch-{torch.__version__}' in l and 'CPU (Apple M2 Pro)' in l for l in lines):logs.append(lp)
        if len(logs)!=1:raise ValueError('Historical environment not uniquely verified')
        paths.extend([args,*logs])
    tests=['tests.test_brightness_transfer','tests.test_lineage_training_fit','tests.test_lineage_capped_control','tests.test_whole_image_hold_train']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if t.returncode:raise ValueError(t.stderr)
    paths += [ROOT/(x.replace('.','/')+'.py') for x in tests]
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Pinned baseline failure')
    frozen(path,dict(status='ready_for_three_explicit_trainings',regressions=t.stderr,baseline=baseline,historical_environment_verified=True,whole_repository_tested=False,inputs={str(x):file_sha256(x) for x in paths}))
    return p

def complete(key,p):
    adapter.complete(OUT/'training'/key/'completion.json',p,key)
    b=read(OUT/'training'/key/'brightness-receipt.json');verify(b)
    check_log(p,key,b['actual_brightness'])
    u=read(OUT/'preflight'/f"{key.split('-')[-1]}.json");verify(u)
    if b['actual_brightness']!=u['logs'][key]:raise ValueError('Training tensors not equal frozen augmentation preflight')

def worker(key):
    p=pretrain();actual=[]
    adapter.OUT=OUT;adapter.KEYS=KEYS;adapter.ready=lambda:p
    adapter.make_dataset=lambda p,k:make_dataset(p,k,actual)
    (OUT/'export').mkdir(exist_ok=True)
    adapter.worker(key)
    check_log(p,key,actual)
    pre=OUT/'preflight'/f"{key.split('-')[-1]}.json"
    if actual!=read(pre)['logs'][key]:raise ValueError('Actual augmentation drift')
    dest=OUT/'training'/key/'brightness-receipt.json'
    frozen(dest,dict(status='complete_brightness_verified',actual_brightness=actual,inputs={str(x):file_sha256(x) for x in [OUT/'training-gate.json',OUT/'training'/key/'completion.json',pre,Path(__file__),ROOT/'scripts/vision/brightness_transfer_runtime.py']}))
    complete(key,p)

def run():
    p=pretrain()
    with (OUT/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for key in KEYS:
            if (OUT/'training'/key/'brightness-receipt.json').exists():complete(key,p);continue
            if (OUT/'training'/key/'completion.json').exists():raise ValueError('Unbound completed training; investigate rather than rerun or mix state')
            root=OUT/'workers'/key;root.mkdir(parents=True,exist_ok=True);n=len(list(root.glob('attempt-*')))+1
            if n>3:raise ValueError('Worker attempt budget exhausted')
            attempt=root/f'attempt-{n:03}';attempt.mkdir();proc=None
            try:
                with (attempt/'log.txt').open('x') as log:
                    proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.train_brightness_transfer','--train','--worker',key],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    frozen(attempt/'launch.json',dict(pid=proc.pid,cell=key,inputs={str(OUT/'training-gate.json'):file_sha256(OUT/'training-gate.json')}))
                    proc.wait(timeout=21600)
                    if proc.returncode:raise RuntimeError('Training worker failed '+key)
                complete(key,p)
                frozen(attempt/'completion.json',dict(status='complete',process_cleanup_confirmed=True,inputs={str(attempt/'log.txt'):file_sha256(attempt/'log.txt')}))
            except BaseException:
                if proc and proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
                frozen(attempt/'failure.json',dict(error=traceback.format_exc(),process_cleanup_confirmed=proc is None or proc.poll() is not None));raise
            print('TRAINED',key,flush=True)
        frozen(OUT/'training/completion.json',dict(status='three_trainings_complete_evaluation_pending',inputs={str(OUT/'training'/k/'brightness-receipt.json'):file_sha256(OUT/'training'/k/'brightness-receipt.json') for k in KEYS}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pretrain',action='store_true');ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);a=ap.parse_args()
    if a.worker and not a.train:ap.error('Worker requires explicit train')
    if a.train:worker(a.worker) if a.worker else run()
    elif a.pretrain:pretrain()
    else:print('READ_ONLY_NO_TRAINING')
