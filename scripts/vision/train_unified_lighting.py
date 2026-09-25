"""Explicit six-cell training using the verified paired loaders; default read-only."""
import argparse,fcntl,os,signal,subprocess,sys,traceback
from pathlib import Path
from scripts.vision.prepare_unified_lighting_training import OUT,prior
from scripts.vision import train_whole_image_hold as adapter
from scripts.vision.brightness_transfer_runtime import make_dataset,check_log
from scripts.vision.order_retention_runtime import make_loader,check_actual
from scripts.vision.brightness_lr_retention import overrides,check_curve
from scripts.vision.preflight_unified_lighting import tensor_hash

KEYS=tuple(f'{arm}-{seed}' for seed in (7,17,27) for arm in ('R-clean','L-physical'))

def ready():
    p=prior.read(OUT/'protocol.json');r=prior.read(OUT/'ready.json');prior.verify(p);prior.verify(r)
    if r['status']!='ready_for_training_not_started' or set(r['cells'])!=set(KEYS):raise ValueError('Invalid readiness')
    import torch,ultralytics
    if {'torch':torch.__version__,'ultralytics':ultralytics.__version__}!=p['environment']:raise ValueError('Environment changed')
    if prior.file_sha256(Path(p['initialization']['path']))!=p['initialization']['sha256']:raise ValueError('Initialization changed')
    for seed in (7,17,27):
        found=list((OUT/'loader-checks'/f'seed-{seed}').glob('attempt-*/complete.json'))
        if len(found)!=1:raise ValueError('Preflight identity ambiguous')
        cell=prior.read(found[0]);prior.verify(cell)
        for key,c in cell['cells'].items():
            check_actual(p,key,c['actual']);check_log(p,key,c['brightness_log'])
            if p['training_config'][key]!=overrides(seed):raise ValueError('Training configuration changed')
    return p

def expected_cell(key):
    seed=int(key.split('-')[-1]);found=list((OUT/'loader-checks'/f'seed-{seed}').glob('attempt-*/complete.json'))
    if len(found)!=1:raise ValueError('Preflight missing/duplicate')
    r=prior.read(found[0]);prior.verify(r);return r['cells'][key]

def complete(key,p):
    cp=OUT/'training'/key/'completion.json';adapter.complete(cp,p,key)
    bp=OUT/'training'/key/'tensor-completion.json';b=prior.read(bp);prior.verify(b)
    if b['batch_count']!=450 or not b['preflight_tensors_equal']:raise ValueError('Tensor audit incomplete')
    check_log(p,key,b['brightness_log'])
    if b['brightness_log']!=expected_cell(key)['brightness_log']:raise ValueError('Training augmentation differs')

def worker(key):
    import torch,csv,yaml
    torch.set_num_threads(4);p=ready();root=OUT/'training'/key
    if (root/'tensor-completion.json').exists():complete(key,p);return
    if (root/'completion.json').exists():raise ValueError('Base completion without tensor receipt; preserve and investigate')
    expected=expected_cell(key);log=[];steps=[0]
    def checked_loader(dataset,protocol,cell,owner):
        loader=make_loader(dataset,protocol,cell,owner);collate=loader.collate_fn
        def checked_collate(items):
            batch=collate(items);n=steps[0]
            if n>=450:raise ValueError('Extra actual batch')
            e=expected['batch_records'][n]
            if tensor_hash(batch['img'])!=e['image_tensor_sha256']:raise ValueError('Training image tensor differs from preflight')
            for field,sha in e['full_supervision'].items():
                if tensor_hash(batch[field])!=sha:raise ValueError('Training supervision tensor differs')
            steps[0]+=1;return batch
        loader.collate_fn=checked_collate;return loader
    adapter.OUT=OUT;adapter.KEYS=KEYS;adapter.ready=lambda:p;adapter.overrides=overrides
    adapter.make_dataset=lambda protocol,cell:make_dataset(protocol,cell,log);adapter.make_loader=checked_loader
    adapter.worker(key)
    if steps[0]!=450 or log!=expected['brightness_log']:raise ValueError('Training exposure/augmentation incomplete')
    cp=root/'completion.json';c=prior.read(cp);prior.verify(c);attempt=Path(c['exposure_path']).parent
    args=yaml.safe_load((attempt/'run/args.yaml').read_text());curve=list(csv.DictReader((attempt/'run/results.csv').open()))
    check_curve(args,curve,int(key.split('-')[-1]),.0005)
    prior.frozen(root/'tensor-completion.json',dict(status='complete_actual_tensors_verified',batch_count=steps[0],preflight_tensors_equal=True,
        brightness_log=log,learning_rate=.0005,training_member_validation_role='fitting_diagnostic_only',
        inputs={str(x):prior.file_sha256(x) for x in [cp,OUT/'ready.json',Path(__file__).resolve(),attempt/'run/args.yaml',attempt/'run/results.csv']}))
    complete(key,p)

def run():
    p=ready();paths=[OUT/'ready.json',Path(__file__).resolve()]
    with (OUT/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        auth=OUT/'training-authorization.json'
        if not auth.exists():prior.frozen(auth,dict(status='explicit_six_cell_training_authorized',cells=list(KEYS),request='开始训练吧',
            inputs={str(x):prior.file_sha256(x) for x in paths}))
        else:prior.verify(prior.read(auth))
        for key in KEYS:
            if (OUT/'training'/key/'tensor-completion.json').exists():complete(key,p);continue
            root=OUT/'workers'/key;root.mkdir(parents=True,exist_ok=True);n=len(list(root.glob('attempt-*')))+1
            if n>3:raise ValueError('Technical attempt budget exhausted')
            attempt=root/f'attempt-{n:03}';attempt.mkdir();proc=None
            try:
                with (attempt/'log.txt').open('x') as log:
                    proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.train_unified_lighting','--train','--worker',key],cwd=prior.ROOT,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    prior.frozen(attempt/'launch.json',dict(pid=proc.pid,cell=key,inputs={str(x):prior.file_sha256(x) for x in [auth,Path(__file__).resolve()]}))
                    print('TRAINING_STARTED',key,'PID',proc.pid,flush=True);proc.wait(timeout=21600)
                    if proc.returncode:raise RuntimeError('Training worker failed; inspect '+str(attempt))
                complete(key,p);prior.frozen(attempt/'completion.json',dict(status='complete',process_cleanup_complete=True,inputs={str(attempt/'log.txt'):prior.file_sha256(attempt/'log.txt')}))
            except BaseException:
                if proc is not None and proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
                prior.frozen(attempt/'failure.json',dict(status='failed',error=traceback.format_exc(),process_cleanup_complete=proc is None or proc.poll() is not None));raise
            print('TRAINING_COMPLETE',key,flush=True)
        paths += [OUT/'training'/key/'tensor-completion.json' for key in KEYS]
        prior.frozen(OUT/'training/completion.json',dict(status='six_cells_trained_evaluation_not_run',cells=list(KEYS),inputs={str(x):prior.file_sha256(x) for x in paths}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);a=ap.parse_args()
    if a.worker and not a.train:ap.error('--worker requires --train')
    if a.worker:worker(a.worker)
    elif a.train:run()
    else:ready();print('READY_ONLY_NO_TRAINING')
