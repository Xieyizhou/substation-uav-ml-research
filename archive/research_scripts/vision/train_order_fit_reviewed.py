"""Independent reviewed-pool entry. Default verifies; --train explicitly runs."""
import argparse
import platform
import subprocess
import sys
from pathlib import Path
from scripts.vision.order_fit_reviewed_schedule import DEST, run as freeze, prior
from scripts.vision.closed_budget_runtime import check
from scripts.vision.exposure_order_retention import baseline_verify

KEYS=tuple(f'{family}1100-{seed}' for seed in (7,17,27) for family in ('ISR','ISM'))
TESTS=('test_order_fit_reviewed_schedule','test_order_fit_reviewed_counts','test_material_mask_coverage',
       'test_small_scale_review_gate','test_order_fit_exposure_impact','test_order_fit_closeout')

def contract():
    p=freeze(); units=[]; deps=[DEST/'design.json',Path(__file__).resolve()]
    for key in KEYS:
        path=DEST/'actual-preflight'/f'{key}.json'; u=prior.read(path); prior.verify(u)
        if u['key']!=key or u['status']!='actual_complete_loader_verified_no_training': raise ValueError('Wrong preflight identity')
        if any(u[k] for k in ('optimizer_created','backward_executed','validation_run')): raise ValueError('Training in preflight')
        if u['checked_draws']!=6600 or u['checked_batches']!=1100 or len(u['batch_records'])!=1100: raise ValueError('Incomplete preflight')
        check(p,key,u['actual'],u['brightness_log']); units.append(dict(u,cell=key)); deps.append(path)
    import torch,ultralytics
    if p['environment']!=dict(python=platform.python_version(),torch=torch.__version__,ultralytics=ultralytics.__version__): raise ValueError('Environment drift')
    init=Path(p['initialization']['path'])
    if prior.file_sha256(init)!=p['initialization']['sha256']: raise ValueError('Initialization drift')
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40: raise ValueError('Baseline drift')
    deps += [init,Path(__file__).with_name('closed_budget_engine_v2.py'),Path(__file__).with_name('train_small_scale.py')]
    return p,dict(units=units),deps,b

def ready():
    path=DEST/'entry-ready.json'; r=prior.read(path); prior.verify(r)
    if r['status']!='ready_for_training_not_started' or r['cells']!=list(KEYS) or r['actual_draws_verified']!=39600: raise ValueError('Readiness incomplete')
    if not r['relevant_regressions_passed'] or not r['exact_image_and_label_tensors_frozen']: raise ValueError('Missing tests/tensors')
    if any(r[k] for k in ('optimizer_created','backward_executed','training_admitted','promotable')): raise ValueError('Readiness flags')
    return path

def prepare():
    _,_,deps,b=contract()
    if (DEST/'entry-ready.json').exists(): return ready()
    result=subprocess.run([sys.executable,'-m','unittest',*('tests.'+x for x in TESTS)],cwd=prior.ROOT,text=True,capture_output=True)
    if result.returncode: raise RuntimeError(result.stdout+result.stderr)
    deps += [prior.ROOT/'tests'/f'{x}.py' for x in TESTS]
    prior.frozen(DEST/'entry-ready.json',dict(status='ready_for_training_not_started',cells=list(KEYS),actual_draws_verified=39600,
        relevant_regressions_passed=True,regression_output=result.stdout+result.stderr,exact_image_and_label_tensors_frozen=True,
        optimizer_created=False,backward_executed=False,baseline=b,inputs={str(x):prior.file_sha256(x) for x in deps}))
    return ready()

def bind():
    from scripts.vision import train_interleaved_small_scale as engine
    engine.OUT=DEST; engine.KEYS=KEYS; engine.freeze=freeze; engine.contract=contract; engine.ready=ready
    return engine

def train():
    # Serial pairs use the already-tested worker implementation, with this entry's
    # own subprocess target and independent artifacts.
    import fcntl
    from scripts.vision.train_closed_source_control import cleanup
    contract(); ready(); root=DEST/'training-runner'; root.mkdir(exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for offset in range(0,len(KEYS),2):
            jobs=[]
            try:
                for key in KEYS[offset:offset+2]:
                    if (DEST/'training'/key/'completion.json').exists(): bind().complete(key); continue
                    folder=root/key; folder.mkdir(exist_ok=True)
                    n=len(list(folder.glob('attempt-*')))+1
                    if n>3: raise ValueError('Attempt cap')
                    attempt=folder/f'attempt-{n:03}'; attempt.mkdir(); log=(attempt/'log.txt').open('x')
                    proc=subprocess.Popen([sys.executable,'-u','-m',__name__ if __name__!='__main__' else 'scripts.vision.train_order_fit_reviewed','--train','--worker',key],cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    jobs.append((proc,log,key)); print('TRAINING_STARTED',key,proc.pid,'THREADS=4',flush=True)
                for proc,_,key in jobs:
                    proc.wait(timeout=14400)
                    if proc.returncode: raise RuntimeError('Worker failed '+key)
                    bind().complete(key); print('TRAINING_VERIFIED',key,flush=True)
            finally:
                for proc,log,_ in jobs: cleanup(proc); log.close()
        deps=[DEST/'training'/key/'completion.json' for key in KEYS]
        prior.frozen(root/'completion.json',dict(status='six_units_complete_evaluation_pending',inputs={str(x):prior.file_sha256(x) for x in deps}))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--prepare',action='store_true'); ap.add_argument('--train',action='store_true'); ap.add_argument('--worker',choices=KEYS); a=ap.parse_args()
    if a.worker and not a.train: ap.error('--worker requires --train')
    if a.prepare: print(prepare())
    elif a.worker: bind().worker(a.worker)
    elif a.train: train()
    else: contract(); print('PREFLIGHT_ONLY_NO_TRAINING')
