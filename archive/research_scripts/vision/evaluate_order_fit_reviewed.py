"""Fixed reviewed-pool endpoint evaluation; no training or automatic review."""
import argparse
import fcntl
import signal
import subprocess
import sys
import traceback
from pathlib import Path
from scripts.vision.train_order_fit_reviewed import DEST,KEYS,contract,ready,bind,prior
from scripts.vision.order_fit_reviewed_schedule import run as freeze
from scripts.vision.diagnose_small_scale_order_fit import TRAIN
from scripts.vision import evaluate_physical_low_light as fixed
from scripts.vision.train_closed_source_control import cleanup

def complete(key): return bind().complete(key)

def evaluate(key):
    if key not in KEYS: raise ValueError('Unknown unit')
    fixed.OUT=DEST; fixed.KEYS=KEYS; fixed.complete=complete
    fixed.contract=lambda key:(None,freeze(),None)
    r=fixed.evaluate(key)
    path=DEST/'evaluation'/f'{key}-reviewed-binding.json'
    if path.exists(): prior.verify(prior.read(path))
    else:
        deps=[Path(__file__).resolve(),Path(fixed.__file__).resolve(),DEST/'evaluation'/f'{key}.json']
        prior.frozen(path,dict(status='reviewed_pool_adapter_bound',inputs={str(d):prior.file_sha256(d) for d in deps}))
    return r

def finish():
    records={k:evaluate(k) for k in KEYS}; p=freeze(); old={}
    deps=[DEST/'design.json',Path(__file__).resolve()]
    for k in KEYS:
        ep=TRAIN/'evaluation'/f'{k}.json'; old[k]=prior.read(ep); fixed.validate_record(old[k],k)
        cp=prior.read(TRAIN/'training'/k/'completion.json'); prior.verify(cp)
        if old[k]['inputs'].get(cp['weights'])!=cp['weights_sha256']: raise ValueError('Direct reference identity mismatch')
        deps += [ep,TRAIN/'training'/k/'completion.json',DEST/'evaluation'/f'{k}.json',DEST/'evaluation'/f'{k}-reviewed-binding.json']
    refs=[fixed.PRIOR/f'evaluation-retained_reference-450-{s}.json' for s in (7,17,27)]
    reference=[prior.read(x) for x in refs]; hp=Path(p['evaluation']['historical_reference']); h=prior.read(hp)
    for r in reference+[h]: prior.verify(r)
    _,policy,_=fixed.reference_contract('fixed-7')
    if any(p['acceptance_policy'][k]!=policy[k] for k in ('acceptance_policy','retention')): raise ValueError('Policy drift')
    groups={f:fixed.aggregate([records[f'{f}-{s}'] for s in (7,17,27)]) for f in ('ISR1100','ISM1100')}
    controls={f:fixed.aggregate([old[f'{f}-{s}'] for s in (7,17,27)]) for f in groups}
    baseline=fixed.baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40: raise ValueError('Baseline drift')
    deps+=refs+[hp]; dest=DEST/'evaluation/summary.json'
    if dest.exists(): r=prior.read(dest); prior.verify(r); return r
    return prior.frozen(dest,dict(status='numerical_complete_explicit_error_review_pending',groups=groups,direct_controls=controls,
        policies={f:fixed.policy_checks(g,fixed.aggregate(reference),h['historical_A'],policy) for f,g in groups.items()},
        direct_changes={f:fixed.direct_checks(g,controls[f]) for f,g in groups.items()},
        planned_paired_changes={f:{str(s):fixed.paired_change(old[f'{f}-{s}']['rows'],records[f'{f}-{s}']['rows']) for s in (7,17,27)} for f in groups},
        matching_conflicts=sum(r['matching_conflicts'] for r in records.values()),baseline=baseline,
        comparison=p['comparison'],review_complete=False,selected_candidate=None,inputs={str(d):prior.file_sha256(d) for d in deps}))

def run():
    contract(); ready(); root=DEST/'evaluation-runner'; root.mkdir(exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def stop(signum,frame): raise KeyboardInterrupt(signum)
        signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
        for k in KEYS: complete(k)
        for start in range(0,len(KEYS),2):
            jobs=[]
            try:
                for k in KEYS[start:start+2]:
                    if (DEST/'evaluation'/f'{k}.json').exists(): evaluate(k); continue
                    folder=root/k; folder.mkdir(exist_ok=True); n=len(list(folder.glob('attempt-*')))+1
                    if n>3: raise ValueError('Inference launch cap')
                    attempt=folder/f'attempt-{n:03}'; attempt.mkdir(); log=(attempt/'log.txt').open('x')
                    proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.evaluate_order_fit_reviewed','--evaluate','--cell',k],cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    jobs.append((proc,log,attempt)); print('EVALUATION_STARTED',k,proc.pid,flush=True)
                for proc,_,_ in jobs:
                    proc.wait(timeout=1800)
                    if proc.returncode: raise RuntimeError('Evaluation worker failed')
            except BaseException:
                error=traceback.format_exc()
                for proc,_,attempt in jobs:
                    cleanup(proc); prior.frozen(attempt/'failure.json',dict(error=error,process_cleanup_complete=proc.poll() is not None))
                raise
            finally:
                for proc,log,_ in jobs: cleanup(proc); log.close()
        print(finish()['status'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--evaluate',action='store_true'); ap.add_argument('--cell',choices=KEYS); a=ap.parse_args()
    if a.cell and not a.evaluate: ap.error('Explicit --evaluate required')
    if a.cell: evaluate(a.cell)
    elif a.evaluate: run()
    else: contract(); ready(); print('PREFLIGHT_ONLY_NO_INFERENCE')
