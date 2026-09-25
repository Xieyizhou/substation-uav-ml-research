"""Fixed formal/diagnostic evaluation of six 1100-step endpoints."""
import argparse
import fcntl
import subprocess
import sys
import signal
import traceback
from pathlib import Path
from scripts.vision.small_scale_control import OUT, REF, KEYS, prior
from scripts.vision.closed_budget_runtime import check
from scripts.vision import evaluate_physical_low_light as fixed
from scripts.vision.train_closed_source_control import cleanup


def protocol():
    p = prior.read(OUT/'design.json'); prior.verify(p); return p


def complete(key):
    if key not in KEYS: raise ValueError('Unknown unit')
    p = protocol(); c = prior.read(OUT/'training'/key/'completion.json'); prior.verify(c)
    x = prior.read(c['exposure_path']); prior.verify(x)
    if c['optimizer_steps'] != 1100 or x['optimizer_steps'] != 1100: raise ValueError('Wrong endpoint')
    check(p, key, x['actual'], x['brightness_log'])
    if prior.file_sha256(c['weights']) != c['weights_sha256']: raise ValueError('Weight drift')
    return c


def evaluate(key):
    fixed.OUT = OUT; fixed.KEYS = KEYS; fixed.complete = complete
    fixed.contract = lambda key: (protocol(), protocol(), None)
    result = fixed.evaluate(key)
    path = OUT/'evaluation'/f'{key}-small-binding.json'
    if path.exists(): prior.verify(prior.read(path))
    else:
        deps = [Path(__file__).resolve(), Path(fixed.__file__).resolve(), OUT/'evaluation'/f'{key}.json']
        prior.frozen(path, dict(status='small_adapter_bound', inputs={str(d):prior.file_sha256(d) for d in deps}))
    return result


def finish():
    path = OUT/'evaluation/summary.json'
    if path.exists():
        r = prior.read(path); prior.verify(r); return r
    records = {k:evaluate(k) for k in KEYS}
    deps = [OUT/'design.json', Path(__file__).resolve()]
    deps += [OUT/'evaluation'/f'{k}{suffix}.json' for k in KEYS for suffix in ('', '-small-binding')]
    p = protocol(); hp = Path(p['evaluation']['historical_reference'])
    h = prior.read(hp); prior.verify(h); deps.append(hp)
    historical = []; previous = []
    for seed in (7,17,27):
        for path, target in ((fixed.PRIOR/f'evaluation-retained_reference-450-{seed}.json', historical),
                             (REF/'evaluation'/f'ICG1000-{seed}.json', previous)):
            r = prior.read(path); prior.verify(r); target.append(r); deps.append(path)
    groups = {family:fixed.aggregate([records[f'{family}-{s}'] for s in (7,17,27)]) for family in ('SR1100','SM1100')}
    _, policy, _ = fixed.reference_contract('fixed-7')
    baseline = fixed.baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified'] != 40: raise ValueError('Baseline drift')
    return prior.frozen(OUT/'evaluation/summary.json', dict(status='numerical_complete_explicit_review_pending',
        groups=groups, ICG1000=fixed.aggregate(previous),
        policies={k:fixed.policy_checks(g, fixed.aggregate(historical), h['historical_A'], policy) for k,g in groups.items()},
        material_relative_reference=fixed.direct_checks(groups['SM1100'], groups['SR1100']),
        paired_changes={str(s):fixed.paired_change(records[f'SR1100-{s}']['rows'], records[f'SM1100-{s}']['rows']) for s in (7,17,27)},
        matching_conflicts=sum(r['matching_conflicts'] for r in records.values()),
        review_complete=False, selected_candidate=None, baseline=baseline,
        inputs={str(d):prior.file_sha256(d) for d in deps}))


def run():
    protocol(); root = OUT/'evaluation-runner'; root.mkdir(exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def stop(signum, frame): raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        for key in KEYS: complete(key)
        for start in range(0,len(KEYS),2):
            jobs = []
            try:
                for key in KEYS[start:start+2]:
                    if (OUT/'evaluation'/f'{key}.json').exists(): evaluate(key); continue
                    folder = root/key; folder.mkdir(exist_ok=True)
                    n = len(list(folder.glob('attempt-*'))) + 1
                    if n > 3: raise ValueError('Inference attempt cap')
                    attempt = folder/f'attempt-{n:03}'; attempt.mkdir()
                    log = (attempt/'log.txt').open('x')
                    proc = subprocess.Popen([sys.executable,'-u','-m','scripts.vision.evaluate_small_scale','--evaluate','--cell',key],
                        cwd=prior.ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    jobs.append((proc,log,attempt)); print('EVALUATION_STARTED',key,proc.pid,flush=True)
                for proc,_,_ in jobs:
                    proc.wait(timeout=1800)
                    if proc.returncode: raise RuntimeError('Evaluation failed; inspect attempt')
            except BaseException:
                error = traceback.format_exc()
                for proc,_,attempt in jobs:
                    cleanup(proc)
                    prior.frozen(attempt/'failure.json', dict(error=error, process_cleanup_complete=proc.poll() is not None))
                raise
            finally:
                for proc,log,_ in jobs: cleanup(proc); log.close()
        finish(); print('NUMERICAL_COMPLETE_AI_REVIEW_PENDING',flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--evaluate',action='store_true'); ap.add_argument('--cell',choices=KEYS); a=ap.parse_args()
    if a.cell and not a.evaluate: ap.error('Explicit evaluation required')
    if a.cell: evaluate(a.cell)
    elif a.evaluate: run()
    else: protocol(); print('PREFLIGHT_ONLY_NO_INFERENCE')
