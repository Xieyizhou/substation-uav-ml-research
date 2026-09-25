"""Explicit six-unit training; two CPU workers, four threads each."""
import argparse
import fcntl
import platform
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from scripts.vision.small_scale_control import OUT, KEYS, freeze, prior
from scripts.vision.closed_budget_runtime import check
from scripts.vision.closed_budget_engine_v2 import execute
from scripts.vision.exposure_order_retention import baseline_verify
from scripts.vision.train_closed_source_control import cleanup


def contract():
    p = freeze(); fp = OUT/'loader-feasibility.json'
    f = prior.read(fp); prior.verify(f)
    if any(f[k] for k in ('optimizer_created', 'backward_executed', 'validation_run')):
        raise ValueError('Invalid preflight')
    if [u['cell'] for u in f['units']] != list(KEYS): raise ValueError('Missing preflight units')
    for u in f['units']:
        check(p, u['cell'], u['actual'], u['brightness_log'])
        if len(u['batch_records']) != 1100: raise ValueError('Incomplete batches')
    import torch, ultralytics
    if p['environment'] != dict(python=platform.python_version(), torch=torch.__version__, ultralytics=ultralytics.__version__):
        raise ValueError('Environment drift')
    init = Path(p['initialization']['path'])
    if prior.file_sha256(init) != p['initialization']['sha256']: raise ValueError('Initialization drift')
    b = baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified'] != 40: raise ValueError('Baseline drift')
    deps = [OUT/'design.json', OUT/'quality-review.json', fp, init, Path(__file__).resolve(),
            Path(__file__).with_name('closed_budget_engine_v2.py')]
    return p, f, deps, b


def ready():
    path = OUT/'entry-ready.json'; r = prior.read(path); prior.verify(r)
    if r['status'] != 'ready_for_training_not_started' or r['cells'] != list(KEYS) or r['actual_draws_verified'] != 39600:
        raise ValueError('Readiness incomplete')
    if not r['relevant_regressions_passed'] or not r['exact_image_and_label_tensors_frozen']:
        raise ValueError('Missing regression/tensor gate')
    if any(r[k] for k in ('optimizer_created', 'backward_executed', 'training_admitted', 'promotable')):
        raise ValueError('Invalid readiness flags')
    return path


def worker(key):
    rp = ready(); p, f, deps, _ = contract(); deps.append(rp)
    unit = next(u for u in f['units'] if u['cell'] == key)
    root = OUT/'training'/key; root.mkdir(parents=True, exist_ok=True)
    with (root/'unit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cp = root/'completion.json'
        if cp.exists():
            r = prior.read(cp); prior.verify(r)
            if r['optimizer_steps'] != 1100 or prior.file_sha256(r['weights']) != r['weights_sha256']:
                raise ValueError('Invalid reusable unit')
            return r
        n = len(list(root.glob('attempt-*'))) + 1
        if n > 3: raise ValueError('Attempt cap')
        if any(prior.read(x).get('semantic') for x in root.glob('attempt-*/failure.json')):
            raise ValueError('Unresolved semantic failure')
        attempt = root/f'attempt-{n:03}'; attempt.mkdir()
        try:
            x = execute(p, key, attempt, 4, unit)
            if x['optimizer_steps'] != 1100: raise ValueError('Wrong endpoint')
            xp = attempt/'exposure.json'
            prior.frozen(xp, dict(x, inputs={str(d):prior.file_sha256(d) for d in deps}))
            wp = attempt/'run/weights/last.pt'
            if not wp.is_file(): raise ValueError('Missing last.pt')
            deps += [xp, wp]
            return prior.frozen(cp, dict(status='trained_not_evaluated', cell=key,
                weights=str(wp), weights_sha256=prior.file_sha256(wp), exposure_path=str(xp),
                optimizer_steps=1100, inputs={str(d):prior.file_sha256(d) for d in deps}))
        except BaseException as exc:
            prior.frozen(attempt/'failure.json', dict(error=traceback.format_exc(),
                semantic=isinstance(exc, ValueError), child_processes_started=0))
            raise


def run():
    contract(); ready(); root = OUT/'training-runner'; root.mkdir(exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def stop(signum, frame): raise KeyboardInterrupt(f'Signal {signum}')
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        started = time.monotonic()
        for start in range(0, len(KEYS), 2):
            jobs = []
            try:
                for key in KEYS[start:start+2]:
                    cp = OUT/'training'/key/'completion.json'
                    if cp.exists():
                        r = prior.read(cp); prior.verify(r)
                        if r['optimizer_steps'] != 1100: raise ValueError('Reusable endpoint drift')
                        continue
                    folder = root/key; folder.mkdir(exist_ok=True)
                    n = len(list(folder.glob('attempt-*'))) + 1
                    if n > 3: raise ValueError('Launch cap')
                    attempt = folder/f'attempt-{n:03}'; attempt.mkdir()
                    log = (attempt/'log.txt').open('x')
                    proc = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.vision.train_small_scale', '--train', '--worker', key],
                        cwd=prior.ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    jobs.append((proc, log, key, attempt))
                    print('TRAINING_STARTED', key, proc.pid, 'THREADS=4', flush=True)
                for proc, _, key, _ in jobs:
                    proc.wait(timeout=7200)
                    if proc.returncode: raise RuntimeError('Worker failed '+key)
                    c = prior.read(OUT/'training'/key/'completion.json'); prior.verify(c)
                    if c['optimizer_steps'] != 1100: raise ValueError('Endpoint drift')
                    print('TRAINING_VERIFIED', key, flush=True)
            except BaseException:
                error = traceback.format_exc()
                for proc, _, key, attempt in jobs:
                    cleanup(proc)
                    prior.frozen(attempt/'failure.json', dict(cell=key, error=error,
                        process_cleanup_complete=proc.poll() is not None))
                raise
            finally:
                for proc, log, _, _ in jobs: cleanup(proc); log.close()
        deps = [OUT/'training'/k/'completion.json' for k in KEYS] + [Path(__file__).resolve()]
        prior.frozen(root/'completion.json', dict(status='six_units_complete_evaluation_pending',
            wall_seconds=time.monotonic()-started, inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--train', action='store_true'); ap.add_argument('--worker', choices=KEYS)
    ap.add_argument('--then-evaluate', action='store_true'); a = ap.parse_args()
    if a.then_evaluate and (not a.train or a.worker): ap.error('--then-evaluate requires the full --train runner')
    if a.worker and not a.train: ap.error('Explicit --train required')
    if a.worker: worker(a.worker)
    elif a.train:
        run()
        if a.then_evaluate:
            from scripts.vision.evaluate_small_scale import run as evaluate_all
            evaluate_all()
    else: contract(); print('PREFLIGHT_ONLY_NO_TRAINING')
