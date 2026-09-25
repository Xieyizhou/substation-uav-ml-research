"""A single prespecified LR change; immutable brightness and exposure replay."""
import argparse
import csv
import fcntl
import math
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

from scripts.vision import train_brightness_transfer as prior
from scripts.vision import train_whole_image_hold as adapter
from scripts.vision.brightness_transfer_runtime import make_dataset, check_log, preflight
from scripts.vision.order_retention_runtime import overrides as original_overrides
from scripts.vision.prepare_whole_image_hold import baseline_verify

SOURCE, ROOT, KEYS = prior.OUT, prior.ROOT, prior.KEYS
OUT = SOURCE / 'brightness-lr-retention-v1'
read, verify, frozen, file_sha256 = prior.read, prior.verify, prior.frozen, prior.file_sha256
LR = .0005
TESTS = ['tests.test_brightness_lr_retention', 'tests.test_brightness_transfer',
         'tests.test_brightness_audit', 'tests.test_whole_image_hold_train']


def overrides(seed):
    return {**original_overrides(seed), 'lr0': LR}


def check_pair(p, old):
    for field in ('pool_rows', 'names', 'schedules', 'brightness_factors', 'listings',
                  'ledger', 'initialization', 'evaluation', 'acceptance_policy', 'retention', 'environment'):
        if p[field] != old[field]:
            raise ValueError('Single-variable control changed: ' + field)
    if p['learning_rate'] != LR or p['reference_learning_rate'] != .001:
        raise ValueError('Unfrozen learning rate')
    for seed in (7, 17, 27):
        a, b = original_overrides(seed), overrides(seed)
        if {k for k in a if a[k] != b[k]} != {'lr0'}:
            raise ValueError('More than learning rate changed')


def check_curve(args, curve, seed, rate):
    expected = {**original_overrides(seed), 'lr0': rate}
    if any(args.get(k) != v for k, v in expected.items()):
        raise ValueError('Actual training configuration drift')
    if len(curve) != 45 or any(not math.isfinite(float(v)) for r in curve
                              for k, v in r.items() if k.startswith('train/')):
        raise ValueError('Missing/nonfinite loss curve')
    for r in curve:
        values = [float(v) for k, v in r.items() if k.startswith('lr/')]
        if len(values) != 3 or any(abs(v-rate) > 1e-12 for v in values):
            raise ValueError('Missing/nonconstant learning-rate curve')


def runtime_check(receipt, seed, rate):
    import yaml
    run = Path(receipt['weights']).parents[1]
    with (run/'results.csv').open() as f:
        curve = list(csv.DictReader(f))
    check_curve(yaml.safe_load((run/'args.yaml').read_text()), curve, seed, rate)


def freeze():
    old = prior.pretrain()
    verify(read(SOURCE/'completion.json'))
    if (OUT/'protocol.json').exists():
        p = read(OUT/'protocol.json'); verify(p); check_pair(p, old); return p
    paths = [SOURCE/'protocol.json', SOURCE/'completion.json', SOURCE/'training-gate.json',
             Path(__file__), ROOT/'scripts/vision/brightness_transfer_runtime.py',
             ROOT/'scripts/vision/order_retention_runtime.py', Path(adapter.__file__),
             ROOT/'docs/brightness-lr-retention-v1.md']
    baselines = {}
    for key in KEYS:
        prior.complete(key, old)
        seed = int(key.split('-')[-1]); cp = SOURCE/'training'/key/'completion.json'
        c = read(cp); runtime_check(c, seed, .001)
        ep = SOURCE/'evaluation'/f'{key}.json'; verify(read(ep))
        baselines[str(seed)] = dict(cell=key, training_receipt=str(cp), evaluation=str(ep))
        paths += [cp, ep, Path(c['weights']), Path(c['exposure_path']),
                  SOURCE/'training'/key/'brightness-receipt.json', SOURCE/'preflight'/f'{seed}.json']
    p = {k: old[k] for k in ('pool_rows', 'names', 'schedules', 'brightness_factors', 'listings',
                             'ledger', 'initialization', 'evaluation', 'acceptance_policy', 'retention', 'environment')}
    p.update(status='frozen_preflight_pending', family='brightness-lr0005-450',
             learning_rate=LR, reference_learning_rate=.001, baselines=baselines,
             unaugmented_context=old['baselines'],
             interpretation='Only lr0 differs from completed brightness control; identical seed-specific members, batches, gains and 450 optimizer steps. Fixed three seeds; no checkpoint selection or sealed scene.',
             prior_quality='Existing risk caps remain; no new semantic admission or visibility certification.',
             inputs={str(x): file_sha256(x) for x in paths})
    OUT.mkdir(exist_ok=True); check_pair(p, old)
    return frozen(OUT/'protocol.json', p)


def validate_preflight(p, seed, u, old_u):
    verify(u); verify(old_u)
    if u['protocol_identity'] != p['identity']:
        raise ValueError('Preflight protocol mismatch')
    for field in ('batches', 'draws', 'logs', 'baseline_tensor_bytes_identical', 'complete_labels_identical'):
        if u[field] != old_u[field]:
            raise ValueError('Actual loader differs from brightness control: ' + field)
    if u['batches'] != 450 or not u['baseline_tensor_bytes_identical'] or not u['complete_labels_identical']:
        raise ValueError('Invalid preflight')
    if any(u[x] is not False for x in ('optimizer_created', 'backward_executed', 'validation_run')):
        raise ValueError('Preflight performed forbidden training')
    for arm in ('noaug', 'brightness'):
        key = f'{arm}-450-{seed}'
        if u['draws'] != p['schedules'][key]: raise ValueError('Wrong actual members')
        check_log(p, key, u['logs'][key])


def precheck():
    p = freeze(); (OUT/'preflight').mkdir(exist_ok=True)
    for seed in (7, 17, 27):
        dest = OUT/'preflight'/f'{seed}.json'
        if not dest.exists():
            attempts = OUT/'preflight'/str(seed); attempts.mkdir(exist_ok=True)
            n = len(list(attempts.glob('attempt-*'))) + 1
            if n > 3: raise ValueError('Preflight attempt budget exhausted')
            attempt = attempts/f'attempt-{n:03}'; attempt.mkdir()
            try:
                result = preflight(p, seed)
                frozen(dest, dict(**result, protocol_identity=p['identity'],
                    inputs={str(x):file_sha256(x) for x in [OUT/'protocol.json', SOURCE/'preflight'/f'{seed}.json', Path(__file__)]}))
            except BaseException:
                frozen(attempt/'failure.json', dict(error=traceback.format_exc(), child_processes_started=0)); raise
        validate_preflight(p, seed, read(dest), read(SOURCE/'preflight'/f'{seed}.json'))
        print('EXACT_BRIGHTNESS_LOADER_PASS', seed, flush=True)
    dest = OUT/'ready.json'
    if not dest.exists():
        t = subprocess.run([sys.executable, '-m', 'unittest', *TESTS], cwd=ROOT, capture_output=True, text=True, timeout=60)
        if t.returncode: raise ValueError(t.stderr)
        baseline = baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
        if not baseline['integrity_passed'] or baseline['pinned_files_verified'] != 40: raise ValueError('Pinned baseline failure')
        paths = [OUT/'protocol.json', Path(__file__)] + [OUT/'preflight'/f'{s}.json' for s in (7,17,27)]
        paths += [ROOT/(x.replace('.', '/')+'.py') for x in TESTS]
        frozen(dest, dict(status='ready_for_training_not_started', cells=list(KEYS), baseline=baseline,
               regression_output=t.stderr, whole_repository_tested=False, inputs={str(x):file_sha256(x) for x in paths}))
    ready(); print('READY_LR_ONLY_CONTROL', flush=True)


def ready():
    old = prior.pretrain(); p = read(OUT/'protocol.json'); verify(p); check_pair(p, old)
    verify(read(SOURCE/'completion.json')); r = read(OUT/'ready.json'); verify(r)
    if r['status'] != 'ready_for_training_not_started' or r['cells'] != list(KEYS): raise ValueError('Not ready')
    for key in KEYS:
        prior.complete(key, old); seed = int(key.split('-')[-1])
        runtime_check(read(p['baselines'][str(seed)]['training_receipt']), seed, .001)
        validate_preflight(p, seed, read(OUT/'preflight'/f'{seed}.json'), read(SOURCE/'preflight'/f'{seed}.json'))
    return p


def complete(key, p):
    cp = OUT/'training'/key/'completion.json'; adapter.complete(cp, p, key)
    b = read(OUT/'training'/key/'brightness-receipt.json'); verify(b)
    check_log(p, key, b['actual_brightness'])
    if b['actual_brightness'] != read(OUT/'preflight'/f"{key.split('-')[-1]}.json")['logs'][key]:
        raise ValueError('Actual brightness tensors changed')
    runtime_check(read(cp), int(key.split('-')[-1]), LR)


def worker(key):
    p = ready(); actual = []
    # Process-local adapter injection only; historical source files and defaults remain unchanged.
    adapter.OUT = OUT; adapter.KEYS = KEYS; adapter.ready = lambda: p
    adapter.overrides = overrides
    adapter.make_dataset = lambda p, k: make_dataset(p, k, actual)
    (OUT/'export').mkdir(exist_ok=True); adapter.worker(key)
    check_log(p, key, actual)
    pre = OUT/'preflight'/f"{key.split('-')[-1]}.json"
    if actual != read(pre)['logs'][key]: raise ValueError('Actual brightness drift')
    cp = OUT/'training'/key/'completion.json'; runtime_check(read(cp), int(key.split('-')[-1]), LR)
    frozen(OUT/'training'/key/'brightness-receipt.json', dict(status='complete_brightness_and_lr_verified',
           actual_brightness=actual, learning_rate=LR,
           inputs={str(x):file_sha256(x) for x in [cp, pre, OUT/'ready.json', Path(__file__)]}))
    complete(key, p)


def train():
    p = ready()
    with (OUT/'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for key in KEYS:
            if (OUT/'training'/key/'brightness-receipt.json').exists(): complete(key, p); continue
            if (OUT/'training'/key/'completion.json').exists(): raise ValueError('Unbound completed training; investigation required')
            root = OUT/'workers'/key; root.mkdir(parents=True, exist_ok=True)
            n = len(list(root.glob('attempt-*'))) + 1
            if n > 3: raise ValueError('Training attempt budget exhausted')
            attempt = root/f'attempt-{n:03}'; attempt.mkdir(); proc = None
            try:
                with (attempt/'log.txt').open('x') as log:
                    proc = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.vision.brightness_lr_retention', '--train', '--worker', key],
                                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    frozen(attempt/'launch.json', dict(pid=proc.pid, cell=key, inputs={str(OUT/'ready.json'):file_sha256(OUT/'ready.json')}))
                    proc.wait(timeout=21600)
                    if proc.returncode: raise RuntimeError('Training worker failed: '+key)
                complete(key, p)
                frozen(attempt/'completion.json', dict(status='complete', process_cleanup_confirmed=True,
                       inputs={str(attempt/'log.txt'):file_sha256(attempt/'log.txt')}))
            except BaseException:
                if proc is not None and proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=10)
                frozen(attempt/'failure.json', dict(error=traceback.format_exc(), process_cleanup_confirmed=proc is None or proc.poll() is not None)); raise
            print('TRAINED_LR0005', key, flush=True)
        dest = OUT/'training/completion.json'
        if dest.exists(): verify(read(dest))
        else: frozen(dest, dict(status='three_trainings_complete_evaluation_pending', inputs={str(OUT/'training'/k/'brightness-receipt.json'):file_sha256(OUT/'training'/k/'brightness-receipt.json') for k in KEYS}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group()
    g.add_argument('--freeze', action='store_true'); g.add_argument('--preflight', action='store_true'); g.add_argument('--train', action='store_true')
    ap.add_argument('--worker', choices=KEYS); a = ap.parse_args()
    if a.worker and not a.train: ap.error('Worker requires explicit --train')
    if a.freeze: freeze()
    elif a.preflight: precheck()
    elif a.train: worker(a.worker) if a.worker else train()
    else: print('READ_ONLY_NO_TRAINING')
