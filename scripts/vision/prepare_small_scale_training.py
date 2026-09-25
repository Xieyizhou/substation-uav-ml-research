"""Readiness receipt after actual loaders, targeted tests and pinned baseline."""
import subprocess
import sys
from pathlib import Path
from scripts.vision.small_scale_control import OUT, KEYS, prior
from scripts.vision.train_small_scale import contract

TESTS = ('tests.test_small_scale_schedule', 'tests.test_small_scale_review_gate', 'tests.test_small_scale_entry',
         'tests.test_cool_light_capture', 'tests.test_neutral_gray_control',
         'tests.test_closed_budget_training', 'tests.test_closed_budget_evaluation',
         'tests.test_canonical_gates', 'tests.test_canonical_recovery', 'tests.test_canonical_shutdown')


def main():
    _, f, deps, baseline = contract()
    dest = OUT/'entry-ready.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    root = OUT/'entry-preflight'; root.mkdir(exist_ok=True)
    n = len(list(root.glob('attempt-*'))) + 1
    if n > 3: raise ValueError('Preflight attempt cap')
    attempt = root/f'attempt-{n:03}'; attempt.mkdir()
    result = subprocess.run([sys.executable, '-m', 'unittest', *TESTS, '-v'],
        cwd=prior.ROOT, capture_output=True, text=True, timeout=180)
    log = attempt/'tests.txt'; log.write_text(result.stdout+'\n'+result.stderr)
    deps += [log, Path(__file__).resolve(), Path(__file__).with_name('evaluate_small_scale.py'),
             prior.ROOT/'docs/plans/ml_small_scale_material_control_20260914.md']
    deps += [prior.ROOT/(name.replace('.', '/')+'.py') for name in TESTS]
    if result.returncode:
        prior.frozen(attempt/'failure.json', dict(status='regression_failed',
            inputs={str(log):prior.file_sha256(log)}))
        raise ValueError('Targeted regression failed')
    return prior.frozen(dest, dict(status='ready_for_training_not_started', cells=list(KEYS),
        actual_draws_verified=sum(len(u['actual']) for u in f['units']),
        exact_image_and_label_tensors_frozen=True, optimizer_created=False, backward_executed=False,
        baseline=baseline, relevant_regressions_passed=True, whole_repository_tests_claimed=False,
        CPU_policy='Three sequential seed pairs; each pair SR1100/SM1100 concurrent, four threads per worker.',
        inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__ == '__main__': print(main()['status'])
