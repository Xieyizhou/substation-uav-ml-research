"""Issue a bounded training entry receipt only after real loaders and regressions."""
import subprocess
import sys
from pathlib import Path
from scripts.vision.neutral_gray_control import OUT, KEYS, prior
from scripts.vision.preflight_neutral_gray import preflight
from scripts.vision.train_neutral_gray import contract

TESTS = ('tests.test_cool_light_capture', 'tests.test_neutral_gray_control',
         'tests.test_physical_low_light', 'tests.test_interleaved_light_control',
         'tests.test_closed_budget_training', 'tests.test_closed_budget_evaluation',
         'tests.test_canonical_gates', 'tests.test_canonical_recovery', 'tests.test_canonical_shutdown')


def main():
    preflight()
    _, f, deps, baseline = contract()
    dest = OUT/'entry-ready.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    root = OUT/'entry-preflight'
    root.mkdir(exist_ok=True)
    n = len(list(root.glob('attempt-*')))+1
    if n > 3:
        raise ValueError('Preflight attempt cap')
    attempt = root/f'attempt-{n:03}'
    attempt.mkdir()
    proc = subprocess.run([sys.executable, '-m', 'unittest', *TESTS, '-v'], cwd=prior.ROOT, capture_output=True, text=True, timeout=180)
    log = attempt/'tests.txt'
    log.write_text(proc.stdout+'\n'+proc.stderr)
    deps.extend([log, Path(__file__).resolve()])
    deps.extend(prior.ROOT/(name.replace('.', '/')+'.py') for name in TESTS)
    if proc.returncode:
        prior.frozen(attempt/'failure.json', dict(status='regression_failed', returncode=proc.returncode, inputs={str(log):prior.file_sha256(log)}))
        raise ValueError('Relevant regression failed')
    return prior.frozen(dest, dict(status='ready_for_training_not_started', cells=list(KEYS),
        actual_draws_verified=sum(len(u['actual']) for u in f['units']),
        exact_image_and_label_tensors_frozen=True, optimizer_created=False, backward_executed=False,
        baseline=baseline, relevant_regressions_passed=True, whole_repository_tests_claimed=False,
        CPU_policy='Two independent workers, four Torch CPU threads each; third seed after first two. Historical accelerated execution retained.',
        inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__ == '__main__': print(main()['status'])
