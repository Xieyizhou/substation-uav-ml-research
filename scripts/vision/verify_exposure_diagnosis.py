"""Persist targeted regression and pinned-baseline verification results."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.vision.exposure_protocol import OUT, save, file_sha256


def main():
    suites = ['tests.test_exposure_diagnosis', 'tests.test_visual_bridge_training',
              'tests.test_visual_bridge_supplement', 'tests.test_canonical_gates', 'tests.test_paired_visual_factors']
    tests = subprocess.run([sys.executable, '-m', 'unittest', *suites], cwd=ROOT, capture_output=True, text=True)
    baseline = subprocess.run([sys.executable, 'scripts/vision/verify_experiment_baseline.py'], cwd=ROOT, capture_output=True, text=True)
    diff = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
    record = json.loads(baseline.stdout) if baseline.returncode == 0 else {}
    passed = tests.returncode == baseline.returncode == diff.returncode == 0 and record.get('integrity_passed') and record.get('pinned_files_verified') == 40
    paths = [ROOT/'scripts/vision'/name for name in ('exposure_protocol.py', 'run_exposure_diagnosis.py',
             'exposure_metrics.py', 'evaluate_exposure_diagnosis.py', 'finalize_exposure_diagnosis.py',
             'record_exposure_review_v2.py', 'build_exposure_review.py', 'exposure_lineage.py', 'verify_exposure_diagnosis.py')]
    paths += [ROOT/(suite.replace('.', '/') + '.py') for suite in suites]
    result = save(OUT/'verification.json', {'status': 'passed' if passed else 'failed',
        'test_output': tests.stdout + tests.stderr, 'test_returncode': tests.returncode,
        'baseline': record, 'diff_check_returncode': diff.returncode, 'diff_output': diff.stdout + diff.stderr,
        'inputs': {str(p): file_sha256(p) for p in paths},
        'scope': 'Targeted regressions only; no claim of whole-repository pass.',
        'known_previous_global_failures': 'Previous full run: 876 tests, 12 failures, 1 skipped (module length, legacy Gazebo truth expectations, legacy identity/message assertions). Not rerun or repaired here.'})
    print(result['status'], result['identity'])
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
