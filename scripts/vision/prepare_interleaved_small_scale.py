"""Readiness only, with actual-loader identity and targeted regression evidence."""
import subprocess
import sys
from pathlib import Path
from scripts.vision.interleaved_small_scale_control import OUT,KEYS,prior
from scripts.vision.train_interleaved_small_scale import contract
from scripts.vision.prepare_small_scale_training import TESTS as HISTORICAL_TESTS

TESTS=HISTORICAL_TESTS+('tests.test_small_scale_fp_review','tests.test_small_scale_interleave','tests.test_interleaved_small_scale_entry')


def main():
    from scripts.vision.analyze_small_scale_false_positives import run as analyze
    from scripts.vision.build_small_scale_fp_review import DEST
    a=analyze()
    if a['status']!='source_and_visual_accounting_complete' or a['unique_images']!=37:
        raise ValueError('Incomplete source/visual audit')
    _,f,deps,b=contract(); dest=OUT/'entry-ready.json'
    deps.append(DEST/'analysis.json')
    if dest.exists():
        r=prior.read(dest); prior.verify(r); return r
    root=OUT/'entry-preflight'; root.mkdir(exist_ok=True); n=len(list(root.glob('attempt-*')))+1
    if n>3: raise ValueError('Entry preflight attempt cap')
    attempt=root/f'attempt-{n:03}'; attempt.mkdir()
    result=subprocess.run([sys.executable,'-m','unittest',*TESTS,'-v'],cwd=prior.ROOT,capture_output=True,text=True,timeout=180)
    log=attempt/'tests.txt'; log.write_text(result.stdout+'\n'+result.stderr)
    deps += [log,Path(__file__).resolve(),Path(__file__).with_name('evaluate_interleaved_small_scale.py'),
             prior.ROOT/'docs/plans/ml_interleaved_small_scale_control_20260915.md']
    deps += [prior.ROOT/(name.replace('.','/')+'.py') for name in TESTS]
    if result.returncode:
        prior.frozen(attempt/'failure.json',dict(status='regression_failed',inputs={str(log):prior.file_sha256(log)}))
        raise ValueError('Relevant tests failed')
    return prior.frozen(dest,dict(status='ready_for_training_not_started',cells=list(KEYS),
        actual_draws_verified=sum(len(u['actual']) for u in f['units']),exact_image_and_label_tensors_frozen=True,
        optimizer_created=False,backward_executed=False,baseline=b,relevant_regressions_passed=True,
        whole_repository_tests_claimed=False,CPU_policy='Two independent four-thread workers; three sequential seed pairs.',
        inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__=='__main__': print(main()['status'])
