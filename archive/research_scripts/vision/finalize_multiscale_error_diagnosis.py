"""Verify explicit review and publish a bounded diagnostic receipt."""
import subprocess,sys
from pathlib import Path
from scripts.vision.record_multiscale_error_review import DEST,prior,validate
from scripts.vision.evaluate_frozen_multiscale import baseline_verify,OUT

def main():
    ep,rp=DEST/'evidence.json',DEST/'review.json'
    e,r=prior.read(ep),prior.read(rp)
    for x in (e,r):prior.verify(x)
    validate(e,r['decisions'])
    suites=['tests.test_multiscale_error_review','tests.test_frozen_multiscale_evaluation','tests.test_exposure_diagnosis','tests.test_hard_negative_coverage_evaluation']
    t=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True)
    if t.returncode:raise ValueError(t.stdout+t.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline failure')
    paths=[ep,rp,OUT/'evaluation/summary.json',Path(__file__).resolve(),prior.ROOT/'docs/results/ml_multiscale_error_diagnosis_20260911.md']
    paths += [prior.ROOT/(x.replace('.','/')+'.py') for x in suites]
    prior.frozen(DEST/'completion.json',dict(status='diagnosis_complete_with_named_gaps',review_decisions=len(r['decisions']),
        pending_ids=[d['decision_id'] for d in r['decisions'] if d['status']=='pending'],
        selected_candidate=None,training_started=False,threshold_changed=False,pixel_visibility_certified=False,
        regression_output=t.stdout+t.stderr,baseline=baseline,whole_repository_tests_claimed=False,
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('DIAGNOSIS_COMPLETE_WITH_NAMED_GAPS')

if __name__=='__main__':main()
