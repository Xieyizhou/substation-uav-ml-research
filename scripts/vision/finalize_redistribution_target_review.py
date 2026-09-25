"""Publish a blocked review outcome, not readiness."""
import subprocess,sys
from pathlib import Path
from scripts.vision.audit_redistribution_targets import OUT,ROOT,read,verify,frozen,file_sha256

def main():
    r=read(OUT/'review.json');verify(r)
    if len(r['decisions'])!=12 or len(r['insufficient_content_events'])!=8 or len(r['attribution_pending_events'])!=4:raise ValueError('Wrong scope')
    for d in r['decisions']:
        if d['pixel_visibility_certified'] is not False or d['training_approved'] is not False:raise ValueError('Unjustified certification')
    tests=['tests.test_redistribution_target_attribution','tests.test_redistribution_member_review','tests.test_hold_error_review']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if result.returncode:raise ValueError(result.stderr)
    paths=[OUT/'review.json',Path(__file__),ROOT/'docs/results/ml_redistribution_target_review_20260909.md']
    paths += [ROOT/(t.replace('.','/')+'.py') for t in tests]
    frozen(OUT/'completion.json',dict(status='review_complete_quality_gate_still_blocked',new_replays=0,training_started=False,
        readiness_issued=False,labels_modified=False,regression_output=result.stderr,whole_repository_tested=False,
        inputs={str(x):file_sha256(x) for x in paths}))
    print('REVIEW_COMPLETE_8_INSUFFICIENT_4_PENDING; NO_TRAINING')
if __name__=='__main__':main()
