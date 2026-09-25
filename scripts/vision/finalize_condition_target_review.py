"""Read explicit decisions and close the audit with gaps; never train or admit."""
import subprocess
import sys
from pathlib import Path
from scripts.vision.record_condition_target_review import OUT, ROOT, read, verify, frozen, file_sha256, validate, summarize
from scripts.vision.condition_transfer_design import baseline_verify

TESTS = ['tests.test_condition_target_review','tests.test_condition_transfer_design',
         'tests.test_switchgear_condition_review','tests.test_brightness_lr_retention',
         'tests.test_brightness_lr_audit','tests.test_brightness_audit',
         'tests.test_brightness_transfer','tests.test_whole_image_hold_train']
REPORT = ROOT/'docs/results/ml_condition_target_review_20260910.md'


def main():
    dest=OUT/'completion.json'
    e=read(OUT/'evidence.json');r=read(OUT/'review.json')
    verify(e);verify(r);validate(e,r['decisions'])
    if summarize(e,r['decisions'])!=r['summary']:raise ValueError('Summary differs from decisions')
    if dest.exists():verify(read(dest));print('REUSED_VALID_REVIEW_COMPLETION_TRAINING_HELD');return
    result=subprocess.run([sys.executable,'-m','unittest',*TESTS],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if result.returncode:raise ValueError(result.stdout+result.stderr)
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline mismatch')
    paths=[OUT/'evidence.json',OUT/'review.json',REPORT,Path(__file__),
           ROOT/'scripts/vision/record_condition_target_review.py',
           *[ROOT/Path(t.replace('.','/')+'.py') for t in TESTS]]
    frozen(dest,dict(status='condition_review_complete_with_gaps_panel_plan_scope_revision_required',
        target_counts=dict(training=49,development=148,decisions=len(r['decisions']),held=len(r['summary']['held_review_ids'])),
        held_review_ids=r['summary']['held_review_ids'],training_ready=False,training_started=False,
        replay_started=False,new_inference_started=False,original_training_admission_changed=False,
        blocking_gaps=['28 target conditions remain unknown, not automatically accepted.',
            'Per-member training source/component replay eligibility is not completed.',
            'Saved development capacitor asset lacks front_panel; the two-class panel-only mechanism is not applicable as written.'],
        next_priority='Audit visible body/side/rear material-condition coverage and source components before freezing a new intervention; do not execute the old two-class panel plan.',
        baseline=baseline,regression=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,whole_repository_tested=False),
        inputs={str(p):file_sha256(p) for p in paths}))
    print(result.stderr);print('REVIEW_COMPLETE_WITH_GAPS; TRAINING_NOT_STARTED; BASELINE_40_VERIFIED')


if __name__=='__main__':main()
