"""Seal partial proposals with explicit replay blockers, not data admission."""
import argparse
from pathlib import Path
import subprocess,sys
from scripts.vision.supervision_risk_revision import OUT,SOURCE,ROOT,read,verify,verify_tree,frozen,file_sha256
from scripts.vision.supervision_risk_proposal import validate
from scripts.vision.exposure_order_retention import baseline_verify

def finish():
    if (OUT/'completion.json').exists():verify_tree(OUT/'completion.json');print('REUSE_VALID_BLOCKED_PROPOSAL');return
    for name in ('protocol.json','review.json','impact.json','reused-evidence/T32/receipt.json'):verify(read(OUT/name))
    p=read(OUT/'protocol.json');r=read(OUT/'review.json');validate(p,r['decisions'])
    attempts=list((OUT/'replay').glob('*/attempt-*/receipt.json'))
    if len(attempts)!=6:raise ValueError('Unexpected attempt count')
    for path in attempts:
        rr=read(path);verify(rr)
        if not rr['process_cleanup_complete'] or rr['status']!='technical_failure':raise ValueError('Unexpected replay outcome')
        if 'Operation not permitted' not in (path.parent/'simulator.log').read_text():raise ValueError('Unexplained technical blocker')
    header=Path('/opt/homebrew/opt/gz-rendering8/include/gz/rendering8/gz/rendering/BoundingBoxCamera.hh')
    source=ROOT/'src/vision/collection/gazebo_truth.py'
    frozen(OUT/'semantics.json',dict(status='semantics_clarified_no_labels_changed',
        full_2d='complete 2D box for occluded objects, not visible-only pixel bounds',
        old_not_truncated='converter did not clamp incoming box; not physical completeness certification',
        all_seven_raw_values_reproduced=True,historical_source_hash_not_available=True,
        sidecar_proposals=[dict(event_id=e,historical_truncation_status='not_truncated',suggested_interpretation='converter_did_not_clip',physical_truncation_certified=False) for e in ('T04','T08','T26','T30','T32')],
        inputs={str(x):file_sha256(x) for x in (header,source,OUT/'protocol.json',Path(__file__))}))
    modules=['tests.test_supervision_risk_proposal','tests.test_instance_visibility_diagnosis','tests.test_structure_fit_integrity','tests.test_structure_fit','tests.test_structure_exposure']
    run=subprocess.run([sys.executable,'-m','unittest',*modules],cwd=ROOT,capture_output=True,text=True,timeout=60)
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if run.returncode or 'Ran 38 tests' not in run.stderr or not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError(run.stdout+run.stderr)
    verify(read(SOURCE/'completion.json'))
    frozen(OUT/'verification.json',dict(status='related_tests_and_pinned40_passed',tests=modules,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr,
        related_tests=38,baseline=baseline,full_repository_tests_run=False,prior_diagnosis_files_unchanged=True,
        inputs={str(x):file_sha256(x) for x in [ROOT/(m.replace('.','/')+'.py') for m in modules]+[SOURCE/'completion.json',Path(__file__)]}))
    inputs={str(x):file_sha256(x) for x in OUT.rglob('*') if x.is_file()}
    for path in [ROOT/'docs/results/ml_supervision_risk_revision_20260909.md',Path(__file__),ROOT/'scripts/vision/supervision_risk_proposal.py',ROOT/'scripts/vision/reuse_supervision_risk_evidence.py']:
        inputs[str(path)]=file_sha256(path)
    frozen(OUT/'completion.json',dict(status='evidence_blocked_no_revision_authorized',proposal_workflow_completed=True,all_issues_resolved=False,
        metadata_only_events=['T04','T26','T32'],whole_image_hold_proposed=['T29','T33'],pending_events=['T30','T08'],
        labels_modified=False,training_started=False,new_training_pool_exported=False,training_frames_collected=0,
        replay_blocker='Gazebo local sockets Operation not permitted; pilots exhausted 3 attempts each, no expansion',
        next_action='Resolve execution permissions and explicitly confirm a new bounded pilot budget before any additional replay.',inputs=inputs))
    print('EVIDENCE_BLOCKED_NO_REVISION_AUTHORIZED; TESTS38_PASS; PINNED40_PASS')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--finalize',action='store_true');a=ap.parse_args()
    if a.finalize:finish()
    else:print('PREFLIGHT_ONLY_NO_TRAINING')
