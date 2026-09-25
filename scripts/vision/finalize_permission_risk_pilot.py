"""Merge explicit evidence supplements; freeze a proposal, never a training pool."""
import argparse
from pathlib import Path
import subprocess,sys
from scripts.vision.resume_supervision_risk_pilot import OUT,PRIOR,ROOT,read,verify,verify_tree,frozen,file_sha256
from scripts.vision.review_permission_risk_pilot import validate_alignment
from scripts.vision.supervision_risk_proposal import validate
from scripts.vision.exposure_order_retention import baseline_verify

def finish():
    cp=OUT/'completion.json'
    if cp.exists():verify_tree(cp);print('REUSE_VALID_PROPOSAL_NOT_APPLIED');return
    for path in (OUT/'protocol.json',OUT/'review.json',PRIOR/'protocol.json',PRIOR/'review.json',PRIOR/'impact.json',PRIOR/'completion.json'):verify(read(path))
    p=read(OUT/'protocol.json');review=read(OUT/'review.json');validate(p,review['decisions'])
    receipts={}
    for f in p['frames']:
        path=OUT/'replay'/f['event_id']/'attempt-01/receipt.json';r=read(path);verify(r);validate_alignment(r,f['event_id']);receipts[str(path)]=file_sha256(path)
    if len(list((OUT/'replay').glob('*/attempt-*')))!=2 or (OUT/'replay.lock').exists():raise ValueError('Unexpected attempts or live lock')
    old=read(PRIOR/'review.json');replacements={r['event_id']:r for r in review['decisions']}
    merged=[replacements.get(d['event_id'],d) for d in old['decisions']];validate(read(PRIOR/'protocol.json'),merged)
    held=sorted(d['event_id'] for d in merged if d['proposal']=='hold_whole_image_proposed')
    if held!=['T08','T29','T30','T33']:raise ValueError('Unexpected proposal scope')
    inputs={str(x):file_sha256(x) for x in (OUT/'review.json',PRIOR/'review.json',PRIOR/'impact.json',Path(__file__))};inputs.update(receipts)
    frozen(OUT/'updated-proposal.json',dict(status='revision_proposal_ready_not_applied',decisions=merged,
        supersedes_pending_decisions_only=['T30','T08'],historical_records_unchanged=True,
        hypothetical_impact=read(PRIOR/'impact.json')['scenarios']['all_four_content_risks'],
        closure_limit=read(PRIOR/'impact.json')['scope_limit'],training_pool_exported=False,inputs=inputs))
    tests=['tests.test_permission_risk_pilot','tests.test_supervision_risk_proposal','tests.test_instance_visibility_diagnosis','tests.test_structure_fit_integrity','tests.test_structure_fit','tests.test_structure_exposure']
    r=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if r.returncode or 'Ran 45 tests' not in r.stderr or not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError(r.stdout+r.stderr)
    paths=[ROOT/(t.replace('.','/')+'.py') for t in tests]+[Path(__file__),ROOT/'config/perception/visual_experiment_baseline_v1.json']
    frozen(OUT/'verification.json',dict(status='related_tests_and_pinned40_passed',tests=tests,stdout=r.stdout,stderr=r.stderr,returncode=r.returncode,
        tests_passed=45,baseline=baseline,full_repository_tests_run=False,inputs={str(x):file_sha256(x) for x in paths}))
    inputs={str(x):file_sha256(x) for x in OUT.rglob('*') if x.is_file()}
    for path in (ROOT/'docs/results/ml_supervision_risk_permission_replay_20260909.md',Path(__file__),ROOT/'scripts/vision/review_permission_risk_pilot.py'):
        inputs[str(path)]=file_sha256(path)
    frozen(cp,dict(status='revision_proposal_ready_not_applied',all_issues_resolved=False,pilots_pixel_certified=['T30','T08'],
        proposed_whole_image_hold=held,metadata_only=['T04','T26','T32'],
        labels_modified=False,training_started=False,training_pool_exported=False,new_training_frames=0,
        previous_six_failed_attempts_preserved=True,new_attempts_used=2,new_budget_per_pilot=3,inputs=inputs))
    print('REVISION_PROPOSAL_READY_NOT_APPLIED; TESTS45_PASS; PINNED40_PASS')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--finalize',action='store_true');args=ap.parse_args()
    if args.finalize:finish()
    else:print('PREFLIGHT_ONLY_NO_TRAINING')
