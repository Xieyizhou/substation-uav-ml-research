"""Completion consumes explicit error decisions; never creates pass decisions."""
import subprocess,sys
from pathlib import Path
from scripts.vision.train_reviewed_hold import OUT,KEYS,complete,prior
from scripts.vision.reviewed_hold_gate import check
from scripts.vision.test_body_material_applicability import baseline_verify


def validate_errors(e,review):
    expected={}
    for event in e['events']:
        count=len(event['predictions']) if event['kind']=='FP' else len(event['loss']['seeds'])
        for n in range(count):expected[f'{event["event_id"]}-{n}']=event
    ds=review['decisions'];ids=[d['decision_id'] for d in ds]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):raise ValueError('Missing or duplicate error review')
    if review['evidence_identity']!=e['identity']:raise ValueError('Stale error evidence')
    for d in ds:
        event=expected[d['decision_id']]
        if d['page_sha256']!=event['page_sha256'] or prior.file_sha256(event['page'])!=d['page_sha256']:raise ValueError('Stale error page')
        if d['review_nature']!='AI-assisted' or not d['reason'] or not d['reviewed_at']:raise ValueError('Incomplete explicit decision')
        if d['status'] not in ('reviewed','unknown'):raise ValueError('Invalid error review status')
    return all(d['status']=='reviewed' for d in ds)


def main():
    p=check()
    for key in KEYS:complete(key,p)
    sp=OUT/'evaluation/summary.json';ep=OUT/'error-review/evidence.json';rp=OUT/'error-review/decisions.json'
    for path in (sp,ep,rp):prior.verify(prior.read(path))
    s,e,r=map(prior.read,(sp,ep,rp));review_pass=validate_errors(e,r)
    tests=['tests.test_reviewed_hold_control','tests.test_reviewed_hold_gate','tests.test_reviewed_hold_finalization','tests.test_revision_minimax','tests.test_revision_exposure','tests.test_revision_sibling_replay','tests.test_revision_source_review','tests.test_annotation_revision_design','tests.test_depth_clip_acceptance','tests.test_dual_box_diagnosis','tests.test_brightness_lr_retention','tests.test_whole_image_hold_train','tests.test_exposure_diagnosis','tests.test_fixed_budget_diagnosis']
    run=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True,timeout=60)
    if run.returncode:raise ValueError(run.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline integrity failed')
    passed=s['policy_results']['passed'] and review_pass and not s['matching_conflicts']
    paths=[sp,ep,rp,Path(__file__),prior.ROOT/'docs/results/ml_reviewed_hold_control_20260910.md']
    paths += [OUT/'training'/key/'brightness-receipt.json' for key in KEYS]
    paths += [prior.ROOT/(t.replace('.','/')+'.py') for t in tests]
    dest=OUT.parent/'completion.json'
    if dest.exists():prior.verify(prior.read(dest));return
    status='experiment_blocked_unresolved_review_or_matching' if not review_pass or s['matching_conflicts'] else 'experiment_complete_development_candidate' if passed else 'experiment_complete_no_candidate'
    prior.frozen(dest,dict(status=status,
        selected_candidate_family='reviewed_hold_compensation_lr0005' if passed else None,
        retained_seeds=[7,17,27],numerical_pass=s['policy_results']['passed'],error_review_complete=review_pass,matching_conflicts=s['matching_conflicts'],
        baseline=baseline,regression=dict(returncode=run.returncode,output=run.stderr,whole_repository_tested=False),
        training_completed=True,new_training_units=3,resolve_count=0,historical_labels_changed=False,production_deployed=False,
        interpretation='Joint whole-frame hold and exposure redistribution, not a pure missing-label causal effect; T023 lineage concentration remains.',
        inputs={str(path):prior.file_sha256(path) for path in paths}))
    print(run.stderr);print(status,'BASELINE40_PASSED')


if __name__=='__main__':main()
