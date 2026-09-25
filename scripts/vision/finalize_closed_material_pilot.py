"""Close only the technical pilot; missing semantic decisions prohibit readiness."""
from pathlib import Path
import subprocess,sys
from scripts.vision.closed_body_material_pilot import OUT,ROOT,read,verify,frozen,file_sha256
from scripts.vision.test_body_material_applicability import baseline_verify


def main():
    paths=[OUT/x for x in ('protocol.json','render-summary.json','review-evidence.json')]
    for p in paths:verify(read(p))
    summary=read(OUT/'render-summary.json');queue=read(OUT/'review-evidence.json')
    if len(summary['units'])!=12 or len(queue['events'])!=46:raise ValueError('Incomplete pilot')
    for u in summary['units']:
        rp=Path(u['receipt']);r=read(rp);verify(r);paths.append(rp)
        expected='original_pixel_evidence_certified' if u['variant']=='control' else 'material_render_passed_full_label_review_pending'
        if r['status']!=expected or not r['process_cleanup_complete']:raise ValueError('Technical gate failed')
    dest=OUT/'technical-completion.json'
    if dest.exists():verify(read(dest));print('VALID_TECHNICAL_PILOT_REUSED_TRAINING_HELD');return
    tests=['tests.test_closed_body_material_pilot','tests.test_body_visibility_replay','tests.test_instance_visibility_diagnosis',
           'tests.test_brightness_lr_retention','tests.test_condition_target_review']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if result.returncode:raise ValueError(result.stderr)
    b=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline check failed')
    paths += [Path(__file__),ROOT/'docs/results/ml_closed_body_material_pilot_20260910.md']
    paths += [ROOT/Path(t.replace('.','/')+'.py') for t in tests]
    frozen(dest,dict(status='pilot_technical_passed_full_label_review_required',original_controls=4,new_material_images=8,
        full_label_reviews_pending=sum(x['decision'] is None for x in queue['events']),
        training_ready=False,training_started=False,final_training_schedule_frozen=False,automatic_approvals_created=0,
        baseline=b,regression=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,whole_repository_tested=False),
        inputs={str(x):file_sha256(x) for x in paths}))
    print(result.stderr);print('TECHNICAL_PILOT_COMPLETE; 46_LABEL_REVIEWS_PENDING; TRAINING_NOT_STARTED')


if __name__=='__main__':main()
