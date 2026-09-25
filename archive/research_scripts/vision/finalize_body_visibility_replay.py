"""Explicit image observations bound to the two inspected attempts; no admission."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
from scripts.vision.run_body_visibility_intervention import OUT,CONTROL,ROOT,read,verify,frozen,file_sha256
from scripts.vision.test_body_material_applicability import baseline_verify

EXPECTED=('d1556ce7531a431a138f0abfd03b1d667feee3b752ec9752067b39eda93c7c41',
          '7f01fa15ee5dbf65de58137353665465ac3b1759c48c4b9a76e84d0a16cce773')


def main():
    cp=CONTROL/'replay/T020/attempt-01/receipt.json';dp=OUT/'replay/T020/attempt-01/receipt.json'
    paths=[CONTROL/'protocol.json',CONTROL/'control-completion.json',OUT/'protocol.json',OUT/'render-completion.json',cp,dp]
    for p in paths:verify(read(p))
    c,d=read(cp),read(dp)
    if (c['identity'],d['identity'])!=EXPECTED:raise ValueError('Observations belong to different inspected attempts')
    if c['status']!='original_pixel_evidence_certified' or d['status']!='diagnostic_render_stable_pending_visual_review':raise ValueError('Invalid replay status')
    if not all(x['process_cleanup_complete'] for x in (c,d)):raise ValueError('Incomplete cleanup')
    dest=CONTROL/'completion.json'
    if dest.exists():verify(read(dest));verify(read(CONTROL/'visual-review.json'));print('VALID_REVIEW_COMPLETION_REUSED');return
    folder=dp.parent
    images=[folder/f for f in ('stable-1-original-crop.png','stable-1-crop.png','stable-1-rgb.png')]
    images += [cp.parent/'first-stable-window/frame-1-rgb.png',cp.parent/'first-stable-window/1-T020-crop.png']
    review_path=CONTROL/'visual-review.json'
    if not review_path.exists():
        frozen(review_path,dict(status='diagnostic_observation_recorded_not_training_admission',review_nature='AI-assisted',
            reviewed_at=datetime.now(timezone.utc).isoformat(),review_id='T020-body-visual-ablation',
            control_receipt_identity=c['identity'],diagnostic_receipt_identity=d['identity'],
            reason='Inspected both full frames and target crops. Original target shows broad opaque blue-green box face and black base. Removing only body visual reveals three distinguishable pale upright cylindrical columns on the retained base, with previously occluded building/ground behind. Supports shell occlusion for this frame; does not separately certify all six cylinders or any model improvement.',
            individual_six_component_visibility='unknown_not_all_separately_resolved',
            modified_frame_original_pixel_certified=False,training_admission_changed=False,
            inputs={str(p):file_sha256(p) for p in [cp,dp,Path(__file__),*images]}))
    else:verify(read(review_path))
    tests=['tests.test_body_visibility_replay','tests.test_body_material_applicability','tests.test_instance_visibility_diagnosis','tests.test_condition_target_review']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if result.returncode:raise ValueError(result.stderr)
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline failed')
    paths += [review_path,Path(__file__),ROOT/'docs/results/ml_asset_visibility_replay_20260910.md',*images]
    paths += [ROOT/Path(t.replace('.','/')+'.py') for t in tests]
    frozen(dest,dict(status='single_frame_shell_occlusion_rendering_diagnosis_complete',
        source_frames=1,technical_attempts=dict(control=1,body_visual_removal=1),
        control_passed=True,diagnostic_review_recorded=True,training_started=False,training_ready=False,
        model_improvement_tested=False,historical_assets_or_labels_modified=False,
        next_step='Confirm intended capacitor asset form before freezing any independent asset revision; diagnostic body removal is not an approved replacement asset.',
        baseline=baseline,regression=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,whole_repository_tested=False),
        inputs={str(p):file_sha256(p) for p in paths}))
    print(result.stderr);print('RENDERING_DIAGNOSIS_COMPLETE; MODEL_NOT_TRAINED; BASELINE_40_PASSED')


if __name__=='__main__':main()
