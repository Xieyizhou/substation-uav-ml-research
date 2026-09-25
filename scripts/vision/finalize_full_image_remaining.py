"""Summarize existing decisions and tests without inventing review approvals."""
import subprocess
import sys
from pathlib import Path
from scripts.vision.capture_full_image_remaining import OUT,ORIGINAL,PRIOR,read,save,file_sha256,verify_tree
from scripts.vision.exact_dedup_semantic_review import ROOT
from scripts.vision.verify_experiment_baseline import verify

def validate_matrix(originals,variants):
    expected={(f'F{i:02}',v) for i in range(1,9) for v in ('steel_original','steel_warm_dim','ochre_original','ochre_warm_dim','sage_original','sage_warm_dim')}
    keys=[(f['original_frame_id'],f['variant']) for f in variants]
    if len(keys)!=len(expected) or set(keys)!=expected:raise ValueError('Missing or duplicate matrix member')
    if len(originals)!=8 or {f['frame_id'] for f in originals}!={f'F{i:02}' for i in range(1,9)}:raise ValueError('Original matrix incomplete')
    if any(f['pair_status']!='aligned' or f['pair_bbox_max_delta']>1 for f in variants):raise ValueError('Pair alignment held')

def main():
    dest=OUT/'matrix-handoff.json'
    if dest.exists():verify_tree(dest);print('VERIFIED_EXISTING');return
    paths=[OUT/'review/decisions.json',OUT/'review/intake-audit.json',OUT/'progress.json',ORIGINAL/'review/decisions.json',ORIGINAL/'review/intake-audit-v2.json',PRIOR/'pilot-handoff.json',PRIOR/'review/decisions.json']
    seen=set()
    for p in paths:verify_tree(p,seen)
    if read(paths[1])['status']!='variant_fingerprint_checks_passed':raise ValueError('Variant audit held')
    originals=read(paths[3])['frames'];variants=read(paths[0])['frames']+read(PRIOR/'review/decisions.json')['frames']
    if len(originals)!=8 or len(variants)!=48:raise ValueError('Incomplete pilot')
    validate_matrix(originals,variants)
    if any(o['review_status']!='visible_content_observed' for f in originals+variants for o in f['objects']):raise ValueError('Review held')
    modules=['tests.test_full_image_remaining','tests.test_full_image_appearance','tests.test_full_image_pilot_review','tests.test_full_image_pose_screen','tests.test_full_image_pose_screen_v2','tests.test_appearance_recovery_world','tests.test_canonical_gates','tests.test_canonical_recovery']
    result=subprocess.run([sys.executable,'-m','unittest',*modules],cwd=ROOT,capture_output=True,text=True,timeout=120)
    if result.returncode:raise ValueError(result.stdout+result.stderr)
    paths.extend(ROOT/(m.replace('.','/')+'.py') for m in modules)
    paths.append(Path(__file__))
    save(dest,dict(status='56_frame_matrix_reviewed_not_training_dataset_frozen',original_frames=8,new_frames=48,total_frames=56,
      reviewed_box_observations=sum(len(f['objects']) for f in originals+variants),source_groups=7,pose_count=8,map_count=1,
      training_started=False,pair_bbox_max_delta=max(f['pair_bbox_max_delta'] for f in variants),
      regression=dict(command=[sys.executable,'-m','unittest',*modules],returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,whole_repository_tested=False),
      baseline=verify(ROOT/'config/perception/visual_experiment_baseline_v1.json'),
      remaining=['Convert reviewed full-image labels and freeze all 56 members with common lineage and role; no automatic training.',
       'Do not count absent switchgear front-panel coverage as resolved.',
       'Freeze complete dataset and controlled three-seed training protocol before training.'],
      inputs={str(p):file_sha256(p) for p in paths}))
    print('MATRIX_HANDOFF',dest,flush=True)

if __name__=='__main__':main()
