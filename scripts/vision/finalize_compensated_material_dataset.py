"""Dataset-only acceptance for bounded development; no training or promotion."""
from collections import Counter
from pathlib import Path
import subprocess,sys
from scripts.vision.merge_compensated_material_candidates import OUT,prior
from scripts.vision.validate_compensated_loader_receipts import main as validate_loaders
from scripts.vision.evaluate_scale_endpoints import baseline_verify


def main():
    paths=validate_loaders()+[Path(__file__)]
    files=[OUT/'reviewed-completion.json',OUT/'candidate-export-v1/manifest.json',OUT/'known-source-pose-audit.json',OUT/'resolved-pose-gaps.json']
    reviewed,export,audit,resolved=[prior.read(p) for p in files]
    for r in (reviewed,export,audit,resolved):prior.verify(r)
    paths+=files
    if len(export['members'])!=37 or export['reference_overlap_gaps']:raise ValueError('Dataset size or exclusion gate')
    if len(reviewed['decisions'])!=113 or any(d['review_type']!='AI辅助审核' or not d['reason'] for d in reviewed['decisions']):raise ValueError('Explicit full-label review coverage')
    if sum(len(m['full_truth']['objects']) for m in reviewed['members'])!=113:raise ValueError('Full truth count')
    if resolved['unresolved'] or resolved['pose_overlaps'] or any(x['known_pose_matches'] for x in audit['sources'].values()):raise ValueError('Source pose gap or overlap')
    if len(audit['sources'])!=13:raise ValueError('Source count')
    for group,src in audit['sources'].items():
        cp=Path(src['fresh_source']);c=prior.read(cp);checks=c['collection_checks']
        if (checks['actual_annotation_mode'],checks['label_mode'],checks['hierarchy_mode'])!=('full_2d','visual-instance','top-level-equipment'):raise ValueError('Actual semantics mismatch')
        if checks['world_sha256']!=prior.file_sha256(src['world']):raise ValueError('Actual world identity')
        root=cp.parents[2];review=prior.read(root/'reviewed-completion.json');prior.verify(review)
        s=[x for x in review['sources'] if x['lineage_id']==group]
        if len(s)!=1 or not s[0]['exact_replay'] or s[0]['unboxed_target_count'] or not s[0]['mask_inside_full_boxes']:raise ValueError('Source replay/coverage')
        paths += [cp,Path(src['world']),root/'reviewed-completion.json']
    for m in export['members']:
        for k in ('image','label'):
            file=Path(m[k+'_path'])
            if prior.file_sha256(file)!=m[k+'_sha256']:raise ValueError('Export identity changed')
            paths.append(file)
    groups=Counter(m['pair_id'] for m in export['members'])
    if sorted(groups.values())!=[1]+[3]*12:raise ValueError('Material/common grouping')
    tests=['tests.test_compensated_loader_receipts','tests.test_compensated_material_sequences','tests.test_material_pose_budget_conflict','tests.test_supplemental_pose_sources','tests.test_material_candidate_export','tests.test_designed_material_triplets','tests.test_full_scene_pose_design','tests.test_material_mask_coverage','tests.test_material_expansion_gate','tests.test_material_view_world_drafts','tests.test_material_view_source_review','tests.test_material_view_design']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True)
    if result.returncode:raise ValueError(result.stdout+result.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline changed')
    paths += [prior.ROOT/(x.replace('.','/')+'.py') for x in tests]
    report=prior.ROOT/'docs/results/ml_compensated_material_dataset_20260911.md';paths.append(report)
    prior.frozen(OUT/'dataset-completion.json',dict(status='ready_for_bounded_development_training_not_started',
        dataset_images=37,material_triplets=12,common_compensation_images=1,source_poses=13,reviewed_full_labels=113,
        training_ready=True,training_started=False,training_admitted=False,promotable=False,
        loader_cells=6,actual_loader_exposures=16200,optimizer_created=False,backward_executed=False,
        source_scope='Fresh captures verified directly. Historical metadata used for exclusions; old approval chain is not renewed.',
        historical_admission_chain_repaired=False,whole_old_pool_quality_recertified=False,
        use_scope='Bounded V/VM development experiment only. Shared layout/assets, high-angle and narrow reactor-distance limits. Not independent-scene generalization or formal training admission.',
        baseline=baseline,regression_output=result.stdout+result.stderr,whole_repository_tests_claimed=False,
        report=str(report),inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('DATASET37_READY_BOUNDED_DEVELOPMENT; REVIEW113; LOADERS6; TESTS56; PINNED40; NO_TRAINING')


if __name__=='__main__':main()
