"""Close a diagnostic comparison without modifying labels or training gates."""
import subprocess,sys
from pathlib import Path
from scripts.vision.run_dual_box_diagnosis import OUT,SOURCE,ROOT,read,verify,frozen,file_sha256
from scripts.vision.test_body_material_applicability import baseline_verify


def main():
    cp=OUT/'render-completion.json';verify(read(cp));verify(read(OUT/'protocol.json'));verify(read(SOURCE/'protocol.json'))
    receipts=[Path(p) for p in read(cp)['inputs'] if Path(p).name=='receipt.json']
    for p in receipts:verify(read(p))
    if len(receipts)!=1:raise ValueError('Report expects the single observed attempt')
    r=read(receipts[0])
    if r['identity']!='4794ef9f3e0f5c890a78ab22bfd756e2dfe857614315d9aafd1691f84e9fb7c6':raise ValueError('Different diagnostic attempt')
    if r['status']!='dual_mode_diagnostic_complete' or not r['process_cleanup_complete']:raise ValueError('Diagnostic gate failed')
    comparisons=r['comparisons']
    if len(comparisons)!=3 or any(128 in x['full_labels'] or x['visible_boxes'].get('128',{}).get('half_open')!=x['mask_bbox_xyxy'] for x in comparisons):raise ValueError('Report result differs from evidence')
    dest=OUT/'completion.json'
    if dest.exists():verify(read(dest));print('VALID_DUAL_MODE_COMPLETION_REUSED');return
    tests=['tests.test_dual_box_diagnosis','tests.test_instance_visibility_diagnosis','tests.test_edge_box_trace','tests.test_closed_material_review']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if result.returncode:raise ValueError(result.stderr)
    b=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline failed')
    paths=[cp,OUT/'protocol.json',SOURCE/'protocol.json',*receipts,Path(__file__),ROOT/'docs/results/ml_dual_box_diagnosis_20260910.md']
    paths += [ROOT/Path(t.replace('.','/')+'.py') for t in tests]
    frozen(dest,dict(status='dual_mode_comparison_complete_full_mode_specific_processing_suspected',
        stable_comparisons=comparisons,training_ready=False,training_started=False,historical_labels_changed=False,
        production_annotation_mode_changed=False,renderer_modified=False,
        established='In aligned same-pose output, visible_2d contains instance 128 with mask-consistent bounds while original full_2d omits it.',
        unresolved='Exact internal full_2d runtime branch and broader impact; edge-fragment supervision policy remains undecided.',
        next_step='Isolated minimal reproduction/instrumentation of full_2d projection and clipping, no production mode switch.',
        baseline=b,regression=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,whole_repository_tested=False),
        inputs={str(p):file_sha256(p) for p in paths}))
    print(result.stderr);print('DUAL_MODE_DIAGNOSIS_COMPLETE; TRAINING_HELD; BASELINE_40_PASSED')


if __name__=='__main__':main()
