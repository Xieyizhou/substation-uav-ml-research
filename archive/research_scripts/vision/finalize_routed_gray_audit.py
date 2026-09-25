"""Close the failed grayscale experiment from explicit, hash-bound reviews."""
import subprocess
import sys
from pathlib import Path
from scripts.vision.record_routed_gray_results_review import OUT, TRAIN, checked, validate_positive, validate_review
from scripts.vision.verify_experiment_baseline import verify, ROOT
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run():
    paths=[OUT/'evidence.json',OUT/'review.json',OUT/'material-evidence.json',TRAIN/'evaluation-v1/error-review-v1/evidence.json',TRAIN/'evaluation-v1/summary.json']
    e,r,m,n,s=map(checked,paths)
    validate_positive(e,r['positive_decisions']);validate_positive(m,r['material_decisions']);validate_review(n,r['negative_decisions'])
    if any(s['matching_conflicts'].values()):raise ValueError('Matching conflict')
    modules=('tests.test_contrast_cpu4_review','tests.test_routed_gray_transfer','tests.test_routed_gray_review_gate','tests.test_reviewed_order_evaluation','tests.test_reviewed_order_audit')
    t=subprocess.run([sys.executable,'-m','unittest',*modules,'-q'],capture_output=True,text=True,timeout=180)
    if t.returncode:raise ValueError(t.stdout+t.stderr)
    contract=ROOT/'config/perception/visual_experiment_baseline_v1.json';integrity=verify(contract)
    if not integrity['integrity_passed'] or integrity['pinned_files_verified']!=40:raise ValueError('Fixed40 failed')
    paths += [contract,Path(__file__),ROOT/'scripts/vision/record_routed_gray_results_review.py']
    return write_record(OUT/'completion.json',dict(status='review_complete_not_candidate_passed',gates=e['gates'],counts=r['counts'],test_output=t.stdout+t.stderr,integrity=integrity,
        conclusion='Adding whole-frame grayscale at routed material contrast positions fails development gates: material recall declines across all seeds while negative FPR increases. Clear bodies and occluded fragments both lose hits. This does not establish color-only recognition or a unique cause.',
        next_direction='Test a geometry-preserving horizontal-reflection augmentation with complete box reflection against the ungrayscaled routed control; not another grayscale amplitude sweep. Hypothesis of orientation/context sensitivity remains unproven.',
        training_admitted=False,promotable=False,selected_candidate=None,inputs={str(p.resolve()):file_sha256(p) for p in paths}))

if __name__=='__main__':print(run()['status'])
