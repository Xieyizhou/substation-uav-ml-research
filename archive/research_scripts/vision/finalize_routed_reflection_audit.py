"""Close reflection content review without admitting a candidate."""
import subprocess
import sys
from pathlib import Path
from scripts.vision.routed_reflection_control import OUT as TRAIN, checked
from scripts.vision.record_contrast_cpu4_review import validate_positive
from scripts.vision.audit_reviewed_order_results import validate_review
from scripts.vision.verify_experiment_baseline import verify, ROOT
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run():
    out=TRAIN/'audit-v1'
    paths=[out/'evidence.json',out/'review.json',out/'material-evidence.json',TRAIN/'evaluation-v1/error-review-v1/evidence.json',TRAIN/'evaluation-v1/summary.json',out/'review-author-receipt.json',out/'evidence-build-receipt.json']
    e,r,m,n,s,_,_=map(checked,paths)
    validate_positive(e,r['positive_decisions']);validate_positive(m,r['material_decisions']);validate_review(n,r['negative_decisions'])
    if any(s['matching_conflicts'].values()): raise ValueError('Matching conflict')
    t=subprocess.run([sys.executable,'-m','unittest','tests.test_contrast_cpu4_review','tests.test_routed_reflection','tests.test_routed_reflection_review_gate','tests.test_reviewed_order_evaluation','tests.test_reviewed_order_audit','-q'],capture_output=True,text=True,timeout=180)
    if t.returncode: raise ValueError(t.stdout+t.stderr)
    contract=ROOT/'config/perception/visual_experiment_baseline_v1.json';integrity=verify(contract)
    if not integrity['integrity_passed'] or integrity['pinned_files_verified']!=40: raise ValueError('Fixed40 failed')
    paths += [contract,Path(__file__),ROOT/'scripts/vision/record_routed_reflection_results_review.py']
    return write_record(out/'completion.json',dict(status='review_complete_not_candidate_passed',gates=e['gates'],counts=r['counts'],test_output=t.stdout+t.stderr,integrity=integrity,
        conclusion='Reflection has mixed gains and losses and fails fixed development gates. Clear reactor bodies as well as occluded/truncated instances lose hits; visibility alone does not explain all losses. False detections include gray block surfaces, poles and mixed foreground structures. No unique mechanism established.',
        next_direction='Inspect feasibility of restricting backbone updates under unchanged routed exposures, before freezing any next training protocol. No further augmentation stacking.',
        training_admitted=False,promotable=False,selected_candidate=None,inputs={str(p.resolve()):file_sha256(p) for p in paths}))

if __name__=='__main__':print(run()['status'])
