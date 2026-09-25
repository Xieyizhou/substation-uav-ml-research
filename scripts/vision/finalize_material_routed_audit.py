"""Validate authored decisions and report fixed gates, without auto-approval."""
from pathlib import Path
import subprocess,sys
from scripts.vision.inspect_material_routed_results import OUT,TRAIN,checked
from scripts.vision.record_contrast_cpu4_review import validate_positive
from scripts.vision.audit_reviewed_order_results import validate_review
from scripts.vision.verify_experiment_baseline import verify,ROOT
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run():
    paths=[OUT/'evidence.json',OUT/'review.json',TRAIN/'evaluation-v1/error-review-v1/evidence.json']
    e,r,n=map(checked,paths);validate_positive(e,r['positive_decisions']);validate_review(n,r['negative_decisions'])
    modules=('tests.test_contrast_cpu4_review','tests.test_dose_matched_contrast_control','tests.test_material_routed_contrast_control','tests.test_reviewed_order_evaluation','tests.test_reviewed_order_audit')
    t=subprocess.run([sys.executable,'-m','unittest',*modules,'-q'],capture_output=True,text=True,timeout=180)
    if t.returncode:raise ValueError(t.stdout+t.stderr)
    contract=ROOT/'config/perception/visual_experiment_baseline_v1.json';integrity=verify(contract)
    if not integrity['integrity_passed'] or integrity['pinned_files_verified']!=40:raise ValueError('Fixed40 failed')
    paths += [contract,Path(__file__)]+[ROOT/('tests/'+m.split('.')[-1]+'.py') for m in modules]
    return write_record(OUT/'completion.json',dict(status='error_review_complete_candidate_failed_fixed_gates',gates=e['gates'],
        reviewed_positive_events=len(r['positive_decisions']),reviewed_negative_events=len(r['negative_decisions']),reasons=e['reasons'],
        test_output=t.stdout+t.stderr,integrity=integrity,
        conclusion='Partial material and no-target improvement versus no augmentation, not consistent across seeds. Common-condition recall recovers versus global augmentation, but historical retention and seed17 original planned gate fail. Clear and occluded targets both have losses; no unique cause certified.',
        next_priority='One nominal dose/window matched non-targeted augmentation control; no new members or labels.',
        selected_candidate=None,training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in paths}))

if __name__=='__main__':print(run()['status'])
