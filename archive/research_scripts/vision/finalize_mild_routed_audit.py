"""Validate explicit current observations; no automatic candidate approval."""
import subprocess, sys
from pathlib import Path
from scripts.vision.record_mild_routed_review import OUT, TRAIN, checked, validate_positive
from scripts.vision.audit_reviewed_order_results import validate_review
from scripts.vision.verify_experiment_baseline import verify, ROOT
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run():
    paths=[OUT/'evidence.json',OUT/'review.json',TRAIN/'evaluation-v1/error-review-v1/evidence.json',OUT/'material-evidence.json',OUT/'material-review.json']
    e,r,n,m,mr=map(checked,paths)
    validate_positive(e,r['positive_decisions']);validate_positive(m,mr['positive_decisions']);validate_review(n,r['negative_decisions'])
    modules=('tests.test_contrast_cpu4_review','tests.test_mild_routed_contrast_control','tests.test_reviewed_order_evaluation','tests.test_reviewed_order_audit')
    t=subprocess.run([sys.executable,'-m','unittest',*modules,'-q'],capture_output=True,text=True,timeout=180)
    if t.returncode:raise ValueError(t.stdout+t.stderr)
    contract=ROOT/'config/perception/visual_experiment_baseline_v1.json';integrity=verify(contract)
    if not integrity['integrity_passed'] or integrity['pinned_files_verified']!=40:raise ValueError('Fixed40 failed')
    paths += [contract,Path(__file__),ROOT/'scripts/vision/record_mild_routed_review.py',ROOT/'scripts/vision/inspect_mild_routed_results.py']
    return write_record(OUT/'completion.json',dict(status='review_complete_not_candidate_passed',gates=e['gates'],
        positive_events=len(r['positive_decisions']),material_events=len(mr['positive_decisions']),negative_events=len(r['negative_decisions']),
        test_output=t.stdout+t.stderr,integrity=integrity,
        conclusion='Half amplitude does not resolve material transfer; clear bodies as well as occluded/truncated targets are affected. Training fitting remains near saturated. No unique mechanism certified.',
        training_admitted=False,promotable=False,selected_candidate=None,inputs={str(p.resolve()):file_sha256(p) for p in paths}))

if __name__=='__main__':print(run()['status'])
