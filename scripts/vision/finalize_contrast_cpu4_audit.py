"""Summarize existing decisions, never auto-pass observations or model gates."""
from pathlib import Path
import subprocess,sys
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision.inspect_contrast_cpu4_results import ROOT,OUT,checked
from scripts.vision.record_contrast_cpu4_review import validate_positive
from scripts.vision.audit_reviewed_order_results import validate_review
from scripts.vision.verify_experiment_baseline import verify,ROOT as REPO

def run():
    ep=OUT/'evidence.json';rp=OUT/'review.json';np=ROOT/'contrast/evaluation-v1/error-review-v1/evidence.json'
    e,r,n=checked(ep),checked(rp),checked(np);validate_positive(e,r['positive_decisions']);validate_review(n,r['negative_decisions'])
    modules=('tests.test_contrast_cpu4_review','tests.test_contrast_cpu4_control','tests.test_locked_cpu_threads','tests.test_contrast_transfer_control','tests.test_reviewed_order_evaluation','tests.test_reviewed_order_audit')
    result=subprocess.run([sys.executable,'-m','unittest',*modules,'-q'],capture_output=True,text=True,timeout=120)
    if result.returncode:raise ValueError(result.stdout+result.stderr)
    contract=REPO/'config/perception/visual_experiment_baseline_v1.json';integrity=verify(contract)
    if not integrity['integrity_passed'] or integrity['pinned_files_verified']!=40:raise ValueError('Fixed-40 failed')
    deps=[ep,rp,np,contract,Path(__file__)]+[REPO/('tests/'+m.split('.')[-1]+'.py') for m in modules]
    return write_record(OUT/'completion.json',dict(status='new_arm_error_review_and_fixed_gate_audit_complete_not_passed',gates=e['gates'],
        positive_events=len(r['positive_decisions']),negative_events=len(r['negative_decisions']),loss_reasons=e['loss_reasons'],test_output=result.stdout+result.stderr,integrity=integrity,
        conclusion='Material recall improves in each seed, but original/lighting class retention fails. This is a tradeoff, not a successful candidate or unique mechanism.',
        next_priority='One bounded routing control: preserve original/physical-lighting/negative pixels, apply the same frozen contrast only on registered material variants. Dose also changes; do not infer pure routing efficacy.',
        selected_candidate=None,training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in deps}))

if __name__=='__main__':print(run()['status'])
