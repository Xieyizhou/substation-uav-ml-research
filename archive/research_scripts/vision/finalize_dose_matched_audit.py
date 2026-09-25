"""Summarize explicit current review and fixed numerical gates."""
from pathlib import Path
import subprocess,sys
from scripts.vision.inspect_dose_matched_results import OUT,TRAIN,checked
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
    return write_record(OUT/'completion.json',dict(status='current_error_review_complete_fixed_retention_failed',gates=e['gates'],
        positive_events=len(r['positive_decisions']),negative_events=len(r['negative_decisions']),reasons=e['reasons'],test_output=t.stdout+t.stderr,integrity=integrity,
        facts='Non-targeted nominal-dose/window-matched augmentation lowers lighting recall in all three seeds versus material routing; material deltas vary by seed. Clear cylinders and clear cabinets as well as occluded targets have errors. 14 false positives on nine images.',
        supported_explanation='Assignment matters at this nominal dose; reduced augmentation count alone does not reproduce routed results. Actual pixel perturbation dose, member combinations and within-window placement still differ.',
        unverified='Low confidence is an operational prediction category, not a certified training mechanism. No unique root cause, pixel visibility or new-scene generalization claimed.',
        next_priority='One material-routed half-amplitude contrast family with exactly the prior routed positions, counts, signs, members and budget; independently initialize all three seeds. Test amplitude tradeoff, not a parameter sweep.',
        selected_candidate=None,training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in paths}))

if __name__=='__main__':print(run()['status'])
