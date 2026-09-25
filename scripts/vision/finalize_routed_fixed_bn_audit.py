"""Close fixed-BN evidence without creating review decisions."""
import subprocess
import sys
from pathlib import Path
from scripts.vision import routed_fixed_bn_control as arm
from scripts.vision.record_contrast_cpu4_review import validate_positive
from scripts.vision.audit_reviewed_order_results import validate_review
from scripts.vision.verify_experiment_baseline import verify, ROOT
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run():
    out=arm.OUT/'audit-v1'
    paths=[out/'evidence.json',out/'loss-review.json',out/'material-evidence.json',out/'material-review.json',arm.OUT/'evaluation-v1/error-review-v1/evidence.json',out/'negative-review.json',arm.OUT/'evaluation-v1/summary.json']
    e,l,m,r,n,f,s=map(arm.checked,paths)
    validate_positive(e,l['positive_decisions']);validate_positive(m,r['positive_decisions']);validate_review(n,f['negative_decisions'])
    if any(s['matching_conflicts'].values()):raise ValueError('Matching conflict')
    for key in arm.KEYS:
        arm.verified_unit(key)
        paths.extend(arm.OUT/'training'/key/name for name in ('completion.json','bn-verification.json','tensor-verification.json','thread-verification.json'))
    modules=('tests.test_contrast_cpu4_review','tests.test_reviewed_order_evaluation','tests.test_reviewed_order_audit','tests.test_routed_fixed_bn_policy')
    t=subprocess.run([sys.executable,'-m','unittest',*modules,'-q'],capture_output=True,text=True,timeout=180)
    if t.returncode:raise ValueError(t.stdout+t.stderr)
    contract=ROOT/'config/perception/visual_experiment_baseline_v1.json';integrity=verify(contract)
    if not integrity['integrity_passed'] or integrity['pinned_files_verified']!=40:raise ValueError('Fixed40 failed')
    paths += [contract,Path(__file__)]
    return write_record(out/'completion.json',dict(status='fixed_bn_review_complete_candidate_failed',gates=s['gates'],counts=dict(original_lighting_losses=len(l['positive_decisions']),material_losses=len(r['positive_decisions']),negative_predictions=len(f['negative_decisions']),negative_images=len(n['frames'])),test_output=t.stdout+t.stderr,integrity=integrity,
        conclusion='Fixed backbone BN with parameter adaptation reduces material recall in all three seeds relative to original material routing. Losses include clear bodies and occluded/truncated content. This arm is not adopted; no unique root cause established.',
        next_direction='Retain original routed inputs and normal BN. Test a predeclared late learning-rate reduction at unchanged exposure and step budget; improvement remains a hypothesis.',
        training_admitted=False,promotable=False,selected_candidate=None,inputs={str(p.resolve()):file_sha256(p) for p in paths}))

if __name__=='__main__':print(run()['status'])
