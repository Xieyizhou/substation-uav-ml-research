"""Verify existing decisions and archive the bounded verification receipt."""
from pathlib import Path
import subprocess
import sys
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision.clear_context_lr_control import OUT,checked
from scripts.vision.record_clear_context_lr_review import validate_material
from scripts.vision.audit_reviewed_order_results import validate_review
from scripts.vision.verify_experiment_baseline import ROOT,verify


def run():
    root=OUT/'verification-v1';ep=root/'evidence.json';rp=root/'review.json';fp=OUT/'evaluation-v1/error-review-v1/evidence.json'
    e,r,f=checked(ep),checked(rp),checked(fp)
    validate_material(e,r['material_decisions']);validate_review(f,r['fp_decisions'])
    if any(e['matching_conflicts'].values()):raise ValueError('Unresolved matching conflict')
    contract=ROOT/'config/perception/visual_experiment_baseline_v1.json';integrity=verify(contract)
    if not integrity['integrity_passed'] or integrity['pinned_files_verified']!=40:raise ValueError('Fixed-40 integrity failed')
    modules=['tests.test_clear_context_lr_review','tests.test_clear_context_lr_control','tests.test_reviewed_order_evaluation','tests.test_reviewed_order_audit']
    command=[sys.executable,'-m','unittest',*modules,'-q'];test=subprocess.run(command,capture_output=True,text=True,timeout=120)
    if test.returncode:raise ValueError(test.stdout+test.stderr)
    deps=[ep,rp,fp,root/'report-zh.md',contract,Path(__file__)]+[ROOT/('tests/'+m.split('.')[-1]+'.py') for m in modules]
    path=root/'completion.json'
    if path.exists():return checked(path)
    return write_record(path,dict(status='bounded_material_and_negative_verification_complete_model_not_passed',
        material_truths=len(r['material_decisions']),material_images=len(e['frames']),negative_predictions=len(r['fp_decisions']),negative_images=len(f['frames']),
        gates=e['gates'],test_command=command,test_output=test.stdout+test.stderr,integrity=integrity,
        scope='Material and negative visual review; retention numerical checks. Not full positive-error review or model admission.',
        training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in deps}))


if __name__=='__main__':print(run()['status'])
