"""Verify completed development review and the subsequent fitting research."""
import subprocess
import sys
from pathlib import Path
from scripts.vision.routed_backbone_control import OUT as TRAIN, KEYS, checked, verified_unit
from scripts.vision.diagnose_routed_backbone_fit import OUT as FIT
from scripts.vision.record_contrast_cpu4_review import validate_positive
from scripts.vision.audit_reviewed_order_results import validate_review
from scripts.vision.review_backbone_fit_residuals import validate
from scripts.vision.verify_experiment_baseline import ROOT, verify
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run():
    audit=TRAIN/'audit-v1'
    paths=[audit/'evidence.json',audit/'material-evidence.json',audit/'review.json',TRAIN/'evaluation-v1/error-review-v1/evidence.json',TRAIN/'evaluation-v1/summary.json',FIT/'residual-review-v1/evidence.json',FIT/'residual-review-v1/review.json',FIT/'paired-comparison.json']
    e,m,r,n,s,fe,fr,p=map(checked,paths)
    validate_positive(e,r['positive_decisions']);validate_positive(m,r['material_decisions']);validate_review(n,r['negative_decisions']);validate(fe,fr['decisions'])
    if any(s['matching_conflicts'].values()):raise ValueError('Unresolved matching conflict')
    if len(p['units'])!=3 or any(u['truth']!=311 for u in p['units']):raise ValueError('Incomplete fit comparison')
    for key in KEYS:
        verified_unit(key)
        for q in [TRAIN/'training'/key/'completion.json',TRAIN/'training'/key/'backbone-verification.json',FIT/f'{key}.json',FIT/f'{key}-verified.json']:
            checked(q);paths.append(q)
    tests=['tests.test_routed_backbone_control','tests.test_reviewed_order_evaluation','tests.test_reviewed_order_audit']
    t=subprocess.run([sys.executable,'-m','unittest',*tests,'-q'],capture_output=True,text=True,timeout=180)
    if t.returncode:raise ValueError(t.stdout+t.stderr)
    contract=ROOT/'config/perception/visual_experiment_baseline_v1.json';integrity=verify(contract)
    if not integrity['integrity_passed'] or integrity['pinned_files_verified']!=40:raise ValueError('Fixed40 failed')
    paths += [contract,Path(__file__),FIT/'research-entry.json',FIT/'research-conclusion-zh.md']
    return write_record(FIT/'completion.json',dict(status='backbone_result_and_followup_fit_research_complete',development_gates=s['gates'],review_counts=r['counts'],fit_review_images=fr['unique_images'],fit_review_events=fr['events'],fit_hits={u['key']:{'reference':u['old_hits'],'frozen_backbone':u['new_hits'],'truth':u['truth']} for u in p['units']},test_output=t.stdout+t.stderr,test_modules=tests,integrity=integrity,selected_candidate=None,next_direction='Separate backbone parameter updates from BatchNorm-statistics policy in a prospectively frozen control; not yet trained.',training_admitted=False,promotable=False,inputs={str(q.resolve()):file_sha256(q) for q in paths}))

if __name__=='__main__':print(run()['status'])
