"""Validate explicit reviews and preserve paired gains as well as losses."""
from collections import Counter
from pathlib import Path
import subprocess
import sys
from scripts.vision import routed_late_decay_control as arm
from scripts.vision.record_contrast_cpu4_review import validate_positive
from scripts.vision.audit_reviewed_order_results import validate_review
from scripts.vision.verify_experiment_baseline import verify,ROOT
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run():
    out=arm.OUT/'audit-v1'
    paths=[out/'evidence.json',out/'material-evidence.json',arm.OUT/'evaluation-v1/error-review-v1/evidence.json',out/'review.json',arm.OUT/'evaluation-v1/summary.json',out/'report-zh.md']
    e,m,n,r,s=map(arm.checked,paths[:5])
    validate_positive(e,r['positive_decisions']);validate_positive(m,r['material_decisions']);validate_review(n,r['negative_decisions'])
    if any(s['matching_conflicts'].values()):raise ValueError('Matching conflict')
    for key in arm.KEYS:
        arm.verified_unit(key)
        paths.extend(arm.OUT/'training'/key/name for name in ('completion.json','lr-verification.json','tensor-verification.json','thread-verification.json'))
    modules=('tests.test_contrast_cpu4_review','tests.test_reviewed_order_evaluation','tests.test_reviewed_order_audit','tests.test_routed_late_decay_policy')
    t=subprocess.run([sys.executable,'-m','unittest',*modules,'-q'],capture_output=True,text=True,timeout=180)
    if t.returncode:raise ValueError(t.stdout+t.stderr)
    contract=ROOT/'config/perception/visual_experiment_baseline_v1.json';integrity=verify(contract)
    if not integrity['integrity_passed'] or integrity['pinned_files_verified']!=40:raise ValueError('Fixed40 failed')
    paired={}
    for variant in ('original','material','background','lighting'):
        paired[variant]={}
        for key in arm.KEYS:
            rows=[x for x in s['instance_comparisons'] if x['key']==key and x['variant']==variant]
            counts=Counter(x['state'] for row in rows for x in row['instances'])
            if sum(counts.values())!=60:raise ValueError('Incomplete paired population')
            paired[variant][key]=dict(counts)
    paths.extend([contract,Path(__file__)])
    return write_record(out/'completion.json',dict(status='late_decay_error_review_complete_candidate_failed',gates=s['gates'],paired_counts=paired,
        counts=dict(original_lighting_losses=len(r['positive_decisions']),material_losses=len(r['material_decisions']),negative_predictions=len(r['negative_decisions']),negative_images=len(n['frames'])),
        diagnosis_counts=dict(Counter(x['diagnosis']['reason'] for x in r['positive_decisions']+r['material_decisions'])),
        test_output=t.stdout+t.stderr,integrity=integrity,selected_candidate=None,training_admitted=False,promotable=False,
        limits='Visual review covers declared loss queues and all negative predictions, not every development truth or pixel visibility. Seed repetitions are not new scenes. Legacy evidence-build status says scale due to renderer reuse; this receipt binds actual late-decay paths.',
        inputs={str(p.resolve()):file_sha256(p) for p in paths}))

if __name__=='__main__':print(run()['status'])
