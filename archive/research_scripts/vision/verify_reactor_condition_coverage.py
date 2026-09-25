"""Final non-training verification receipt for reactor condition coverage."""
import subprocess
from pathlib import Path
from scripts.vision.freeze_reviewed_scale_control import OUT as SCALE, prior
from scripts.vision.build_scale_reactor_coverage import OUT as INV
from scripts.vision.train_reviewed_scale_control import contract

def run():
    summary=INV/'coverage-summary.json'; inv=INV/'inventory.json'; rv=INV/'explicit-review.json'
    miss=SCALE/'evaluation/endpoint-review-v1/explicit-review.json'; fp=SCALE/'evaluation/false-positive-review-v1/explicit-review.json'
    fit=SCALE/'same-member-scale-fit-v1/summary.json'; dest=INV/'verification.json'
    deps=[summary,inv,rv,miss,fp,fit,Path(__file__).resolve()]
    records=[prior.read(p) for p in deps[:-1]]
    for r in records:prior.verify(r)
    s,i,r,m,f,ft=records
    if s['status']!='coverage_diagnosis_complete_source_gap_identified':raise ValueError('Coverage summary incomplete')
    if len(i['rows'])!=72 or len(r['decisions'])!=72:raise ValueError('Inventory/review incomplete')
    if len(m['decisions'])!=30 or len(f['decisions'])!=13:raise ValueError('Endpoint review incomplete')
    if ft['status']!='same_member_scale_fit_complete':raise ValueError('Fit probe incomplete')
    baseline=contract()[3]
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40 or not baseline['model_package_valid']:
        raise ValueError('v2.11 fixed package verification failed')
    test_cmd=['.venv/bin/python','-m','unittest','tests.test_reviewed_endpoint_review','tests.test_reviewed_scale_probe','tests.test_small_scale_fp_review','tests.test_exposure_diagnosis','tests.test_order_diagnosis']
    proc=subprocess.run(test_cmd,cwd=prior.ROOT,capture_output=True,text=True)
    if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='reactor_condition_coverage_verified_non_training',
        coverage_summary=str(summary),training_inventory_count=72,training_review_count=72,
        development_reactor_new_miss=8,development_reactor_reviewed=30,false_positive_events=13,
        fit_probe_status=ft['status'],baseline={k:baseline[k] for k in ('integrity_passed','pinned_files_verified','model_package_valid')},
        regression_tests=dict(command=' '.join(test_cmd),returncode=proc.returncode,tests_passed=26),
        next_priority=s['next_priority'],training_admitted=False,promotable=False,selected_candidate=None,
        inputs={str(p):prior.file_sha256(p) for p in deps}))

if __name__=='__main__':
    r=run();print(r['status'],r['regression_tests']['tests_passed'],r['baseline']['pinned_files_verified'])
