"""Finalize diagnostic coverage only, explicitly not training readiness."""
from pathlib import Path
import subprocess,sys
from scripts.vision.audit_candidate29_completeness import OUT,prior,capture
from scripts.vision.record_candidate29_screen import validate_screen
from scripts.vision.test_body_material_applicability import baseline_verify

def main():
    paths=[OUT/'protocol.json',OUT/'replay-index.json',OUT/'evidence/manifest.json',OUT/'review-pages/manifest.json',OUT/'review-pages/A16-boundary.json',OUT/'screening.json']
    for x in paths:prior.verify(prior.read(x))
    p,r,e,_,boundary,screen=map(prior.read,paths)
    expected=[f['pair_id'] for f in p['frames']]
    if len(expected)!=29 or {x['pair_id'] for x in r['results']}!=set(expected):raise ValueError('Incomplete replay scope')
    validate_screen(screen['decisions'],expected)
    conflict=[x['pair_id'] for x in e['events'] if x.get('membership_conflict')]
    gaps=[x['pair_id'] for x in e['events'] if x['status']=='named_evidence_gap']
    if conflict!=['A25','A28'] or gaps!=['A16']:raise ValueError('Findings changed; explicit report review required')
    if not all(x['raw_box_equality'] and x['rgb_exact'] and not x['unboxed_visible_labels'] for x in boundary['records']):raise ValueError('Boundary diagnosis changed')
    for x in r['results']:
        receipt=prior.read(x['receipt']);prior.verify(receipt)
        if not receipt['process_cleanup_complete']:raise ValueError('Unclean replay process')
    tests=['tests.test_candidate29_audit','tests.test_l05_hold_gate','tests.test_l05_risk_scope','tests.test_physical_lighting_preflight','tests.test_physical_lighting_capture','tests.test_condition_coverage','tests.test_structure_fit','tests.test_structure_fit_integrity','tests.test_lineage_training_fit']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True,timeout=60)
    if t.returncode:raise ValueError(t.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline changed')
    capture.base.guard()
    paths += [Path(__file__).resolve(),prior.ROOT/'docs/results/ml_candidate29_completeness_20260910.md']+[prior.ROOT/(x.replace('.','/')+'.py') for x in tests]
    prior.frozen(OUT/'completion.json',dict(status='29_source_completeness_screen_complete_with_named_risks',counts=dict(no_membership_conflict=26,omission_risk=2,numeric_boundary_gap=1),
        training_ready=False,training_started=False,labels_modified=False,counts_resolved=False,process_cleanup_complete=True,
        baseline=baseline,regression=dict(output=t.stderr,whole_repository_tested=False),inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('29 AUDITED; 26 NO MEMBERSHIP CONFLICT; 2 OMISSION RISKS; 1 NUMERIC GAP; BASELINE40 PASS');print(t.stderr)

if __name__=='__main__':main()
