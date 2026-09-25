"""Finish diagnostic policy/precheck, never grants training readiness."""
from pathlib import Path
import subprocess,sys
from scripts.vision.validate_boundary_roundoff import OUT,prior,capture
from scripts.vision.test_body_material_applicability import baseline_verify

def main():
    paths=[OUT/'boundary-validation.json',OUT/'policy.json',OUT/'quality-precheck.json']
    for p in paths:prior.verify(prior.read(p))
    policy=prior.read(paths[1]);quality=prior.read(paths[2])
    if len(policy['future_zero_exposure_members'])!=10 or len(policy['new_held_members'])!=2:raise ValueError('Hold scope changed')
    if len(quality['eligible_members'])!=27 or quality['uncovered_members']:raise ValueError('Quality evidence missing')
    tests=['tests.test_boundary_roundoff','tests.test_candidate29_audit','tests.test_l05_hold_gate','tests.test_l05_risk_scope','tests.test_physical_lighting_preflight','tests.test_physical_lighting_capture','tests.test_condition_coverage','tests.test_structure_fit','tests.test_structure_fit_integrity','tests.test_lineage_training_fit']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True,timeout=60)
    if t.returncode:raise ValueError(t.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline changed')
    capture.base.guard()
    paths += [Path(__file__).resolve(),prior.ROOT/'docs/results/ml_candidate29_unified_policy_20260910.md']+[prior.ROOT/(x.replace('.','/')+'.py') for x in tests]
    prior.frozen(OUT/'completion.json',dict(status='unified_policy_and_quality_precheck_complete_new_design_required',training_ready=False,training_started=False,
        historical_labels_modified=False,production_parser_modified=False,counts_resolved=False,loader_preflight_run=False,
        baseline=baseline,regression=dict(output=t.stderr,whole_repository_tested=False),inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('UNIFIED_POLICY_COMPLETE; NO TRAINING READINESS');print(t.stderr)

if __name__=='__main__':main()
