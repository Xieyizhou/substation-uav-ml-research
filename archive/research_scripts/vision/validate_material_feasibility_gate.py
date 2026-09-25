"""Validate the bounded blocked handoff, not completion of downstream science."""
import subprocess
import sys
from pathlib import Path
from scripts.vision.material_control_feasibility import OUT, prior
from scripts.vision.verify_experiment_baseline import verify as baseline_verify

def main():
    p=OUT/'initial-gate.json';r=prior.read(p);prior.verify(r)
    if r['status']!='blocked_at_material_candidate_quality_gate' or not r['blockers']:
        raise ValueError('Expected documented quality blocker')
    for x in r['corrected_target_records']:
        if prior.file_sha256(x['crop_path'])!=x['crop_sha256']:raise ValueError('Stale crop')
    suites=['tests.test_material_control_feasibility','tests.test_closed_material_review',
            'tests.test_canonical_gates','tests.test_exposure_diagnosis']
    test=subprocess.run([sys.executable,'-m','unittest',*suites],capture_output=True,text=True)
    if test.returncode:raise ValueError(test.stdout+test.stderr)
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline changed')
    paths=[p,Path(__file__).resolve(),prior.ROOT/'docs/results/ml_material_control_feasibility_20260911.md']
    paths += [prior.ROOT/(s.replace('.','/')+'.py') for s in suites]
    prior.frozen(OUT/'gate-validation.json',dict(status='blocked_handoff_validated_not_full_stage_complete',
        full_plan_complete=False,baseline=b,tests_output=test.stdout+test.stderr,
        untested_deferred=['new all-condition visual review import','full metrics/version closure','complete coverage census'],
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print(test.stdout+test.stderr);print('BLOCKED_HANDOFF_VALIDATED; PINNED40_PASS')

if __name__=='__main__':main()
