"""Verify static artifacts; explicitly cannot grant capture/training success."""
from pathlib import Path
import subprocess,sys
from scripts.vision.prepare_physical_lighting_control import OUT,prior
from scripts.vision.test_body_material_applicability import baseline_verify

def main():
    pp=OUT/'protocol.json';p=prior.read(pp);prior.verify(p)
    if len(p['selected'])!=5 or len(p['excluded'])!=8:raise ValueError('Candidate scope changed')
    for seed,s in p['schedules'].items():
        if s['replaced_exposures']!=98 or len(s['position_to_source'])!=98:raise ValueError('Quota mismatch')
        if any(s['original_sequence'][int(i)]!=m for i,m in s['position_to_source'].items()):raise ValueError('Position mismatch')
        if sum(s['windows_50_steps'])!=98:raise ValueError('Window mismatch')
    tests=['tests.test_physical_lighting_preflight','tests.test_condition_coverage','tests.test_structure_fit','tests.test_structure_fit_integrity','tests.test_lineage_training_fit']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True,timeout=60)
    if t.returncode:raise ValueError(t.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline changed')
    paths=[pp,Path(__file__).resolve(),prior.ROOT/'docs/results/ml_physical_lighting_preflight_20260910.md']+[prior.ROOT/(x.replace('.','/')+'.py') for x in tests]
    prior.frozen(OUT/'completion.json',dict(status='static_preflight_complete_new_rendering_required',training_ready=False,training_started=False,collection_started=False,
        regression=dict(output=t.stderr,whole_repository_tested=False),baseline=baseline,
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('STATIC_PREFLIGHT_COMPLETE_RENDERING_REQUIRED')

if __name__=='__main__':main()
