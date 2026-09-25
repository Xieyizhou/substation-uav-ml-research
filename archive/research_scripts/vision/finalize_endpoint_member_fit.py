"""Finalize bounded native-image fit; do not certify unmeasured lighting inputs."""
from pathlib import Path
import subprocess,sys
from scripts.vision.endpoint_member_fit import OUT,KEYS,prior,freeze,check,validate_unit,reviewed
from scripts.vision.evaluate_scale_endpoints import baseline_verify

def main():
    p=freeze();check(p);sp=OUT/'summary.json';s=prior.read(sp);prior.verify(s)
    paths=[OUT/'protocol.json',sp,Path(__file__).resolve(),prior.ROOT/'docs/results/ml_endpoint_member_fit_20260911.md']
    for key in KEYS:
        path=OUT/'inference'/f'{key}.json';validate_unit(prior.read(path),key,p);paths.append(path)
    cp=reviewed.PRIOR/'coverage-census.json';c=prior.read(cp);prior.verify(c);paths.append(cp)
    lighting=[dict(member_id=x['member_id'],actual_exposures={k:m['counts'].get(x['member_id'],0) for k,m in p['models'].items()}) for x in c['members'] if x.get('source_variant')=='physical-lighting']
    tests=['tests.test_endpoint_member_fit','tests.test_material_member_fit','tests.test_exposure_diagnosis','tests.test_pixel_duplicates']
    t=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True)
    if t.returncode:raise ValueError(t.stdout+t.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline failure')
    paths += [prior.ROOT/(x.replace('.','/')+'.py') for x in tests]
    prior.frozen(OUT/'completion.json',dict(status='bounded_native_fit_complete_lighting_evidence_gap',weights=9,images=12,truths=42,independent_pose_groups=4,
        known_physical_lighting_members=lighting,all_actual_brightness_tensors_evaluated=False,
        selected_candidate=None,training_started=False,whole_repository_tests_claimed=False,regression_output=t.stdout+t.stderr,baseline=baseline,
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('BOUNDED_FIT_COMPLETE; PINNED40_PASS; NO_TRAINING')

if __name__=='__main__':main()
