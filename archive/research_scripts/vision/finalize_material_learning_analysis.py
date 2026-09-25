"""Close the analysis only after complete output, endpoint and source checks."""
import subprocess
import sys
from pathlib import Path
from scripts.vision.analyze_material_learning_trajectory import OUT,TRAIN,prior,freeze,validate,MODES
from scripts.vision.train_material_learning_trajectory import complete,historical
from scripts.vision.evaluate_scale_endpoints import baseline_verify


def main():
    p=freeze();paths=[OUT/'protocol.json',OUT/'summary.json',OUT/'transition-windows.json',Path(__file__).resolve()]
    for path in paths[1:3]:prior.verify(prior.read(path))
    import torch
    hp=prior.read(historical.OUT/'protocol.json')
    init=torch.load(hp['initialization']['path'],map_location='cpu',weights_only=False)
    init=(init.get('ema') or init['model']).float().state_dict()
    verified=0
    for key,cs in p['cells'].items():
        complete(key)
        first=torch.load(cs[0]['path'],map_location='cpu',weights_only=False)
        for mode in ('model','ema'):
            state=first[mode].state_dict()
            if set(state)!=set(init) or any(not torch.equal(state[k],init[k]) for k in init):raise ValueError('Initial state mismatch')
        for c in cs:
            for mode in MODES:
                path=OUT/key/f"{c['step']:03}-{mode}"/'complete.json'
                validate(prior.read(path),p,key,c,mode);paths.append(path);verified+=1
    if verified!=394:raise ValueError('Missing unit')
    suites=['test_material_transition_windows','test_material_trajectory_analysis','test_material_learning_trajectory',
            'test_material_trajectory_runtime','test_exposure_diagnosis','test_material_retention_review']
    r=subprocess.run([sys.executable,'-m','unittest',*['tests.'+s for s in suites]],capture_output=True,text=True,timeout=120)
    if r.returncode:raise ValueError(r.stdout+r.stderr)
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline changed')
    paths.append(prior.ROOT/'docs/results/ml_material_learning_trajectory_20260912.md')
    paths.extend(prior.ROOT/'tests'/f'{s}.py' for s in suites)
    return prior.frozen(OUT/'completion.json',dict(status='trajectory_analysis_complete_no_candidate_selected',
        inference_units_verified=verified,initial_raw_and_ema_equal_initialization=True,
        terminal_half_precision_equal_historical=True,baseline=b,regression_output=r.stdout+r.stderr,
        whole_repository_tests_claimed=False,selected_candidate=None,
        next_direction='feasibility_of_fixed_multiset_source_level_late_material_rehearsal',
        inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':print(main()['status'])
