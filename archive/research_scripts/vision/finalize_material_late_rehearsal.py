"""Verify all immutable configuration gates; never start an optimizer."""
import subprocess,sys
from pathlib import Path
from scripts.vision.train_material_late_rehearsal import OUT,KEYS,prior,contract,reference
from scripts.vision.freeze_material_late_rehearsal import main as feasibility
from scripts.vision.evaluate_scale_endpoints import baseline_verify

def main():
    f=feasibility();paths=[OUT/'feasibility.json',OUT/'protocol.json',Path(__file__).resolve()];cells={}
    for key in KEYS:
        p,_,new=contract(key);seed=key.split('-')[-1];rk='T-'+seed
        reference.complete(rk);old,_,before=reference.contract(rk)
        changed={i for i,(a,b) in enumerate(zip(old['schedules'][rk],p['schedules'][key],strict=True)) if a!=b}
        if len(changed)!=12:raise ValueError('Change count drift')
        if p['brightness_factors'][key]!=old['brightness_factors'][rk]:raise ValueError('Brightness changed')
        for i,(a,b) in enumerate(zip(before['brightness_log'],new['brightness_log'],strict=True)):
            if a['gain']!=b['gain']:raise ValueError('Actual brightness changed')
            if i not in changed and a!=b:raise ValueError('Untreated actual image/tensor changed')
        for i,b in enumerate(new['batch_records']):
            if b['step']!=i or b['members']!=p['schedules'][key][i*6:(i+1)*6]:raise ValueError('Batch/member mismatch')
            if not changed.intersection(range(i*6,(i+1)*6)) and b!=before['batch_records'][i]:raise ValueError('Untreated batch/supervision changed')
        cells[key]=dict(actual_samples=2700,actual_batches=450,changed_slots=sorted(changed),
            first_minimum_last_source_step=f['cells'][key]['earliest_last_source_step'],historical_reference=rk)
        paths.extend((OUT/'loader-checks'/key).glob('attempt-*/complete.json'))
        paths.append(reference.OUT/'training'/rk/'completion.json')
    suites=['test_material_late_rehearsal','test_material_late_rehearsal_entry','test_material_retention_entry',
            'test_compensated_loader_receipts','test_exposure_diagnosis']
    test=subprocess.run([sys.executable,'-m','unittest',*['tests.'+s for s in suites]],capture_output=True,text=True,timeout=120)
    if test.returncode:raise ValueError(test.stdout+test.stderr)
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline changed')
    paths.extend(prior.ROOT/'tests'/f'{s}.py' for s in suites)
    paths.extend(prior.ROOT/'scripts/vision'/s for s in ['train_material_late_rehearsal.py','preflight_material_late_rehearsal.py',
        'brightness_transfer_runtime.py','order_retention_runtime.py','frozen_multiscale_runtime.py'])
    paths.append(prior.ROOT/'docs/plans/ml_material_late_rehearsal_20260912.md')
    if (OUT/'training-authorization.json').exists() or (OUT/'training').exists():raise ValueError('Unexpected training already started')
    dest=OUT/'entry-ready.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='ready_for_training_not_started',cells=list(KEYS),verified_cells=cells,
        baseline=b,regression_output=test.stdout+test.stderr,training_started=False,
        whole_repository_tests_claimed=False,interpretation='Fixed-multiset source-level timing strategy, not pure exposure increase.',
        inputs={str(x):prior.file_sha256(x) for x in paths}))

if __name__=='__main__':print(main()['status'])
