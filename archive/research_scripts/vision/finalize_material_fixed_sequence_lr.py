"""Complete a bounded training-preparation stage, never invoke training."""
import subprocess
import sys
from pathlib import Path
from scripts.vision.material_fixed_sequence_lr import OUT,KEYS,prior,freeze,reference
from scripts.vision.train_material_fixed_sequence_lr import contract
from scripts.vision.exposure_order_retention import baseline_verify

TESTS=('test_material_fixed_sequence_lr','test_material_late_rehearsal_entry','test_material_late_rehearsal',
       'test_exposure_diagnosis','test_active_inspection','test_active_inspection_live','test_rgb_depth_active_inspection')


def main():
    p=freeze();paths=[OUT/'protocol.json',Path(__file__).resolve()]
    oldfit=reference.OUT/'fit-transfer-diagnosis-v1/protocol.json';old=prior.read(oldfit);prior.verify(old)
    if p['environment']!=old['environment']:raise ValueError('Reference inference environment incompatible')
    paths.append(oldfit)
    for key in KEYS:
        files=list((OUT/'loader-checks'/key).glob('attempt-*/complete.json'))
        if len(files)!=1:raise ValueError('Missing/duplicate actual loader')
        r=prior.read(files[0]);prior.verify(r)
        _,_,e=reference.contract('T-'+key.split('-')[-1])
        if any(r[k]!=e[k] for k in e):raise ValueError('Historical tensors or supervision differ')
        if any(r.get(k) is not False for k in ('optimizer_created','backward_executed','training_validation_executed')):raise ValueError('Forbidden operation')
        paths.append(files[0])
        if key.startswith('Q'):contract(key)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline integrity')
    run=subprocess.run([sys.executable,'-m','unittest',*('tests.'+x for x in TESTS)],cwd=prior.ROOT,capture_output=True,text=True,timeout=120)
    if run.returncode:raise RuntimeError(run.stdout+run.stderr)
    paths.extend(prior.ROOT/'tests'/f'{x}.py' for x in TESTS)
    paths.extend((prior.ROOT/'scripts/vision/train_material_fixed_sequence_lr.py',prior.ROOT/'docs/plans/ml_material_fixed_sequence_lr_20260912.md'))
    dest=OUT/'entry-ready.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='ready_for_training_not_started',cells=['Q-7','Q-17','Q-27'],
        preflight_cells=list(KEYS),actual_draws_checked=16200,actual_batches_checked=2700,
        new_training_started=False,baseline=baseline,tests=dict(exit_code=run.returncode,stdout=run.stdout,stderr=run.stderr,scope='targeted_not_full_repository'),
        inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':print(main()['status'])
