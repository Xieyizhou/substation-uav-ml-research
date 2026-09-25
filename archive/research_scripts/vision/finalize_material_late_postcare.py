"""Read-only scientific checks followed by an immutable postcare receipt."""
import subprocess
import sys
from pathlib import Path
from scripts.vision.diagnose_material_late_rehearsal_fit import OUT, PREVIOUS, TRAIN, prior, freeze, validate
from scripts.vision.diagnose_material_retention_transfer import validate as old_validate
from scripts.vision.finalize_material_late_rehearsal_evaluation import TESTS
from scripts.vision.evaluate_material_late_rehearsal import evaluate, baseline_verify
from scripts.vision.record_material_late_rehearsal_review import validate as review_validate


def main():
    p=freeze(); op=prior.read(PREVIOUS/'protocol.json'); prior.verify(op)
    paths=[OUT/'protocol.json', PREVIOUS/'protocol.json', OUT/'summary.json', TRAIN/'evaluation/completion.json']
    for path in paths: prior.verify(prior.read(path))
    for seed in (7,17,27):
        evaluate(f'L-{seed}')
        for key,root,protocol,validator in ((f'L-{seed}',OUT,p,validate),(f'T-{seed}',PREVIOUS,op,old_validate)):
            path=root/(key+'.json'); r=prior.read(path);validator(r,key,protocol)
            if any(r.get(k) is not False for k in ('optimizer_created','backward_executed','training_validation_executed')):
                raise ValueError('Forbidden operation or missing runtime record')
            paths.append(path)
    for name in ('evidence.json','review.json','review-summary.json'):
        path=TRAIN/'evaluation/error-review-v1'/name; prior.verify(prior.read(path));paths.append(path)
    e,r=(prior.read(TRAIN/'evaluation/error-review-v1'/n) for n in ('evidence.json','review.json'))
    review_validate(e,r['decisions'])
    s=prior.read(OUT/'summary.json')
    if len(s['historical_cohort'])!=13 or len(s['counts'])!=21: raise ValueError('Incomplete fit summary')
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40: raise ValueError('Baseline drift')
    tests=(*TESTS,'test_material_late_fit')
    command=[sys.executable,'-m','unittest',*('tests.'+x for x in tests)]
    run=subprocess.run(command,cwd=prior.ROOT,capture_output=True,text=True,timeout=120)
    if run.returncode: raise RuntimeError(run.stdout+run.stderr)
    paths.extend(prior.ROOT/'tests'/f'{x}.py' for x in tests)
    paths.extend((Path(__file__).resolve(),prior.ROOT/'docs/results/ml_material_late_rehearsal_postcare_20260912.md'))
    dest=OUT/'completion.json'
    if dest.exists():
        record=prior.read(dest);prior.verify(record);return record
    return prior.frozen(dest,dict(status='post_training_diagnosis_complete_no_candidate',
        selected_candidate=None,next_direction='fixed_sequence_single_learning_rate_control_not_started',
        new_training_started=False,baseline=baseline,
        tests=dict(command=command,exit_code=run.returncode,stdout=run.stdout,stderr=run.stderr,scope='targeted_not_full_repository'),
        inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__': print(main()['status'])
