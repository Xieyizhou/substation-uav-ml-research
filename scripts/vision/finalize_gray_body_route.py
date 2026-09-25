"""Audit a usable bounded training route; never start training."""
import copy
import subprocess
import sys
from pathlib import Path
from scripts.vision.freeze_gray_body_control import OUT,prior,freeze,source
from scripts.vision import import_transfer_pilot_review as pilot
from scripts.vision import import_transfer_expansion_review as expansion
from scripts.vision.validate_compensated_loader_receipts import validate
from scripts.vision.evaluate_scale_endpoints import baseline_verify


def check_receipt(p,r,seed):
    if r['status']!='actual_paired_loaders_verified' or set(r['cells'])!={f'R-{seed}',f'G-{seed}'}:raise ValueError('Incomplete cells')
    q=copy.deepcopy(p);rr=copy.deepcopy(r);rr['status']='paired_actual_loaders_verified';rr['cells']={}
    for old,new in ((f'R-{seed}',f'V-{seed}'),(f'G-{seed}',f'VM-{seed}')):
        for field in ('schedules','brightness_factors'):q[field][new]=q[field][old]
        c=copy.deepcopy(r['cells'][old])
        for b in c['batch_records']:
            b['image_tensor_sha256']=b.pop('tensor_sha256');b['full_supervision']=b.pop('supervision')
        rr['cells'][new]=c
    validate(q,rr,seed)


def main():
    p=freeze();paths=[OUT/'protocol.json',OUT/'loader-completion.json',Path(__file__).resolve()]
    prior.verify(prior.read(paths[1]))
    for module in (pilot,expansion):
        r=module.main();e=prior.read(module.DEST/'evidence.json');prior.verify(e)
        if not module.validate(e,r['decisions']):raise ValueError('Unknown review')
        paths += [module.DEST/'label-review.json',module.DEST/'evidence.json']
    refs=[]
    for seed in (7,17,27):
        files=list((OUT/'loader-checks'/str(seed)).glob('attempt-*/complete.json'))
        if len(files)!=1:raise ValueError('Missing/duplicate real preflight')
        r=prior.read(files[0]);prior.verify(r);check_receipt(p,r,seed);paths+=files
        source.complete(f'VM-{seed}')
        cp=source.OUT/'training'/f'VM-{seed}'/'completion.json';c=prior.read(cp)
        ep=source.OUT/'evaluation'/f'VM-{seed}.json';ev=prior.read(ep);prior.verify(ev)
        paths += [cp,ep,Path(c['weights']),Path(c['exposure_path'])]
        refs.append(dict(seed=seed,weights=c['weights'],weights_sha256=c['weights_sha256'],evaluation=str(ep)))
    tests=['tests.test_gray_body_control','tests.test_transfer_pilot_review','tests.test_material_transfer_scope',
           'tests.test_double_dose_fit_comparison','tests.test_material_member_fit']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],capture_output=True,text=True,timeout=120)
    if result.returncode:raise ValueError(result.stdout+result.stderr)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline drift')
    report=prior.ROOT/'docs/results/ml_gray_body_route_ready_20260911.md';paths.append(report)
    paths += [prior.ROOT/(t.replace('.','/')+'.py') for t in tests]
    paths += [prior.ROOT/'scripts/vision'/f for f in ('brightness_transfer_runtime.py','order_retention_runtime.py',
        'preflight_gray_body_control.py','prepare_gray_body_training_candidates.py','validate_compensated_loader_receipts.py')]
    dest=OUT/'route-ready.json'
    if dest.exists():prior.verify(prior.read(dest));return prior.read(dest)
    return prior.frozen(dest,dict(status='ready_for_bounded_condition_transfer_training_not_started',
        route='Same-source target-body gray replacement at 30 frozen positions, not increased exposure.',
        planned_new_training_cells=['G-7','G-17','G-27'],historical_reference_cells=refs,
        actual_loader_cells=6,actual_loaded_images=16200,source_poses=12,new_gray_images=12,
        training_ready=True,training_started=False,training_admitted=False,promotable=False,
        optimizer_created=False,backward_executed=False,training_validation_run=False,
        training_entry_scope='Dataset, schedules and real loader ready; starting optimizer runs remains a separate next-stage action.',
        claim='Feasible evidence-backed experiment route, not proof that this training will improve development results.',
        baseline=baseline,regression_output=result.stdout+result.stderr,whole_repository_tests_claimed=False,
        limits=['Same-source/layout/assets; S07 has 8/30 replacements.','Warm/cool diversity is exchanged for gray condition coverage.',
                'Historical whole-pool risk not re-certified; formal training admission stays false.',
                'No sealed-scene evaluation or model promotion.'],
        inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':print(main()['status'])
