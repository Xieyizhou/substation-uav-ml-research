"""Audit actual three-arm loaders and historical controls before optimizer entry."""
import copy
from pathlib import Path
from scripts.vision.freeze_material_retention_coverage import OUT,prior,freeze,source
from scripts.vision.finalize_gray_body_route import check_receipt as pair_receipt
from scripts.vision.train_gray_body_control import complete as gray_complete,OUT as GRAY
from scripts.vision.evaluate_scale_endpoints import baseline_verify


def check_receipt(p,r,seed):
    if r['status']!='actual_three_arm_loaders_verified' or set(r['cells'])!={f'{a}-{seed}' for a in ('R','T','A')}:raise ValueError('Incomplete three-arm preflight')
    for arm in ('T','A'):
        q=copy.deepcopy(p);rr=copy.deepcopy(r);rr['status']='actual_paired_loaders_verified'
        rr['cells']={f'R-{seed}':r['cells'][f'R-{seed}'],f'G-{seed}':r['cells'][f'{arm}-{seed}']}
        for field in ('schedules','brightness_factors'):q[field][f'G-{seed}']=q[field][f'{arm}-{seed}']
        pair_receipt(q,rr,seed)


def main():
    p=freeze();paths=[OUT/'protocol.json',OUT/'loader-completion.json',Path(__file__).resolve(),OUT/'duplicate-resolution.json']
    for path in paths[:2]:prior.verify(prior.read(path))
    for seed in (7,17,27):
        files=list((OUT/'loader-checks'/str(seed)).glob('attempt-*/complete.json'))
        if len(files)!=1:raise ValueError('Missing/ambiguous loader result')
        r=prior.read(files[0]);prior.verify(r);check_receipt(p,r,seed);paths+=files
        for module,key,root in ((source,f'VM-{seed}',source.OUT),(None,f'G-{seed}',GRAY)):
            (module.complete if module else gray_complete)(key)
            cp=root/'training'/key/'completion.json';ep=root/'evaluation'/f'{key}.json'
            c=prior.read(cp);e=prior.read(ep);prior.verify(c);prior.verify(e)
            if e['inputs'].get(c['weights'])!=c['weights_sha256']:raise ValueError('Control binding drift')
            paths.extend([cp,ep,Path(c['weights']),Path(c['exposure_path'])])
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline drift')
    dest=OUT/'route-ready.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='ready_for_material_retention_coverage_training_not_started',
        planned_new_training_cells=[f'{a}-{s}' for a in ('T','A') for s in (7,17,27)],
        actual_loader_cells=9,actual_loaded_images=24300,training_ready=True,training_started=False,
        baseline=baseline,inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':print(main()['status'])
