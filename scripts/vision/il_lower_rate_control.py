"""Single learning-rate change; all real training inputs remain identical."""
import copy
from pathlib import Path
from scripts.vision.interleaved_light_control import OUT as REF,freeze as reference_freeze,prior
from scripts.vision.evaluate_interleaved_light import configure,adapter
OUT=REF/'lower-rate-retention-v1'
KEYS=tuple(f'ILR1000-{s}' for s in (7,17,27))

def constraints(p,ref):
    for field in ('pool_rows','initialization','evaluation','held_members','environment','acceptance_policy'):
        if p[field]!=ref[field]:raise ValueError('Non-LR input drift: '+field)
    for key in KEYS:
        old='IL1000-'+key.split('-')[-1]
        for field in ('schedules','brightness_factors','listings'):
            if p[field][key]!=ref[field][old]:raise ValueError('Exposure/augmentation drift')
        if p['training_config'][key]!={**ref['training_config'][old],'lr0':.00025}:raise ValueError('More than LR changed')

def freeze():
    ref=reference_freeze();configure()
    deps=[REF/'design.json',REF/'evaluation/error-review-v1/completion.json',REF/'pool-fit-diagnosis-v1/summary.json',Path(__file__).resolve()]
    for key in KEYS:
        old='IL1000-'+key.split('-')[-1];adapter.complete(old);deps.append(REF/'training'/old/'completion.json')
    for path in deps:
        if path.suffix=='.json':prior.verify(prior.read(path))
    dest=OUT/'design.json'
    if dest.exists():p=prior.read(dest);prior.verify(p);constraints(p,ref);return p
    p=copy.deepcopy(ref);p.pop('identity',None)
    for key in KEYS:
        old='IL1000-'+key.split('-')[-1]
        for field in ('schedules','brightness_factors','listings'):p[field][key]=copy.deepcopy(ref[field][old])
        p['training_config'][key]={**ref['training_config'][old],'lr0':.00025}
    constraints(p,ref);OUT.mkdir(exist_ok=True)
    p.update(status='fixed_lower_learning_rate_control_frozen',comparison='ILR1000 vs same-seed IL1000: only constant learning rate .0005 -> .00025; independent v2.11 initialization, same 1000 steps and exact input sequence.',
        limitations=['52/52 low-light training instances matched for each IL1000 seed, but development transfer is limited.','This is optimization sensitivity diagnosis, not proof that learning rate is the cause or that data coverage is sufficient.','Current development E07/E20 remain unknown; not removed from metrics.'],inputs={**ref['inputs'],**{str(d):prior.file_sha256(d) for d in deps}})
    return prior.frozen(dest,p)

if __name__=='__main__':freeze();print('FROZEN_NO_TRAINING')
