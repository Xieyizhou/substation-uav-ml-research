"""Independent explicit training entry. Default validates only; never auto-trains."""
import argparse,copy,traceback,fcntl
from pathlib import Path
from scripts.vision.preflight_tail_interleaving import OUT,SOURCE,prior,permutation
from scripts.vision.closed_budget_runtime import check
from scripts.vision.closed_budget_engine_v2 import execute
from scripts.vision.record_bn_review import DEST,validate
from scripts.vision.exposure_order_retention import baseline_verify

def contract():
    dp=SOURCE/'design.json';p=copy.deepcopy(prior.read(dp));prior.verify(p)
    fp=OUT/'loader-feasibility.json';f=prior.read(fp);prior.verify(f)
    ep,rp=DEST/'evidence.json',DEST/'review.json';e,r=prior.read(ep),prior.read(rp)
    prior.verify(e);prior.verify(r);validate(e,r['decisions'])
    if f['permutation']!=permutation() or len(f['units'])!=3:raise ValueError('Permutation/unit drift')
    deps=[dp,fp,ep,rp,Path(__file__).resolve(),Path(__file__).with_name('closed_budget_engine_v2.py')]
    for unit in f['units']:
        key=unit['cell'];old='R1000-'+key.split('-')[-1]
        p['schedules'][key]=unit['actual'];p['brightness_factors'][key]=[z['gain'] for z in unit['brightness_log']]
        p['training_config'][key]=p['training_config'][old];p['listings'][key]=p['listings'][old]
        check(p,key,unit['actual'],unit['brightness_log'])
        cp=SOURCE/'training'/old/'completion.json';c=prior.read(cp);prior.verify(c);deps.extend([cp,Path(c['weights'])])
        if prior.file_sha256(c['weights'])!=c['weights_sha256']:raise ValueError('Reference weight drift')
    init=Path(p['initialization']['path']);deps.append(init)
    b=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not b['integrity_passed'] or b['pinned_files_verified']!=40:raise ValueError('Baseline integrity')
    import platform,torch,ultralytics
    if p['environment']!=dict(python=platform.python_version(),torch=torch.__version__,ultralytics=ultralytics.__version__):raise ValueError('Training environment changed')
    return p,f,deps,b

def _run(key):
    p,f,deps,_=contract();unit=next(x for x in f['units'] if x['cell']==key)
    root=OUT/'training'/key;root.mkdir(parents=True,exist_ok=True)
    cp=root/'completion.json'
    if cp.exists():r=prior.read(cp);prior.verify(r);return r
    n=len(list(root.glob('attempt-*')))+1
    if n>3:raise ValueError('Attempt cap')
    if any(prior.read(x).get('semantic') for x in root.glob('attempt-*/failure.json')):raise ValueError('Unresolved semantic failure')
    attempt=root/f'attempt-{n:03}';attempt.mkdir()
    try:
        x=execute(p,key,attempt,4,unit)
        if x['optimizer_steps']!=1000:raise ValueError('Wrong endpoint')
        xp=attempt/'exposure.json';prior.frozen(xp,{**x,'inputs':{str(d):prior.file_sha256(d) for d in deps}})
        wp=attempt/'run/weights/last.pt'
        if not wp.is_file():raise ValueError('Missing endpoint')
        deps.extend([xp,wp])
        return prior.frozen(cp,dict(status='trained_not_evaluated',cell=key,weights=str(wp),weights_sha256=prior.file_sha256(wp),exposure_path=str(xp),optimizer_steps=1000,inputs={str(d):prior.file_sha256(d) for d in deps}))
    except BaseException as exc:
        prior.frozen(attempt/'failure.json',dict(error=traceback.format_exc(),semantic=isinstance(exc,ValueError),child_processes_started=0));raise

def run(key):
    root=OUT/'training'/key;root.mkdir(parents=True,exist_ok=True)
    with (root/'unit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run(key)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--seed',type=int,choices=(7,17,27),default=7);a=ap.parse_args()
    if a.train:run(f'I1000-{a.seed}')
    else:
        _,f,_,_=contract();print('THREE_CONFIGS_VERIFIED_NOT_STARTED',len(f['units']))
