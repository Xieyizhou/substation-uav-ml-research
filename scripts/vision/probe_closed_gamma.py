"""Ten-step technical probes; no probe checkpoint is a candidate or resumed."""
import argparse,copy,subprocess,sys,traceback
from pathlib import Path
from scripts.vision.closed_gamma_design import OUT,SOURCE,freeze,prior
from scripts.vision.closed_gamma_engine import execute
from scripts.vision.preflight_closed_gamma import receipt
from scripts.vision.preflight_closed_budget import receipt as base_receipt
from scripts.vision.closed_budget_design import freeze as base_design
from scripts.vision.review_closed_gamma import main as review
from scripts.vision.train_closed_source_control import cleanup

def worker(mode):
    p=freeze();review();key='G900-7';expected,ep=receipt(key,p)
    deps=[OUT/'design.json',ep,OUT/'augmentation-review/review.json',Path(__file__).resolve(),Path(__file__).with_name('closed_gamma_engine.py')]
    if mode=='identity':
        expected,ep=base_receipt('B900-7',base_design());expected=copy.deepcopy(expected);deps.append(ep)
        p=copy.deepcopy(p);p['gamma_factors'][key]=[1.]*900
        expected['gamma_log']=[dict(position=i,member_id=m,gamma=1.,before=b['after'],after=b['after']) for i,(m,b) in enumerate(zip(expected['actual'],expected['brightness_log'],strict=True))]
    root=OUT/'entry-probes'/mode;root.mkdir(parents=True,exist_ok=True)
    dest=root/'completion.json'
    if dest.exists():prior.verify(prior.read(dest));return
    if list(root.glob('attempt-*/failure.json')):raise ValueError('Investigate prior probe failure before retry')
    attempt=root/'attempt-001';attempt.mkdir()
    try:
        r=execute(p,key,attempt,4,expected,probe_steps=10)
        if mode=='identity':
            cp=SOURCE/'training/B900-7/completion.json';c=prior.read(cp);prior.verify(c);xp=Path(c['exposure_path']);x=prior.read(xp);prior.verify(x);deps.extend([cp,xp])
            if [s['loss_items'] for s in r['step_records']]!=[s['loss_items'] for s in x['step_records'][:10]]:raise ValueError('Neutral integration changes optimization')
        prior.frozen(dest,dict(status='real_ten_step_probe_verified',mode=mode,result=r,neutral_expected_is_explicit_test_fixture=mode=='identity',inputs={str(d):prior.file_sha256(d) for d in deps}))
    except BaseException:
        prior.frozen(attempt/'failure.json',dict(error=traceback.format_exc()));raise

def main():
    jobs=[]
    try:
        for mode in ('identity','gamma'):
            root=OUT/'entry-probes'/mode;root.mkdir(parents=True,exist_ok=True)
            log=(root/'worker.log').open('a')
            proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.probe_closed_gamma','--worker',mode],cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append((proc,log))
        for proc,_ in jobs:
            proc.wait(timeout=600)
            if proc.returncode:raise RuntimeError('Entry probe failed; inspect independent attempt')
    finally:
        for proc,log in jobs:cleanup(proc);log.close()
    deps=[OUT/'entry-probes'/m/'completion.json' for m in ('identity','gamma')]
    for d in deps:prior.verify(prior.read(d))
    dest=OUT/'entry-probes/completion.json'
    if not dest.exists():prior.frozen(dest,dict(status='neutral_optimization_exact_and_gamma_entry_verified',inputs={str(d):prior.file_sha256(d) for d in deps}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=('identity','gamma'));a=ap.parse_args()
    if a.worker:worker(a.worker)
    else:main()
