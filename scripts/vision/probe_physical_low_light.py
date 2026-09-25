"""Actual model entry probe; fresh v2.11 starts, no probe weights reused."""
import argparse,subprocess,sys,traceback
from pathlib import Path
from scripts.vision.physical_low_light_design import OUT,BASE,freeze,prior
from scripts.vision.preflight_physical_low_light import receipt
from scripts.vision.closed_budget_engine_v2 import execute
from scripts.vision.train_closed_source_control import cleanup

def worker(key):
    p=freeze();expected,ep=receipt(key,p);root=OUT/'entry-probes'/key;root.mkdir(parents=True,exist_ok=True);dest=root/'completion.json'
    if dest.exists():prior.verify(prior.read(dest));return
    if list(root.glob('attempt-*')):raise ValueError('Prior probe attempt requires investigation')
    attempt=root/'attempt-001';attempt.mkdir()
    try:
        result=execute(p,key,attempt,4,expected,probe_steps=10)
        cp=BASE/'training/B900-7/completion.json';c=prior.read(cp);prior.verify(c);xp=Path(c['exposure_path']);ref=prior.read(xp);prior.verify(ref)
        if [x['loss_items'] for x in result['step_records']]!=[x['loss_items'] for x in ref['step_records'][:10]]:raise ValueError('Shared prefix optimization drift')
        deps=[ep,cp,xp,OUT/'design.json',Path(__file__).resolve(),Path(__file__).with_name('closed_budget_engine_v2.py')]
        prior.frozen(dest,dict(status='actual_entry_prefix_probe_verified',result=result,inputs={str(d):prior.file_sha256(d) for d in deps}))
    except BaseException:prior.frozen(attempt/'failure.json',dict(error=traceback.format_exc()));raise

def main():
    jobs=[]
    try:
        for key in ('R1000-7','L1000-7'):
            root=OUT/'entry-probes'/key;root.mkdir(parents=True,exist_ok=True);log=(root/'worker.log').open('a')
            proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.probe_physical_low_light','--worker',key],cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);jobs.append((proc,log))
        for proc,_ in jobs:
            proc.wait(timeout=600)
            if proc.returncode:raise RuntimeError('Entry probe failure')
    finally:
        for proc,log in jobs:cleanup(proc);log.close()
    deps=[OUT/'entry-probes'/k/'completion.json' for k in ('R1000-7','L1000-7')]
    a,b=[prior.read(p) for p in deps]
    for r in (a,b):prior.verify(r)
    if a['result']['terminal_model_state_identity']!=b['result']['terminal_model_state_identity']:raise ValueError('Paired probe model states differ')
    dest=OUT/'entry-probes/completion.json'
    if not dest.exists():prior.frozen(dest,dict(status='paired_actual_entry_prefix_exact',inputs={str(d):prior.file_sha256(d) for d in deps}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=('R1000-7','L1000-7'));a=ap.parse_args()
    if a.worker:worker(a.worker)
    else:main()
