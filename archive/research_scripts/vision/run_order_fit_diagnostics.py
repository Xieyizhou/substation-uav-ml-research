"""Explicit bounded two-worker CPU inference runner, not a scheduled task."""
import subprocess
import sys
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT,freeze,prior,runtime


def main():
    p=freeze();pending=[]
    for key in p['models']:
        path=OUT/(key+'.json')
        if path.exists():runtime.validate(prior.read(path),key,p)
        else:pending.append(key)
    logs=OUT/'runner';logs.mkdir(exist_ok=True)
    active=[]
    try:
        for start in range(0,len(pending),2):
            active=[]
            for key in pending[start:start+2]:
                log=logs/(key+'.log')
                stream=log.open('a')
                child=subprocess.Popen([sys.executable,'-m','scripts.vision.diagnose_small_scale_order_fit','--infer-key',key],
                    cwd=prior.ROOT,stdout=stream,stderr=subprocess.STDOUT)
                active.append((key,child,stream));print('START',key,child.pid,flush=True)
            for key,child,stream in active:
                code=child.wait();stream.close()
                if code:raise RuntimeError('Inference failed: '+key+' see preserved log')
                runtime.validate(prior.read(OUT/(key+'.json')),key,p)
                print('VERIFIED',key,flush=True)
        deps=[OUT/'protocol.json',Path(__file__).resolve()]+[OUT/(k+'.json') for k in p['models']]
        dest=logs/'completion.json'
        if not dest.exists():prior.frozen(dest,dict(status='twelve_fit_units_complete_dataset_review_pending',
            inputs={str(x):prior.file_sha256(x) for x in deps}))
    finally:
        for _,child,stream in active:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            if not stream.closed:stream.close()


if __name__=='__main__':main()
