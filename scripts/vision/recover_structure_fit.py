"""Receipt-publication repair only; frozen predictions/protocol are unchanged."""
import argparse
from pathlib import Path
import os,signal,subprocess,sys
from scripts.vision.run_structure_fit import OUT,ROOT,read,verify,file_sha256,frozen,validate_unit,worker,cleanup

def publish(key,attempt,p):
    rp=attempt/'result.json';r=read(rp);validate_unit(r,key,p)
    cp=OUT/'inference'/f'{key}.json'
    if cp.exists():validate_unit(read(cp),key,p);return
    frozen(cp,{**{k:v for k,v in r.items() if k!='identity'},'inputs':{**r['inputs'],str(rp):file_sha256(rp),str(OUT/'publication-repair.json'):file_sha256(OUT/'publication-repair.json')}})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');ap.add_argument('--worker');ap.add_argument('--attempt',type=Path);args=ap.parse_args()
    if not args.infer and not args.worker:print('PREFLIGHT_ONLY');return
    p=read(OUT/'protocol.json');verify(p)
    if args.worker:
        try:worker(args.worker,args.attempt)
        except TypeError as exc:
            if "multiple values for keyword argument 'inputs'" not in str(exc):raise
            publish(args.worker,args.attempt,p)
        return
    repair=OUT/'publication-repair.json'
    if not repair.exists():frozen(repair,dict(status='publication_only_repair',reason='dict inputs keyword duplicated after complete result saved; validate and publish complete result without recomputing predictions.',inputs={str(OUT/'protocol.json'):file_sha256(OUT/'protocol.json'),str(Path(__file__)):file_sha256(Path(__file__))}))
    else:verify(read(repair))
    lock=OUT/'inference.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    try:
        for key in p['models']:
            cp=OUT/'inference'/f'{key}.json'
            if cp.exists():validate_unit(read(cp),key,p);continue
            folder=OUT/'inference'/key;folder.mkdir(exist_ok=True)
            complete=sorted(folder.glob('attempt-*/result.json'))
            if complete:
                publish(key,complete[0].parent,p);print('RECOVER_VALID_RESULT',key,flush=True);continue
            for number in range(len(list(folder.glob('attempt-*')))+1,4):
                attempt=folder/f'attempt-{number:03}';attempt.mkdir();proc=None
                try:
                    with (attempt/'worker.log').open('w') as log:
                        proc=subprocess.Popen([sys.executable,'-m','scripts.vision.recover_structure_fit','--worker',key,'--attempt',str(attempt)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        code=proc.wait(timeout=3600)
                    if code:
                        frozen(attempt/'failure.json',dict(status='failed',returncode=code,inputs={str(attempt/'worker.log'):file_sha256(attempt/'worker.log')}))
                        if 'ValueError:' in (attempt/'worker.log').read_text():raise ValueError('Semantic worker failure')
                        continue
                    validate_unit(read(cp),key,p);print('COMPLETE',key,flush=True);break
                except BaseException as exc:
                    if not (attempt/'failure.json').exists():frozen(attempt/'failure.json',dict(status='interrupted_or_exception',error=str(exc),inputs={}))
                    raise
                finally:
                    if proc:cleanup(proc)
            else:raise ValueError('Attempt limit exhausted')
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
