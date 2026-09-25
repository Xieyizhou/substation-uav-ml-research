"""Same 480 batches, with added negative batches distributed over training."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import read_record, write_record
from scripts.vision import prepare_negative_rehearsal_control as prep
from scripts.vision import train_reactor_visibility_expansion as trainer

SOURCE = prep.OUT
OUT = SOURCE.parent/'negative-tail-interleaved-control-v1'
KEYS = tuple(f'interleaved-negative-480-{s}' for s in (7,17,27))

def reorder(seq):
    batches = [seq[i:i+6] for i in range(0,len(seq),6)]
    if len(batches)!=480 or any(len(b)!=6 for b in batches): raise ValueError('Batch count')
    base, tail = iter(batches[:460]), iter(batches[460:])
    result = [next(tail) if (i+1)*20//480 > i*20//480 else next(base) for i in range(480)]
    if Counter(map(tuple,result))!=Counter(map(tuple,batches)): raise ValueError('Batch multiset changed')
    return [m for b in result for m in b]

def verify(r):
    for p,h in r.get('inputs',{}).items():
        if file_sha256(p)!=h: raise ValueError('Stale dependency '+p)

def prepare():
    OUT.mkdir(exist_ok=True)
    dest=OUT/'protocol.json'
    if dest.exists():
        p=read_record(dest);verify(p);return p
    source=read_record(SOURCE/'protocol.json');verify(source)
    p={k:source[k] for k in ('pool_rows','names','initialization','training_config','evaluation')}
    p.update(schedules={},listings={},exposures={})
    for seed,key in zip((7,17,27),KEYS):
        old=f'negative-rehearsal-480-{seed}'
        p['schedules'][key]=reorder(source['schedules'][old])
        p['listings'][key]=source['listings'][old]
        p['exposures'][key]=prep._exposure(p['pool_rows'],p['schedules'][key])
        if p['exposures'][key]!=source['exposures'][old]: raise ValueError('Exposure changed')
    p.update(status='frozen',comparison='Identical members, multiplicities, 480 complete batches and within-batch order; move 20 final negative batches to every 24th batch.',
             training_admitted=False,promotable=False,inputs={str(x):file_sha256(x) for x in (SOURCE/'protocol.json',Path(__file__).resolve())})
    return write_record(dest,p)

def preflight():
    p=prepare();prep.OUT=OUT;prep.KEYS=KEYS;prep.freeze=lambda:p
    for row in p['pool_rows']:
        for kind in ('image','label'):
            if file_sha256(row[kind+'_path'])!=row[kind+'_sha256']:raise ValueError('Member changed')
    r=prep.preflight();verify(r)
    return r

def worker(key):
    preflight();trainer.OUT=OUT;trainer.KEYS=KEYS;trainer.STEPS=480
    trainer._worker(key)

def train():
    preflight()
    for start in (0,2):
        jobs=[]
        try:
            for key in KEYS[start:start+2]:
                root=OUT/'workers'/key;root.mkdir(parents=True,exist_ok=True)
                n=len(list(root.glob('attempt-*.log')))+1
                if n>3:raise ValueError('Attempt cap')
                log=(root/f'attempt-{n:03}.log').open('x')
                proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.negative_tail_interleaved_control','--worker',key],stdout=log,stderr=subprocess.STDOUT)
                jobs.append((proc,log));print('STARTED',key,proc.pid,flush=True)
            for proc,_ in jobs:
                if proc.wait()!=0:raise RuntimeError('Worker failed')
        finally:
            for proc,log in jobs:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                log.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--worker',choices=KEYS);a=ap.parse_args()
    if a.worker:worker(a.worker)
    elif a.train:train()
    else:preflight()
