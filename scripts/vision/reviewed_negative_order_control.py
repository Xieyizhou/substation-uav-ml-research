"""Whole-frame holds followed by a fresh, paired batch-order experiment.

Default performs only genuine loader preflight. Explicit --train runs six
independent units, never resuming an unverified intermediate checkpoint.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
from pathlib import Path
import subprocess
import sys
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import read_record, write_record
from scripts.vision import prepare_negative_rehearsal_control as prep
from scripts.vision import train_reactor_visibility_expansion as trainer
from scripts.vision.record_pilot_full_label_audit import BASE
from scripts.vision.negative_tail_interleaved_control import reorder

OUT=BASE/'reviewed-negative-order-control-v1'
SOURCE=BASE/'negative-rehearsal-control-v2/protocol.json'
REVIEW=BASE/'pilot-full-label-replay-v1/mask-review.json'
KEYS=tuple(f'{arm}-480-{seed}' for seed in (7,17,27) for arm in ('reviewed-tail','reviewed-interleaved'))

def verify(record):
    for path,digest in record.get('inputs',{}).items():
        if file_sha256(path)!=digest:raise ValueError('Changed dependency '+path)

def compensated(seq,held,clear,seed):
    if len(seq)!=2880 or len(clear)!=4 or len(set(clear))!=4:raise ValueError('Population mismatch')
    ranked=sorted(clear,key=lambda x:hashlib.sha256(f'reviewed-negative-order-v1|{seed}|{x}'.encode()).hexdigest())
    result=list(seq);positions=[i for i,m in enumerate(seq) if m in held]
    if len(positions)!=40 or any(i<2700 or i>=2760 for i in positions):raise ValueError('Held positions changed')
    for j,i in enumerate(positions):result[i]=ranked[j%4]
    if any(m in held for m in result):raise ValueError('Held member remains')
    if result[:2700]!=seq[:2700] or result[2760:]!=seq[2760:]:raise ValueError('Historical/negative exposure changed')
    if any(result.count(m)!=15 for m in clear):raise ValueError('Unbalanced cleared-frame exposure')
    return result

def loader(p,key,owner):
    from scripts.vision.order_retention_runtime import make_dataset,make_loader
    dataset=make_dataset(p,key)
    return dataset,make_loader(dataset,p,key,owner)

def freeze():
    path=OUT/'protocol.json'
    if path.exists():
        p=read_record(path);verify(p);return p
    source=read_record(SOURCE);verify(source)
    review=read_record(REVIEW);verify(review)
    if review['status']!='complete_with_eight_whole_frame_holds':raise ValueError('Review incomplete')
    held={'extreme:'+v for v in review['held_view_ids']};clear=['extreme:'+v for v in review['clear_view_ids']]
    rows=[r for r in source['pool_rows'] if r['member_id'] not in held]
    if len(rows)!=380:raise ValueError('Expected 376 historical plus 4 cleared pilot frames')
    for r in rows:
        for kind in ('image','label'):
            if file_sha256(r[kind+'_path'])!=r[kind+'_sha256']:raise ValueError('Member changed')
        if r['member_id'] in clear:
            from scripts.vision.prepare_reactor_visibility_training import _label_text
            if Path(r['label_path']).read_text()!=_label_text(r['truth']):raise ValueError('Exported full label differs from source truth')
    OUT.mkdir(parents=True,exist_ok=True);schedules={};exposures={};listings={};windows={};paths=[SOURCE,REVIEW,Path(__file__),Path(prep.__file__),Path(trainer.__file__)]
    from scripts.vision import order_retention_runtime as runtime
    paths.append(Path(runtime.__file__))
    from scripts.vision import negative_tail_interleaved_control as order
    paths.append(Path(order.__file__))
    for seed in (7,17,27):
        seq=compensated(source['schedules'][f'negative-rehearsal-480-{seed}'],held,clear,seed)
        shuffled=reorder(seq)
        if Counter(seq)!=Counter(shuffled):raise ValueError('Paired member exposures differ')
        for arm,values in [('reviewed-tail',seq),('reviewed-interleaved',shuffled)]:
            key=f'{arm}-480-{seed}';schedules[key]=values;exposures[key]=prep._exposure(rows,values)
            listing=OUT/f'{key}.txt';by={r['member_id']:r for r in rows}
            with listing.open('x') as handle:handle.write(''.join(by[m]['image_path']+'\n' for m in sorted(set(values))))
            paths.append(listing);listings[key]=str(listing.resolve())
            windows[key]=[dict(first_step=i//6+1,last_step=min(i+300,len(values))//6,**prep._exposure(rows,values[i:i+300])) for i in range(0,len(values),300)]
        if exposures[f'reviewed-tail-480-{seed}']!=exposures[f'reviewed-interleaved-480-{seed}']:raise ValueError('Paired exposure ledger differs')
    p={k:source[k] for k in ('names','initialization','training_config','evaluation')}
    import torch,ultralytics,platform
    p.update(pool_rows=rows,schedules=schedules,exposures=exposures,listings=listings,windows=windows,
        held_members=sorted(held),cleared_pilot_members=clear,status='frozen_preflight_pending',
        design='Six fresh units, two arms x three seeds. Same 480 intact batches and member counts within seed. Only batch order differs between arms. The 8 held frames are replaced by 4 cleared frames, each exposed 15 times.',
        limits='New supervision policy changes member and class distribution versus history; historical results are context, not an order-only control. Four pilot views share layout/assets and do not establish independent scene diversity. No model-size/learning-rate/threshold changes.',
        configuration=dict(steps=480,batch=6,nbs=6,cpu_threads_per_worker=4,max_parallel_workers=2,lr=.0005,augmentation='off',initialization='independent_v2.11',checkpoint='last.pt'),
        environment=dict(python=sys.version,torch=torch.__version__,ultralytics=ultralytics.__version__,platform=platform.platform()),
        training_admitted=False,promotable=False,inputs={str(x.resolve()):file_sha256(x) for x in paths})
    return write_record(path,p)

def preflight():
    p=freeze();verify(read_record(REVIEW))
    for row in p['pool_rows']:
        for kind in ('image','label'):
            if file_sha256(row[kind+'_path'])!=row[kind+'_sha256']:raise ValueError('Member changed')
    prep.OUT=OUT.resolve();prep.KEYS=KEYS;prep.freeze=lambda:p;trainer._loader=loader
    from unittest.mock import patch
    from ultralytics import YOLO
    with patch.object(YOLO,'train',side_effect=AssertionError('Training forbidden')),patch.object(YOLO,'val',side_effect=AssertionError('Validation forbidden')):
        ready=prep.preflight()
    verify(ready)
    for key in KEYS:verify(read_record(OUT/'actual-preflight'/f'{key}.json'))
    return p,ready

def worker(key):
    p,ready=preflight()
    trainer.OUT=OUT.resolve();trainer.KEYS=KEYS;trainer.STEPS=480;trainer._loader=loader
    def checked_ready():
        verify(p);verify(ready)
        if file_sha256(p['initialization']['path'])!=p['initialization']['sha256']:raise ValueError('Initialization changed')
        return p,ready
    trainer._ready=checked_ready
    existing=OUT/'training'/key/'completion.json'
    if existing.exists():
        r=read_record(existing);verify(r);verify(read_record(r['exposure_path']))
        if read_record(r['exposure_path'])['actual']!=p['schedules'][key]:raise ValueError('Completed exposure drift')
    return trainer._worker(key)

def train():
    preflight();root=OUT/'runner';root.mkdir(exist_ok=True)
    with (root/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for start in range(0,len(KEYS),2):
            jobs=[]
            try:
                for key in KEYS[start:start+2]:
                    folder=root/key;folder.mkdir(exist_ok=True);n=len(list(folder.glob('attempt-*.log')))+1
                    if n>3:raise ValueError('Runner attempt cap')
                    log=(folder/f'attempt-{n:03}.log').open('x')
                    proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.reviewed_negative_order_control','--worker',key],stdout=log,stderr=subprocess.STDOUT)
                    jobs.append((proc,log,key));print('TRAINING_STARTED',key,proc.pid,flush=True)
                for proc,_,key in jobs:
                    proc.wait(timeout=14400)
                    if proc.returncode:raise RuntimeError('Worker failed '+key)
            finally:
                for proc,log,_ in jobs:
                    if proc.poll() is None:
                        proc.terminate()
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    log.close()
        write_record(root/'completion.json',dict(status='six_units_complete_evaluation_pending',training_admitted=False,promotable=False,inputs={str((OUT/'training'/k/'completion.json').resolve()):file_sha256(OUT/'training'/k/'completion.json') for k in KEYS}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',action='store_true');p.add_argument('--worker',choices=KEYS);p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.worker:worker(a.worker)
    elif a.train:train()
    elif a.freeze:print(freeze()['status'])
    else:preflight()
