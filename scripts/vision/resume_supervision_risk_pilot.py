"""Explicit second pilot budget after user approval; no training or relabeling."""
import argparse
import asyncio
import os
from pathlib import Path
import shutil
import socket
from scripts.vision.supervision_risk_revision import OUT as PRIOR,ROOT,read,verify,verify_tree,frozen,file_sha256,replay

OUT=PRIOR.parent/'supervision-risk-permission-replay-v1'

def probe_loopback():
    for kind in (socket.SOCK_DGRAM,socket.SOCK_STREAM):
        with socket.socket(socket.AF_INET,kind) as sock:sock.bind(('127.0.0.1',0))
    return {'udp_loopback_bind':True,'tcp_loopback_bind':True,'external_network_contacted':False}

def freeze():
    if (OUT/'protocol.json').exists():verify_tree(OUT/'protocol.json');return
    verify_tree(PRIOR/'completion.json');p=read(PRIOR/'protocol.json');OUT.mkdir(exist_ok=True)
    binary=PRIOR/'gz_visibility_capture_cleanup_fixed';dest=OUT/binary.name;shutil.copyfile(binary,dest);dest.chmod(0o755)
    paths=[PRIOR/'completion.json',PRIOR/'protocol.json',PRIOR/'review.json',binary,dest,Path(__file__),ROOT/'scripts/vision/run_visibility_cleanup_validation.py']
    frozen(OUT/'protocol.json',dict(status='new_bounded_pilot_budget_frozen',
        authorization='User said 好的，下一步吧 after report required permission resolution and confirmation of a new bounded pilot budget; scope announced as T30/T08 each at most 3 new attempts.',
        prior_attempts_preserved=6,frames=[f for f in p['frames'] if f['event_id'] in ('T30','T08')],policy=p['policy'],
        expansion_to_other_frames=False,labels_modified=False,training_started=False,
        inputs={str(x):file_sha256(x) for x in paths}))
    print('NEW_PILOT_BUDGET_FROZEN',flush=True)

async def unit(f):
    for n in range(1,4):
        path=OUT/'replay'/f['event_id']/f'attempt-{n:02}'/'receipt.json'
        if path.exists():verify_tree(path);r=read(path)
        elif path.parent.exists():continue
        else:r=await replay.attempt(f,n)
        if not r['process_cleanup_complete']:raise ValueError('Process cleanup incomplete')
        if r['status']!='technical_failure':return r
        # Systematic permission errors stop immediately rather than consume another attempt.
        log=path.parent/'simulator.log'
        if log.exists() and 'Operation not permitted' in log.read_text():return r
    return dict(status='technical_attempts_exhausted',reason='This newly approved budget of three attempts is exhausted')

async def run():
    p=read(OUT/'protocol.json');verify(p);verify(read(PRIOR/'protocol.json'))
    preflight=probe_loopback() # no attempt directory is created on a permission failure
    folder=OUT/'environment';folder.mkdir(exist_ok=True)
    path=folder/f'probe-{len(list(folder.glob("probe-*.json")))+1:03}.json'
    frozen(path,dict(status='loopback_available',**preflight,inputs={str(OUT/'protocol.json'):file_sha256(OUT/'protocol.json')}))
    old_out=replay.OUT;old_log=os.environ.get('GZ_LOG_PATH');logdir=OUT/'runtime-logs';logdir.mkdir(exist_ok=True)
    lock=OUT/'replay.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    replay.OUT=OUT;os.environ['GZ_LOG_PATH']=str(logdir)
    try:
        for f in p['frames']:
            r=await unit(f);print('PILOT_RESULT',f['event_id'],r['status'],flush=True)
    finally:
        replay.OUT=old_out
        if old_log is None:os.environ.pop('GZ_LOG_PATH',None)
        else:os.environ['GZ_LOG_PATH']=old_log
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--replay',action='store_true');args=ap.parse_args()
    if args.freeze:freeze()
    elif args.replay:asyncio.run(run())
    else:print('PREFLIGHT_ONLY_NO_REPLAY_NO_TRAINING')
