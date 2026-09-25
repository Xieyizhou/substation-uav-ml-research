"""Bounded explicit clock warmup; missing seconds are never accepted as valid."""
import asyncio
import fcntl
import json
import math
import time
from pathlib import Path
from scripts.vision import run_depth_clip_test as experiment
from scripts.vision.run_depth_clip_test import OUT,BUILD,base,regression,PrefixAsync

CURRENT=None
ORIGINAL_COMMAND=base.replay.command


def valid_clock(message):
    try:
        stamp=message['header']['stamp'];sec=stamp.get('sec',stamp.get('seconds'))
        nano=stamp.get('nsec',stamp.get('nanoseconds',0))
        if sec is None or not all(math.isfinite(float(x)) for x in (sec,nano)):return False
        return float(sec)>=1 and float(sec).is_integer() and 0<=float(nano)<1e9 and float(nano).is_integer()
    except (KeyError,TypeError,ValueError,AttributeError):return False


async def command(*args,**kwargs):
    if args[:3]!=('gz','topic','-e') or not any(str(a).endswith('/pose/info') for a in args):
        return await ORIGINAL_COMMAND(*args,**kwargs)
    attempts=[];deadline=time.monotonic()+15
    for n in range(1,7):
        if time.monotonic()>=deadline:break
        raw=await ORIGINAL_COMMAND(*args,**dict(kwargs,timeout=min(5,max(.1,deadline-time.monotonic()))))
        message=json.loads(raw);accepted=valid_clock(message)
        attempts.append(dict(attempt=n,accepted=accepted,message=message))
        if accepted:
            base.frozen(CURRENT/'clock-preflight.json',dict(status='explicit_clock_ready',samples=attempts,
                policy='Reject missing seconds, wait for explicit sec>=1. Existing post-move fence and <=33.334ms checks unchanged.',
                training_ready=False,training_started=False))
            return raw
        await asyncio.sleep(.25)
    base.frozen(CURRENT/'clock-preflight.json',dict(status='clock_preflight_blocked',samples=attempts,training_ready=False,training_started=False))
    raise ValueError('No explicit valid clock within bounded preflight')


async def run():
    global CURRENT
    protocol=experiment.freeze();base.guard()
    prior=OUT/'replay/T020/attempt-01/receipt.json';base.verify(base.read(prior))
    if base.read(prior)['reason']!='Gazebo message timestamp is missing seconds':raise ValueError('Different original blocker')
    pp=OUT/'clock-fence-protocol.json'
    if pp.exists():base.verify(base.read(pp))
    else:base.frozen(pp,dict(status='bounded_clock_preflight_frozen',max_clock_samples=6,clock_deadline_seconds=15,
        selection='T020 attempt 02 retains failed attempt 01; remaining frozen poses attempt 01. No further automatic retry.',
        training_ready=False,training_started=False,inputs={str(p):base.file_sha256(p) for p in (OUT/'protocol.json',prior,Path(__file__))}))
    replay=base.replay
    with (BUILD/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=(base.OUT,base.VARIANT,replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command)
        base.OUT,base.VARIANT=OUT,'depth'
        replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command=OUT,base.dual_sensor,base.validate_dual,regression.analyze,PrefixAsync(),command
        paths=[pp];results=[]
        try:
            for f in protocol['frames']:
                tag=f['review_ids'][0];number=2 if tag=='T020' else 1
                CURRENT=OUT/'replay'/tag/f'attempt-{number:02}'
                rp=CURRENT/'receipt.json'
                if rp.exists():last=base.read(rp);base.verify(last)
                elif CURRENT.exists():raise ValueError('Incomplete attempt retained; no retry')
                else:last=await replay.attempt(f,number)
                paths.append(rp);results.append(dict(review_id=tag,receipt=str(rp),status=last['status']))
                if last['status']!='existing_pose_technical_checks_passed':break
        finally:(base.OUT,base.VARIANT,replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command)=old
        base.guard()
        dest=OUT/'fenced-run-receipt.json'
        if dest.exists():base.verify(base.read(dest));return
        base.frozen(dest,dict(status='runs_complete' if len(results)==4 and all(r['status']=='existing_pose_technical_checks_passed' for r in results) else 'runs_incomplete',
            results=results,training_ready=False,training_started=False,inputs={str(p):base.file_sha256(p) for p in paths}))


if __name__=='__main__':asyncio.run(run())
