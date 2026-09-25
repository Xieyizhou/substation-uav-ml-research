"""Capture four fixed same-frame lighting variants; no training."""
import argparse,asyncio,fcntl,shutil
from pathlib import Path
from scripts.vision.unified_hold_lighting_counts import OUT as DESIGN,prior
from scripts.vision import run_physical_lighting_capture_v2 as capture

OUT=DESIGN/'light-capture'

def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():prior.verify(prior.read(dest));return prior.read(dest)
    cp=DESIGN/'counts.json';c=prior.read(cp);prior.verify(c)
    if c['status']!='counts_verified_capture_pending':raise ValueError('Counts not ready')
    op=capture.OUT/'protocol.json';p=prior.read(op);prior.verify(p)
    frames=[f for f in p['frames'] if f['variant']=='physical-lighting' and f['pair_id'] in ('L01','L02','L03','L04')]
    paths=[cp,op,Path(__file__).resolve(),Path(capture.__file__)];OUT.mkdir(exist_ok=True)
    for f in frames:
        normal=capture.OUT/'replay'/f"{f['pair_id']}-original/attempt-01/receipt.json";r=prior.read(normal);prior.verify(r)
        if r['status']!='capture_technical_checks_passed':raise ValueError('Original alignment missing')
        paths += [normal,Path(f['source_world']),Path(f['source_plan'])]
    helper=OUT/'gz_visibility_capture_cleanup_fixed';shutil.copy2(capture.DEPTH/helper.name,helper);paths.append(helper)
    return prior.frozen(dest,dict(status='four_lighting_frames_frozen',frames=frames,training_ready=False,training_started=False,
        inputs={str(x):prior.file_sha256(x) for x in paths}))

async def run():
    p=freeze();base=capture.base;replay=base.replay;base.guard();results=[];paths=[OUT/'protocol.json']
    with (capture.BUILD/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=(base.OUT,base.VARIANT,replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command)
        base.OUT,base.VARIANT=capture.DEPTH,'depth'
        replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command=OUT,base.dual_sensor,base.validate_dual,capture.analyze,capture.PrefixAsync(),capture.clock.command
        try:
            for f in p['frames']:
                last=None
                for n in range(1,4):
                    capture.clock.CURRENT=OUT/'replay'/f['review_ids'][0]/f'attempt-{n:02}';rp=capture.clock.CURRENT/'receipt.json'
                    if rp.exists():last=prior.read(rp);prior.verify(last)
                    elif capture.clock.CURRENT.exists():continue
                    else:last=await replay.attempt(f,n)
                    if not last['process_cleanup_complete']:raise ValueError('Process not cleaned')
                    if last['status']!='technical_failure':break
                if last is None:raise ValueError('Attempt cap exhausted')
                paths.append(rp);results.append(dict(pair_id=f['pair_id'],receipt=str(rp),status=last['status'],reason=last.get('reason')))
                print(f['pair_id'],last['status'],last.get('reason'),flush=True)
                if last['status']!='capture_technical_checks_passed':break
        finally:(base.OUT,base.VARIANT,replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command)=old
    base.guard();prior.frozen(OUT/'receipt.json',dict(status='four_captured_review_pending' if len(results)==4 and all(x['status']=='capture_technical_checks_passed' for x in results) else 'blocked',
        results=results,training_ready=False,training_started=False,inputs={str(x):prior.file_sha256(x) for x in paths}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--capture',action='store_true');a=ap.parse_args()
    if a.capture:asyncio.run(run())
    else:freeze();print('FROZEN_NO_CAPTURE')
