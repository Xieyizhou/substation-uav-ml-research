"""Correct missing unversioned plugin alias; retain the rejected launch unchanged."""
import argparse
import asyncio
import fcntl
import shutil
from pathlib import Path
from scripts.vision import run_near_clip_build_test as base


async def run(variant):
    p=base.freeze();base.VARIANT=variant
    dest=base.OUT/'verified-loader'/variant
    for v in ('control','clipped'):
        alias=base.OUT/'variants'/v/'libgz-rendering-ogre2.dylib'
        if not alias.exists():alias.symlink_to('libgz-rendering8-ogre2.8.2.3.dylib')
    dest.mkdir(parents=True,exist_ok=True)
    helper=dest/'gz_visibility_capture_cleanup_fixed'
    if not helper.exists():shutil.copy2(base.DUAL/'gz_visibility_capture_cleanup_fixed',helper)
    protocol=dest/'protocol.json'
    paths=[base.OUT/'protocol.json',Path(__file__),helper,base.OUT/'variants'/variant/'libgz-rendering-ogre2.dylib']
    if protocol.exists():base.verify(base.read(protocol))
    else:base.frozen(protocol,dict(status='loader_alias_corrected_before_replay',variant=variant,frames=p['frames'],
        correction='Original engine plugin alias is gz-rendering-ogre2, without major version. Failed original launch preserved; actual loaded path must still pass.',
        max_attempts=2,training_ready=False,training_started=False,inputs={str(p):base.file_sha256(p) for p in paths}))
    replay=base.replay
    with (base.OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=(replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio)
        replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio=dest,base.dual_sensor,base.validate_dual,base.analyze,base.IsolatedAsync()
        paths=[protocol];last=None
        try:
            for number in range(1,3):
                rp=dest/'replay/T027'/f'attempt-{number:02}'/'receipt.json'
                if rp.exists():last=base.read(rp);base.verify(last)
                elif rp.parent.exists():continue
                else:last=await replay.attempt(p['frames'][0],number)
                paths.append(rp)
                if last['status']!='technical_failure':break
        finally:(replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio)=old
        base.guard()
        path=dest/'completion.json'
        if path.exists():base.verify(base.read(path))
        else:base.frozen(path,dict(status=last['status'] if last else 'incomplete_attempts',training_ready=False,training_started=False,
            inputs={str(p):base.file_sha256(p) for p in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replay',required=True,choices=['control','clipped']);args=ap.parse_args()
    asyncio.run(run(args.replay))
