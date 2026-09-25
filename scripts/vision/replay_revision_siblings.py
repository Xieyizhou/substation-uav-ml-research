"""Two preselected saved frames; isolated depth renderer, no training writes."""
import argparse
import asyncio
import fcntl
import shutil
from pathlib import Path
from scripts.vision.trace_revision_sources import OUT as SOURCE
from scripts.vision import run_depth_clip_clock_fence as clock
from scripts.vision.run_depth_clip_test import OUT as DEPTH, BUILD, base, regression, PrefixAsync

OUT=SOURCE/'sibling-replay-v1'


def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():base.verify(base.read(dest));return base.read(dest)
    paths=[SOURCE/'evidence.json',SOURCE/'review.json',DEPTH/'completion.json',DEPTH/'protocol.json',Path(__file__),Path(clock.__file__),Path(regression.__file__)]
    for p in paths[:4]:base.verify(base.read(p))
    OUT.mkdir(parents=True,exist_ok=True);frames=[]
    for r in base.read(paths[0])['members']:
        if r['variant'] not in ('original','background_bridge'):continue
        cp=Path(r['source_receipt']);receipt=base.read(cp)
        vid=r['member_id'].split(':')[1];view=next(v for v in receipt['views'] if v['view_id']==vid)
        ip=cp.parent/vid/'rgb.ppm';wp=Path(r['source_world']);pp=wp.parent/'plan.json';plan=base.read(pp)
        sp=OUT/(r['variant']+'-source.json');base.frozen(sp,dict(**view,inputs={str(cp):base.file_sha256(cp)}))
        frames.append(dict(member_id=r['member_id'],review_ids=['T027-'+r['variant']],source_receipt=str(sp),source_image=str(ip),
            source_world=str(wp),source_plan=str(pp),world_name=plan['world_name'],actual_pose=view['actual_pose'],
            instance_mapping=receipt['collection_checks']['instance_mapping']))
        paths += [cp,ip,wp,pp,sp,wp.parent/'obstacles.json']
    if len(frames)!=2:raise ValueError('Two frozen siblings required')
    helper=OUT/'gz_visibility_capture_cleanup_fixed';shutil.copy2(DEPTH/'gz_visibility_capture_cleanup_fixed',helper)
    paths += [helper,DEPTH/'diagnostic-server',DEPTH/'variants/depth/libgz-rendering8-ogre2.8.2.3.dylib']
    return base.frozen(dest,dict(status='two_saved_sibling_frames_frozen',frames=frames,max_attempts=3,
        training_ready=False,training_started=False,historical_labels_changed=False,
        inputs={str(p):base.file_sha256(p) for p in paths}))


async def run():
    protocol=freeze();base.guard();replay=base.replay
    with (BUILD/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=(base.OUT,base.VARIANT,replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command)
        base.OUT,base.VARIANT=DEPTH,'depth'
        replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command=OUT,base.dual_sensor,base.validate_dual,regression.analyze,PrefixAsync(),clock.command
        results=[];paths=[OUT/'protocol.json']
        try:
            for frame in protocol['frames']:
                for n in range(1,4):
                    clock.CURRENT=OUT/'replay'/frame['review_ids'][0]/f'attempt-{n:02}'
                    rp=clock.CURRENT/'receipt.json'
                    if rp.exists():last=base.read(rp);base.verify(last)
                    elif clock.CURRENT.exists():raise ValueError('Incomplete attempt retained; manual investigation required')
                    else:last=await replay.attempt(frame,n)
                    paths.append(rp)
                    if not last['process_cleanup_complete']:raise ValueError('Cleanup incomplete')
                    if last['status']!='technical_failure':break
                results.append(dict(review_id=frame['review_ids'][0],receipt=str(rp),status=last['status']))
                if last['status']!='existing_pose_technical_checks_passed':break
        finally:(base.OUT,base.VARIANT,replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command)=old
        base.guard();dest=OUT/'run-receipt.json'
        if dest.exists():base.verify(base.read(dest));return
        base.frozen(dest,dict(status='runs_complete' if len(results)==2 and all(r['status']=='existing_pose_technical_checks_passed' for r in results) else 'runs_blocked',
            results=results,training_ready=False,training_started=False,inputs={str(p):base.file_sha256(p) for p in paths}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    if args.replay:asyncio.run(run())
    else:freeze();print('FROZEN_NO_RENDER')
