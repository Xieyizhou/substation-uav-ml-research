"""Replay newly increased members; stop at first semantic conflict; never train."""
import argparse,asyncio,fcntl,shutil
from pathlib import Path
from scripts.vision import run_physical_lighting_capture_v2 as capture
from scripts.vision.prepare_l05_compensation_review import OUT as DESIGN,prior
from scripts.vision.prepare_physical_lighting_control import history

OUT=DESIGN/'increase-replay'

def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():prior.verify(prior.read(dest));return prior.read(dest)
    cp=DESIGN/'counts.json';c=prior.read(cp);prior.verify(c)
    if c['status']!='count_solution_quality_pending':raise ValueError('No valid count solution')
    ep=DESIGN/'evidence/manifest.json';e=prior.read(ep);prior.verify(e)
    already={x['member_id'] for x in e['events']};sources={};paths=[cp,ep,Path(__file__).resolve(),Path(capture.__file__)]
    for root in (history.CONTROL,history.SECOND):
        path=root/'evidence.json';data=prior.read(path);prior.verify(data);paths.append(path)
        for f in data['events']:
            mid=f['member']['member_id']
            if mid in sources:raise ValueError('Duplicate source mapping')
            sources[mid]=f
    OUT.mkdir(exist_ok=True);frames=[]
    for i,mid in enumerate(c['increase_review_queue'],1):
        if mid in already:continue
        f=sources[mid];plan=prior.read(f['source_plan']);rec=prior.read(f['source_receipt'])
        from scripts.vision.test_body_material_applicability import equal_rgb,label_correspondence,model_mapping
        from scripts.vision.structure_fit import truth_for
        from src.vision.canonical.gates import instance_mapping,annotation_mode_from_world
        equal_rgb(f['source_image'],f['member']['image_path']);label_correspondence(truth_for(f['member']),rec['truth']['objects'])
        if prior.file_sha256(Path(f['source_world']))!=plan['files']['world.sdf'] or annotation_mode_from_world(Path(f['source_world']))!='full_2d':raise ValueError('Invalid world')
        mapping={str(k):v for k,v in instance_mapping(plan).items()};wm,_=model_mapping(capture.base.ET.parse(f['source_world']))
        if len({v['object_id'] for v in mapping.values()})!=len(mapping) or any(wm.get(k)!=v['object_id'] for k,v in mapping.items()):raise ValueError('Mapping conflict')
        pid=f'I{i:02}'
        frames.append(dict(pair_id=pid,variant='original',review_ids=[pid+'-original'],member_id=mid,source_image=f['source_image'],
            source_receipt=f['source_receipt'],source_plan=f['source_plan'],source_world=f['source_world'],world_name=plan['world_name'],actual_pose=rec['actual_pose'],instance_mapping=mapping))
        paths += [Path(f[k]) for k in ('source_image','source_receipt','source_plan','source_world')]
    helper=OUT/'gz_visibility_capture_cleanup_fixed';shutil.copy2(capture.DEPTH/helper.name,helper);paths.append(helper)
    return prior.frozen(dest,dict(status='increase_replay_frozen',frames=frames,reused_original_members=sorted(already),training_ready=False,
        max_attempts=3,inputs={str(x):prior.file_sha256(x) for x in paths}))

async def run():
    p=freeze();base=capture.base;replay=base.replay;base.guard();results=[];paths=[OUT/'protocol.json']
    with (capture.BUILD/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=(capture.OUT,base.OUT,base.VARIANT,replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command)
        capture.OUT=OUT;base.OUT,base.VARIANT=capture.DEPTH,'depth'
        replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command=OUT,base.dual_sensor,base.validate_dual,capture.analyze,capture.PrefixAsync(),capture.clock.command
        try:
            for f in p['frames']:
                last=None
                for n in range(1,4):
                    capture.clock.CURRENT=OUT/'replay'/f['review_ids'][0]/f'attempt-{n:02}';rp=capture.clock.CURRENT/'receipt.json'
                    if rp.exists():last=prior.read(rp);prior.verify(last)
                    elif capture.clock.CURRENT.exists():continue
                    else:last=await replay.attempt(f,n)
                    if not last['process_cleanup_complete']:raise ValueError('Process cleanup incomplete')
                    if last['status']!='technical_failure':break
                if last is None:raise ValueError('No valid attempt in cap')
                paths.append(rp);results.append(dict(member_id=f['member_id'],pair_id=f['pair_id'],receipt=str(rp),status=last['status'],reason=last.get('reason')))
                print(f['pair_id'],last['status'],last.get('reason'),flush=True)
                if last['status']!='capture_technical_checks_passed':break
        finally:(capture.OUT,base.OUT,base.VARIANT,replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command)=old
    base.guard()
    prior.frozen(OUT/'receipt.json',dict(status='all_replayed_review_pending' if len(results)==len(p['frames']) and all(x['status']=='capture_technical_checks_passed' for x in results) else 'blocked',
        results=results,training_ready=False,training_started=False,inputs={str(x):prior.file_sha256(x) for x in paths}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if a.run:asyncio.run(run())
    else:freeze();print('FROZEN_NOT_RUNNING')
