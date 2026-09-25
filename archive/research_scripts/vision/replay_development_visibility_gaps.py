"""Ten named development gaps: source trace and bounded same-frame replay."""
import argparse,asyncio,fcntl,re,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.vision.evaluate_unified_lighting import OUT as RUN,prior
from scripts.vision import run_physical_lighting_capture_v2 as capture
from scripts.vision.instance_visibility_diagnosis import raw_box
from src.vision.canonical.gates import instance_mapping
from src.ml.artifacts import object_sha256

OUT=prior.ROOT/'data/research/ml_training_recovery_v1/development-visibility-gap-check-v1'
BASE=prior.ROOT/'data/research/ml_training_recovery_v1/paired-visual-factors-v1'

def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():prior.verify(prior.read(dest));return prior.read(dest)
    ep,rp=RUN/'error-review/evidence.json',RUN/'error-review/review.json';e,r=prior.read(ep),prior.read(rp)
    for x in (e,r):prior.verify(x)
    wanted={x.split(':')[0] for x in r['pending_ids']};events=[x for x in e['events'] if x['event_id'] in wanted]
    if len(events)!=10:raise ValueError('Expected ten frozen gaps')
    progress=prior.read(BASE/'capture-progress.json');runs={x['variant']:x for x in progress['runs']};frames={};paths=[ep,rp,BASE/'capture-progress.json',Path(__file__).resolve()]
    OUT.mkdir(parents=True,exist_ok=True)
    for ev in events:
        src=ev['source'];key=(src['view_id'],src['variant'])
        if key not in frames:
            pp=BASE/src['variant']/'plan/plan.json';wp=pp.parent/'world.sdf';p=prior.read(pp)
            cr=Path(runs[src['variant']]['receipt_path']);coll=prior.read(cr);views=[x for x in coll['views'] if x['view_id']==src['view_id']]
            if len(views)!=1:raise ValueError('Ambiguous collection view')
            v=views[0];ip=Path(src['image_path']);single=cr.parent/(src['view_id']+'.json')
            if not single.exists():single=ip.parent.parent/(src['view_id']+'.json')
            record=prior.read(single)
            if record['raw_truth']!=v['raw_truth'] or record['actual_pose']!=v['actual_pose']:raise ValueError('Per-view and collection receipt disagree')
            if prior.file_sha256(ip)!=src['image_sha256'] or v['image_sha256']!=src['image_sha256']:raise ValueError('RGB identity mismatch')
            if object_sha256(v['truth'])!=src['truth_sha256']:raise ValueError('Truth identity mismatch')
            if prior.file_sha256(wp)!=p['files']['world.sdf'] or coll['world_sha256']!=p['files']['world.sdf']:raise ValueError('Loaded world mismatch')
            mapping={str(k):x for k,x in instance_mapping(p).items()}
            if mapping!=coll['collection_checks']['instance_mapping']:raise ValueError('Recorded mapping differs')
            if len({x['object_id'] for x in mapping.values()})!=len(mapping):raise ValueError('Mapping collision')
            tag=ev['event_id']+'-'+src['variant']
            frames[key]=dict(pair_id=tag,variant='original',source_visual_condition=src['variant'],member_id=src['image_sha256'],review_ids=[tag],
                source_image=str(ip),source_receipt=str(single),source_world=str(wp),source_plan=str(pp),world_name=p['world_name'],
                actual_pose=v['actual_pose'],instance_mapping=mapping,events=[],source_status='source_identity_verified',collection_receipt=str(cr),
                annotation_mode=coll['actual_annotation_mode'],historical_skew_ms=v['skew_ms'])
            paths += [pp,wp,pp.parent/'obstacles.json',cr,single,ip,Path(v['depth_path'])]
            if prior.file_sha256(v['depth_path'])!=v['depth_sha256']:raise ValueError('Depth identity mismatch')
        f=frames[key];truth=ev['loss']['truth'];label=re.search(r'-instance-(\d+)-',truth['annotation_id']).group(1);label=str(int(label))
        if label not in f['instance_mapping'] or f['instance_mapping'][label]['category']!=truth['class_name']:raise ValueError('Target mapping mismatch')
        raw=[b for b in prior.read(f['source_receipt'])['raw_truth']['annotatedBox'] if int(b['label'])==int(label)]
        if len(raw)!=1 or max(abs(a-b) for a,b in zip(raw_box(raw[0]),truth['bbox_xyxy']))>1e-4:raise ValueError('Target raw box mismatch')
        f['events'].append(dict(event_id=ev['event_id'],runtime_label=label,object_id=f['instance_mapping'][label]['object_id'],truth=truth))
    helper=OUT/'gz_visibility_capture_cleanup_fixed';shutil.copy2(capture.DEPTH/helper.name,helper)
    paths += [helper,capture.DEPTH/'protocol.json',Path(capture.__file__).resolve(),Path(capture.clock.__file__).resolve()]
    return prior.frozen(dest,dict(status='ten_gaps_eight_frames_source_frozen',frames=list(frames.values()),
        scope='Existing development images only; original means exact replay of the selected source including its lighting, not reset to baseline lighting.',
        policy=dict(max_attempts=3,exact_rgb_required=True,stable_frames=3,max_skew_ms=33.334,max_box_delta_px=1),
        inputs={str(x):prior.file_sha256(x) for x in paths}))

def analyze(folder,frame,fence):
    result=capture.analyze(folder,frame,fence);targets=[]
    for record in result['records']:
        i=record['capture_index'];mask=np.frombuffer((folder/f'frame-{i}-mask.bin').read_bytes(),dtype='u1').reshape(1080,1920,3)
        rgb=np.asarray(Image.open(folder/f'frame-{i}-rgb.png').convert('RGB'))
        for e in frame['events']:
            selected=mask[:,:,2]==int(e['runtime_label']);ys,xs=np.where(selected)
            targets.append(dict(event_id=e['event_id'],frame=i,visible_pixels=int(selected.sum()),
                visible_bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None,component_evidence='unknown'))
            overlay=rgb.copy();overlay[selected]=(overlay[selected]*.5+np.array([255,0,255])*.5).astype('u1')
            Image.fromarray(overlay).save(folder/f'{i}-{e["event_id"]}-overlay.png')
    result['targets']=targets;return result

async def run():
    p=freeze();base=capture.base;replay=base.replay;base.guard();results=[];paths=[OUT/'protocol.json']
    with (capture.BUILD/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=(base.OUT,base.VARIANT,replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command)
        base.OUT,base.VARIANT=capture.DEPTH,'depth'
        replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command=OUT,base.dual_sensor,base.validate_dual,analyze,capture.PrefixAsync(),capture.clock.command
        try:
            for frame in p['frames']:
                last=None
                for n in range(1,4):
                    capture.clock.CURRENT=OUT/'replay'/frame['review_ids'][0]/f'attempt-{n:02}';rp=capture.clock.CURRENT/'receipt.json'
                    if rp.exists():last=prior.read(rp);prior.verify(last)
                    elif capture.clock.CURRENT.exists():continue
                    else:last=await replay.attempt(frame,n)
                    paths.append(rp)
                    if not last['process_cleanup_complete']:raise ValueError('Cleanup incomplete')
                    if last['status']!='technical_failure':break
                results.append(dict(events=[x['event_id'] for x in frame['events']],receipt=str(rp),status=last['status'] if last else 'attempts_exhausted',reason=last.get('reason') if last else 'Incomplete attempts'))
                if last is None or last['status']!='capture_technical_checks_passed':break
        finally:(base.OUT,base.VARIANT,replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command)=old
    base.guard()
    prior.frozen(OUT/'replay-receipt.json',dict(status='replays_complete_review_pending' if len(results)==len(p['frames']) and all(x['status']=='capture_technical_checks_passed' for x in results) else 'replay_path_blocked',
        results=results,not_attempted=[e['event_id'] for f in p['frames'][len(results):] for e in f['events']],
        inputs={str(x):prior.file_sha256(x) for x in paths}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');args=ap.parse_args()
    if args.replay:asyncio.run(run())
    else:print(freeze()['status'])
