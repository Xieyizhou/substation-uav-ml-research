"""Three pre-existing pilot poses; original versus isolated clipped renderer."""
import argparse
import asyncio
import fcntl
import shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageChops
from scripts.vision import run_near_clip_build_test as base
from scripts.vision.run_near_clip_prefix_isolation import PrefixAsync,prepare as prepare_loader

OUT=base.OUT/'edge-raster-validation-v1/pose-regression-v1'


def freeze():
    path=OUT/'protocol.json'
    base.guard();base.verify(base.read(base.OUT/'completion.json'));base.verify(base.read(base.SOURCE/'protocol.json'))
    if path.exists():base.verify(base.read(path));return base.read(path)
    frames=[f for f in base.read(base.SOURCE/'protocol.json')['frames'] if f['review_ids']!=['T027']]
    if [f['review_ids'][0] for f in frames]!=['T020','T036','T023']:raise ValueError('Unexpected predefined pilot membership')
    paths=[base.OUT/'completion.json',base.SOURCE/'protocol.json',Path(__file__),Path(base.__file__),base.ROOT/'scripts/vision/run_near_clip_prefix_isolation.py']
    for f in frames:
        base.verify(base.read(f['source_receipt']))
        paths += [Path(f[k]) for k in ('source_receipt','source_world','source_plan','source_image')]
    OUT.mkdir(parents=True,exist_ok=True)
    for variant in ('control','clipped'):
        prepare_loader(variant)
        paths.append(base.OUT/'prefix-isolated'/variant/'protocol.json')
        dest=OUT/variant;dest.mkdir()
        shutil.copy2(base.DUAL/'gz_visibility_capture_cleanup_fixed',dest/'gz_visibility_capture_cleanup_fixed')
        paths.append(dest/'gz_visibility_capture_cleanup_fixed')
    p=base.frozen(path,dict(status='existing_pose_regression_frozen',frames=frames,max_attempts=2,
        scope='All three remaining previously fixed closed-body pilot poses, no replacement, no new training images, no labels changed. Registered poses are not independent scenes.',
        training_ready=False,training_started=False,inputs={str(p):base.file_sha256(p) for p in paths}))
    for variant in ('control','clipped'):
        base.frozen(OUT/variant/'protocol.json',dict(status='existing_pose_variant_frozen',variant=variant,frames=frames,
            training_ready=False,training_started=False,inputs={str(path):base.file_sha256(path)}))
    return p


def analyze(folder,frame,fence):
    selected=None
    for start in range(1,11):
        signs=[tuple(base.file_sha256(folder/f'frame-{n}-{k}.bin') for k in ('rgb','mask')) for n in range(start,start+3)]
        if len(set(signs))==1:selected=list(range(start,start+3));break
    if selected is None:raise ValueError('No stable RGB/mask window')
    libraries=[s.split('DIAGNOSTIC_LOADED_LIBRARY ',1)[1].strip() for s in (folder/'simulator.log').read_text().splitlines() if 'DIAGNOSTIC_LOADED_LIBRARY ' in s]
    engines=[Path(s) for s in libraries if 'libgz-rendering8-ogre2' in s]
    expected=base.OUT/'variants'/base.VARIANT/'libgz-rendering8-ogre2.8.2.3.dylib'
    if len(engines)!=1 or engines[0].resolve()!=expected.resolve():raise ValueError('Wrong plugin loaded')
    original=Image.open(frame['source_image']).convert('RGB');mapping=frame['instance_mapping']
    old=base.box_map(base.read(frame['source_receipt'])['raw_truth'],mapping)
    config=base.read(Path(frame['source_plan']).parent/'obstacles.json')
    camera=base.ET.parse(frame['source_world']).find(".//model[@name='canonical_camera']/link[@name='research_camera_link']")
    offset=list(map(float,camera.findtext('pose').split()))[:3]
    rows=[]
    for n in selected:
        metadata={k:base.read(folder/f'frame-{n}-{k}.json') for k in ('rgb','depth','mask','pose','boxes','visible-boxes')}
        stamps=[base.message_timestamp(m) for m in metadata.values()];skew=max(abs(t-stamps[0]) for t in stamps)*1000
        if min(stamps)<=fence or skew>33.334001:raise ValueError('Time gate failed')
        pose=base.pose_record(metadata['pose'])
        if not pose or not base.pose_close(pose,frame['actual_pose']):raise ValueError('Pose gate failed')
        base.validate_point(pose['position'],config,role='actual carrier')
        base.validate_point((np.array(pose['position'])+base.rotate(pose['orientation'],offset)).tolist(),config,role='actual optical')
        for k,w,h,fmt,size in [('rgb',1920,1080,'RGB_INT8',3),('mask',1920,1080,'RGB_INT8',3),('depth',640,360,'R_FLOAT32',4)]:
            m=metadata[k]
            if (m['width'],m['height'],m.get('pixelFormatType'))!=(w,h,fmt) or (folder/f'frame-{n}-{k}.bin').stat().st_size!=w*h*size:raise ValueError('Sensor format changed')
        rgb=np.frombuffer((folder/f'frame-{n}-rgb.bin').read_bytes(),dtype='u1').reshape(1080,1920,3)
        mask=np.frombuffer((folder/f'frame-{n}-mask.bin').read_bytes(),dtype='u1').reshape(1080,1920,3)
        labels=base.check_mask(metadata['mask'],mask,mapping)
        image=Image.fromarray(rgb);image.save(folder/f'frame-{n}-rgb.png');ImageChops.difference(original,image).save(folder/f'frame-{n}-difference.png')
        if original.tobytes()!=image.tobytes():raise ValueError('Original RGB pixel mismatch')
        boxes=base.box_map(metadata['boxes'],mapping);visible=base.parse_visible(metadata['visible-boxes'],mapping)
        lost=sorted(set(old)-set(boxes));added=sorted(set(boxes)-set(old))
        delta={k:max(abs(a-b) for a,b in zip(old[k],boxes[k])) for k in set(old)&set(boxes)}
        if base.VARIANT=='control' and (lost or added or max(delta.values(),default=0)>1):raise ValueError('Rebuilt control differs')
        mapped=[str(k) for k in labels if k not in (0,255)]
        rows.append(dict(capture_index=n,rgb_exact=True,skew_ms=skew,actual_pose=pose,full_boxes=boxes,visible_boxes=visible,
            historical_deltas=delta,lost_labels=lost,added_labels=added,unboxed_visible_labels=sorted(set(mapped)-set(boxes)),
            mask_sha256=base.file_sha256(folder/f'frame-{n}-mask.bin')))
    if any(r['full_boxes']!=rows[0]['full_boxes'] or r['visible_boxes']!=rows[0]['visible_boxes'] for r in rows):raise ValueError('Unstable boxes')
    return dict(status='existing_pose_technical_checks_passed',records=rows,selected_capture_indices=selected,loaded_libraries=libraries,
        training_ready=False,training_started=False,reason='Label changes are reported for independent comparison, not approved as training annotations.')


async def run():
    protocol=freeze();replay=base.replay
    with (base.OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=(replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio)
        paths=[OUT/'protocol.json'];results=[]
        try:
            for variant in ('control','clipped'):
                base.VARIANT=variant
                replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio=OUT/variant,base.dual_sensor,base.validate_dual,analyze,PrefixAsync()
                for frame in protocol['frames']:
                    last=None
                    for n in (1,2):
                        rp=OUT/variant/'replay'/frame['review_ids'][0]/f'attempt-{n:02}'/'receipt.json'
                        if rp.exists():last=base.read(rp);base.verify(last)
                        elif rp.parent.exists():continue
                        else:last=await replay.attempt(frame,n)
                        paths.append(rp)
                        if last['status']!='technical_failure':break
                    results.append(dict(variant=variant,review_id=frame['review_ids'][0],status=last['status'] if last else 'incomplete_attempts'))
                    if last is None or last['status']!='existing_pose_technical_checks_passed':break
                if len(results)%3:break
        finally:(replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio)=old
        base.guard()
        dest=OUT/'run-receipt.json'
        if dest.exists():base.verify(base.read(dest));return
        base.frozen(dest,dict(status='runs_complete' if len(results)==6 and all(r['status']=='existing_pose_technical_checks_passed' for r in results) else 'runs_incomplete',
            results=results,training_ready=False,training_started=False,inputs={str(p):base.file_sha256(p) for p in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');args=ap.parse_args()
    if args.replay:asyncio.run(run())
    else:freeze();print('THREE_EXISTING_POSES_FROZEN_NO_RENDER')
