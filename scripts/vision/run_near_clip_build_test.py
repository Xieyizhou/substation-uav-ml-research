"""Explicit isolated renderer A/B replay; no production installation or training."""
import argparse
import asyncio
import fcntl
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image,ImageChops
from scripts.vision.run_dual_box_diagnosis import ROOT,OUT as DUAL,SOURCE,read,verify,frozen,file_sha256,dual_sensor,validate_dual,parse_visible
from scripts.vision.verify_full_box_runtime_trace import guard,OUT as PRIOR
from scripts.vision.instance_visibility_diagnosis import ET,raw_box
from scripts.vision.run_visibility_cleanup_validation import check_mask
from src.vision.canonical.collect import pose_record
from src.vision.canonical.plan import pose_close,rotate
from src.vision.canonical.gates import validate_point
from src.sensors.gazebo_visual_transport import message_timestamp
import scripts.vision.run_visibility_cleanup_validation as replay

OUT=ROOT/'data/research/ml_training_recovery_v1/renderer-near-clip-test-v1'
VARIANT=None


def freeze():
    guard();verify(read(OUT/'control-build.json'));verify(read(PRIOR/'completion.json'))
    path=OUT/'protocol.json'
    if path.exists():verify(read(path));return read(path)
    lib=OUT/'build-clipped/lib/libgz-rendering8-ogre2.8.2.3.dylib'
    target=OUT/'variants/clipped';target.mkdir(exist_ok=False)
    shutil.copy2(lib,target/lib.name)
    for name in ('libgz-rendering8-ogre2.dylib','libgz-rendering8-ogre2.8.dylib'):(target/name).symlink_to(lib.name)
    paths=[OUT/'control-build.json',PRIOR/'completion.json',DUAL/'protocol.json',SOURCE/'protocol.json',Path(__file__),
           ROOT/'tools/gz_clip_diagnostic_server.cc',OUT/'diagnostic-server',ROOT/'tools/renderer_clip/test_clip.cc',OUT/'test-clip',
           OUT/'configure-clipped.log',OUT/'build-clipped.log',OUT/'build-clipped/CMakeCache.txt',
           OUT/'patched-source/ogre2/src/Ogre2BoundingBoxCamera.cc',OUT/'build/lib/libgz-rendering8.8.2.3.dylib',
           OUT/'build-clipped/lib/libgz-rendering8.8.2.3.dylib']
    paths += list((ROOT/'tools/renderer_clip').glob('*.hh'))+list((OUT/'patched-source/ogre2/src').glob('Diagnostic*.hh'))
    paths += [OUT/'variants'/v/lib.name for v in ('control','clipped')]
    for variant in ('control','clipped'):
        dest=OUT/'runs'/variant;dest.mkdir(parents=True)
        shutil.copy2(DUAL/'gz_visibility_capture_cleanup_fixed',dest/'gz_visibility_capture_cleanup_fixed')
        paths.append(dest/'gz_visibility_capture_cleanup_fixed')
    p=frozen(path,dict(status='isolated_renderer_ab_frozen',frames=read(DUAL/'protocol.json')['frames'],
        policy='Same upstream commit, compiler, Release flags and dependencies; separate build directories. Only MeshMinimalBox triangle homogeneous clipping and outside/empty predicate differ. No cmake install; plugin path environment is scoped to child process. Existing full_2d semantics of visible-component eligibility retained.',
        max_attempts_per_variant=3,training_ready=False,training_started=False,
        inputs={str(p):file_sha256(p) for p in paths}))
    for variant in ('control','clipped'):
        frozen(OUT/'runs'/variant/'protocol.json',dict(status='isolated_renderer_variant',variant=variant,frames=p['frames'],
            training_ready=False,training_started=False,inputs={str(path):file_sha256(path)}))
    return p


def box_map(message,mapping):
    boxes={}
    for item in message.get('annotatedBox',[]):
        label=str(int(item['label']));box=raw_box(item)
        if label not in mapping or label in boxes or not np.isfinite(box).all():raise ValueError('Unknown/duplicate/invalid box')
        if not(0<=box[0]<box[2]<=1920 and 0<=box[1]<box[3]<=1080):raise ValueError('Invalid positive-area box')
        boxes[label]=box
    return boxes


def analyze(folder,frame,fence):
    selected=None
    for start in range(1,11):
        signatures=[tuple(file_sha256(folder/f'frame-{n}-{k}.bin') for k in ('rgb','mask')) for n in range(start,start+3)]
        if len(set(signatures))==1:selected=list(range(start,start+3));break
    if selected is None:raise ValueError('No three stable RGB/mask frames')
    loaded=[line.split('DIAGNOSTIC_LOADED_LIBRARY ',1)[1].strip() for line in (folder/'simulator.log').read_text().splitlines() if 'DIAGNOSTIC_LOADED_LIBRARY ' in line]
    engines=[Path(p) for p in loaded if 'libgz-rendering8-ogre2' in p]
    expected=OUT/'variants'/VARIANT/'libgz-rendering8-ogre2.8.2.3.dylib'
    if len(engines)!=1 or engines[0].resolve()!=expected.resolve() or file_sha256(engines[0])!=file_sha256(expected):raise ValueError('Wrong isolated plugin loaded')
    original=Image.open(frame['source_image']).convert('RGB')
    mapping=frame['instance_mapping'];old=box_map(read(frame['source_receipt'])['raw_truth'],mapping)
    config=read(Path(frame['source_plan']).parent/'obstacles.json')
    camera=ET.parse(frame['source_world']).find(".//model[@name='canonical_camera']/link[@name='research_camera_link']")
    offset=list(map(float,camera.findtext('pose').split()))[:3]
    rows=[]
    for n in selected:
        prefix=folder/f'frame-{n}'
        metadata={k:read(Path(str(prefix)+'-'+k+'.json')) for k in ('rgb','depth','mask','pose','boxes','visible-boxes')}
        times=[message_timestamp(m) for m in metadata.values()]
        skew=max(abs(t-times[0]) for t in times)*1000
        if min(times)<=fence or skew>33.334001:raise ValueError('Stale or unsynchronized sensor data')
        pose=pose_record(metadata['pose'])
        if not pose or not pose_close(pose,frame['actual_pose']):raise ValueError('Pose mismatch')
        validate_point(pose['position'],config,role='actual carrier')
        validate_point((np.array(pose['position'])+rotate(pose['orientation'],offset)).tolist(),config,role='actual optical center')
        for k,w,h,fmt,size in [('rgb',1920,1080,'RGB_INT8',3),('mask',1920,1080,'RGB_INT8',3),('depth',640,360,'R_FLOAT32',4)]:
            m=metadata[k]
            if (m['width'],m['height'],m.get('pixelFormatType'))!=(w,h,fmt) or Path(str(prefix)+'-'+k+'.bin').stat().st_size!=w*h*size:raise ValueError('Invalid sensor format/payload')
        rgb=np.frombuffer(Path(str(prefix)+'-rgb.bin').read_bytes(),dtype='u1').reshape(1080,1920,3)
        mask=np.frombuffer(Path(str(prefix)+'-mask.bin').read_bytes(),dtype='u1').reshape(1080,1920,3)
        check_mask(metadata['mask'],mask,mapping)
        image=Image.fromarray(rgb);image.save(Path(str(prefix)+'-rgb.png'));ImageChops.difference(original,image).save(Path(str(prefix)+'-difference.png'))
        if image.tobytes()!=original.tobytes():raise ValueError('Replay RGB differs from original')
        boxes=box_map(metadata['boxes'],mapping);visible=parse_visible(metadata['visible-boxes'],mapping)
        if not set(old)<=set(boxes):raise ValueError('Original instance lost')
        delta={k:max(abs(a-b) for a,b in zip(old[k],boxes[k])) for k in old}
        if VARIANT=='control' and (set(boxes)!=set(old) or max(delta.values())>1):raise ValueError('Rebuilt control differs from historical boxes')
        ys,xs=np.where(mask[:,:,2]==128)
        extent=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None
        if extent!=[59,1052,306,1080] or len(xs)!=3534 or visible['128']['half_open']!=extent:raise ValueError('Instance evidence changed')
        rows.append(dict(capture_index=n,skew_ms=skew,actual_pose=pose,rgb_exact=True,full_boxes=boxes,
            original_box_deltas=delta,new_labels=sorted(set(boxes)-set(old)),visible_128=visible['128'],mask_128_bbox=extent,mask_128_pixels=len(xs)))
    if any(row['full_boxes']!=rows[0]['full_boxes'] for row in rows):raise ValueError('Unstable full boxes')
    return dict(status='isolated_replay_verified',variant=VARIANT,selected_capture_indices=selected,records=rows,
        loaded_libraries=loaded,isolated_plugin_sha256=file_sha256(expected),training_ready=False,training_started=False,
        reason='Pixel/pose/sync checks passed. Changed full box extents are diagnostic outputs only, never replacement training labels.')


class IsolatedAsync:
    def __getattr__(self,name):return getattr(asyncio,name)
    async def create_subprocess_exec(self,*args,**kwargs):
        if args[:4]!=('gz','sim','-s','-r'):raise ValueError('Unexpected launch')
        kwargs['env']=dict(kwargs['env'],GZ_RENDERING_PLUGIN_PATH=str(OUT/'variants'/VARIANT),
            GZ_RENDERING_RESOURCE_PATH='/opt/homebrew/opt/gz-rendering8/share/gz/gz-rendering8')
        return await asyncio.create_subprocess_exec(str(OUT/'diagnostic-server'),args[4],**kwargs)


async def run(variant):
    global VARIANT
    protocol=freeze();VARIANT=variant;dest=OUT/'runs'/variant
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=(replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio)
        replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio=dest,dual_sensor,validate_dual,analyze,IsolatedAsync()
        paths=[dest/'protocol.json'];last=None
        try:
            for number in range(1,4):
                rp=dest/'replay/T027'/f'attempt-{number:02}'/'receipt.json'
                if rp.exists():last=read(rp);verify(last)
                elif rp.parent.exists():continue
                else:last=await replay.attempt(protocol['frames'][0],number)
                paths.append(rp)
                if last['status']!='technical_failure':break
        finally:(replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio)=old
        guard()
        p=dest/'completion.json'
        if p.exists():verify(read(p));return
        frozen(p,dict(status=last['status'] if last else 'incomplete_attempts',training_ready=False,training_started=False,
            inputs={str(p):file_sha256(p) for p in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replay',choices=['control','clipped']);args=ap.parse_args()
    if args.replay:asyncio.run(run(args.replay))
    else:freeze();print('ISOLATED_BUILD_FROZEN_NO_REPLAY_NO_TRAINING')
