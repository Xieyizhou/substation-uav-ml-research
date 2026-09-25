"""Explicit diagnostic body-visual ablation after an exact saved-frame control."""
import argparse
import asyncio
import copy
import fcntl
import os
from pathlib import Path
import shutil
import numpy as np
from PIL import Image,ImageChops
from scripts.vision.run_body_visibility_control import OUT as CONTROL,ROOT,read,verify,frozen,file_sha256,replay
from scripts.vision.instance_visibility_diagnosis import add_sensor,validate_world,ET
from src.vision.canonical.collect import pose_record
from src.vision.canonical.plan import pose_close,rotate
from src.vision.canonical.gates import validate_point
from src.sensors.gazebo_visual_transport import message_timestamp

OUT=CONTROL/'body-visual-removal-v1'


def without_body(tree):
    result=copy.deepcopy(tree)
    matches=result.findall(".//world/model[@name='capacitor_east']/link/visual[@name='body']")
    if len(matches)!=1:raise ValueError('Ambiguous selected body visual')
    body=matches[0]
    link=next(x for x in result.findall('.//link') if body in list(x))
    if len([v for v in link.findall('visual') if v.get('name','').startswith('capacitor_')])!=6:
        raise ValueError('Six capacitor visual components required')
    link.remove(body)
    return result


def modified_sensor(tree):return add_sensor(without_body(tree))


def validate_modified(original,derived):validate_world(without_body(original),derived)


def analyze(folder,frame,fence):
    selected=None
    for start in range(1,11):
        hashes=[tuple(file_sha256(folder/f'frame-{i}-{k}.bin') for k in ('rgb','mask')) for i in range(start,start+3)]
        if len(set(hashes))==1:selected=list(range(start,start+3));break
    if selected is None:raise ValueError('No three consecutive stable diagnostic frames')
    config=read(Path(frame['source_plan']).parent/'obstacles.json')
    camera=ET.parse(frame['source_world']).find(".//model[@name='canonical_camera']/link[@name='research_camera_link']")
    offset=list(map(float,camera.findtext('pose').split()))[:3]
    original=Image.open(frame['source_image']).convert('RGB');records=[]
    for ordinal,i in enumerate(selected,1):
        prefix=folder/f'frame-{i}';meta={k:read(str(prefix)+f'-{k}.json') for k in ('rgb','mask','depth','pose','boxes')}
        stamps=[message_timestamp(m) for m in meta.values()];skew=max(abs(t-stamps[0])*1000 for t in stamps)
        if min(stamps)<=fence or skew>33.334+1e-6:raise ValueError('Stale or unsynchronized diagnostic')
        pose=pose_record(meta['pose'])
        if not pose or not pose_close(pose,frame['actual_pose']):raise ValueError('Invalid diagnostic pose')
        validate_point(pose['position'],config,role='diagnostic carrier')
        optical=(np.array(pose['position'])+rotate(pose['orientation'],offset)).tolist()
        validate_point(optical,config,role='diagnostic optical')
        arrays={}
        for k,w,h,dtype,ch,fmt in [('rgb',1920,1080,'u1',3,'RGB_INT8'),('mask',1920,1080,'u1',3,'RGB_INT8'),('depth',640,360,'<f4',1,'R_FLOAT32')]:
            m=meta[k];raw=Path(str(prefix)+f'-{k}.bin').read_bytes()
            if (m['width'],m['height'],m.get('pixelFormatType'))!=(w,h,fmt) or len(raw)!=w*h*ch*np.dtype(dtype).itemsize:
                raise ValueError('Invalid sensor metadata or payload')
            arrays[k]=np.frombuffer(raw,dtype=dtype).reshape((h,w,ch) if ch>1 else (h,w))
        labels=replay.check_mask(meta['mask'],arrays['mask'],frame['instance_mapping'])
        im=Image.fromarray(arrays['rgb']);im.save(folder/f'stable-{ordinal}-rgb.png')
        ImageChops.difference(original,im).save(folder/f'stable-{ordinal}-difference.png')
        e=frame['events'][0];mask=arrays['mask'][:,:,2]==e['runtime_label'];ys,xs=np.where(mask)
        x1,y1,x2,y2=e['bbox_xyxy'];crop=(max(0,int(x1)-30),max(0,int(y1)-30),min(1920,int(x2)+31),min(1080,int(y2)+31))
        im.crop(crop).save(folder/f'stable-{ordinal}-crop.png')
        original.crop(crop).save(folder/f'stable-{ordinal}-original-crop.png')
        records.append(dict(capture_index=i,actual_pose=pose,skew_ms=skew,mask_labels=labels,
            aggregate_equipment_mask_pixels=int(mask.sum()),equipment_bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if xs.size else None,
            differing_rgb_pixels=int(np.any(arrays['rgb']!=np.asarray(original),axis=2).sum()),
            boxes=meta['boxes'],original_pixel_visibility_certified=False,component_mask_identity='unknown_equipment_label_only'))
    return dict(status='diagnostic_render_stable_pending_visual_review',records=records,stable_frames=3,
        selected_capture_indices=selected,selection_rule='First three consecutive byte-identical RGB and mask; no selection against original image or outcome',
        original_pixel_visibility_certified=False,reason='Body visual removed; original labels/boxes are not ground truth for this diagnostic image.')


def freeze():
    p=read(CONTROL/'protocol.json');verify(p)
    completed=read(CONTROL/'control-completion.json');verify(completed)
    if not completed['control_passed']:raise ValueError('Exact replay control required before intervention')
    for path in completed['inputs']:
        if Path(path).name=='receipt.json':
            r=read(path);verify(r)
            if r['status']!='original_pixel_evidence_certified' or not r['process_cleanup_complete']:raise ValueError('Invalid control receipt')
    dest=OUT/'protocol.json'
    if dest.exists():verify(read(dest));return read(dest)
    OUT.mkdir(exist_ok=True);binary=CONTROL/'gz_visibility_capture_cleanup_fixed';local=OUT/binary.name;shutil.copy2(binary,local)
    frame=p['frames'][0]
    derived=modified_sensor(ET.parse(frame['source_world']));validate_modified(ET.parse(frame['source_world']),derived)
    paths=[CONTROL/'protocol.json',CONTROL/'control-completion.json',Path(__file__),binary,local]
    return frozen(dest,dict(status='body_visual_removal_frozen_after_exact_control',frames=[frame],max_attempts=3,
        allowed_change='Remove capacitor_east/link/visual[name=body] only; add aligned diagnostic instance sensor. All collision, six cylinder visuals, base, other assets, materials, lighting and camera unchanged.',
        original_labels_applicable=False,training_started=False,training_ready=False,
        inputs={str(x):file_sha256(x) for x in paths}))


async def run():
    p=freeze();dest=OUT/'render-completion.json'
    if dest.exists():verify(read(dest));print('VALID_DIAGNOSTIC_RENDER_REUSED');return
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=(replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze);oldlog=os.environ.get('GZ_LOG_PATH')
        replay.OUT=OUT;replay.add_sensor=modified_sensor;replay.validate_world=validate_modified;replay.analyze=analyze
        logs=OUT/'runtime-logs';logs.mkdir(exist_ok=True);os.environ['GZ_LOG_PATH']=str(logs)
        paths=[OUT/'protocol.json'];last=None
        try:
            for n in range(1,4):
                rp=OUT/'replay/T020'/f'attempt-{n:02}'/'receipt.json'
                if rp.exists():r=read(rp);verify(r)
                elif rp.parent.exists():continue
                else:r=await replay.attempt(p['frames'][0],n)
                paths.append(rp);last=r
                if not r['process_cleanup_complete']:raise ValueError('Incomplete cleanup')
                if r['status']!='technical_failure':break
        finally:
            replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze=old
            if oldlog is None:os.environ.pop('GZ_LOG_PATH',None)
            else:os.environ['GZ_LOG_PATH']=oldlog
        frozen(dest,dict(status=last['status'] if last else 'attempts_exhausted',reason=last.get('reason','') if last else 'Incomplete attempts',
            training_started=False,training_ready=False,original_pixel_visibility_certified=False,
            inputs={str(x):file_sha256(x) for x in paths}))
        print(read(dest)['status'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');a=ap.parse_args()
    if a.replay:asyncio.run(run())
    else:freeze();print('FROZEN_NO_RENDER_NO_TRAINING')
