"""Explicit bounded five-pair capture. Default freezes only; never trains."""
import argparse,asyncio,copy,fcntl,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageChops
from scripts.vision.prepare_physical_lighting_control import OUT as PREP,prior,validate_change
from scripts.vision.run_depth_clip_test import OUT as DEPTH,BUILD,base,PrefixAsync
from scripts.vision import run_depth_clip_clock_fence as clock

OUT=PREP/'capture-v2-sync-reference'

def freeze():
    pp=PREP/'protocol.json';p=prior.read(pp);prior.verify(p);prior.verify(prior.read(PREP/'completion.json'))
    dest=OUT/'protocol.json'
    if dest.exists():prior.verify(prior.read(dest));return prior.read(dest)
    OUT.mkdir(exist_ok=True);paths=[pp,PREP/'completion.json',Path(__file__).resolve(),Path(base.replay.__file__),Path(clock.__file__),DEPTH/'protocol.json'];frames=[]
    for variant in ('original','physical-lighting'):
        for selected in p['selected']:
            source=selected['source'];folder=OUT/'plans'/f'{selected["id"]}-{variant}';folder.mkdir(parents=True)
            original=base.ET.parse(selected['original_world']).getroot();changed=base.ET.parse(selected['lighting_world']).getroot();validate_change(original,changed)
            wp=folder/'world.sdf';base.ET.ElementTree(original if variant=='original' else changed).write(wp,encoding='utf-8',xml_declaration=True)
            obs=folder/'obstacles.json';shutil.copy2(Path(source['source_plan']).parent/'obstacles.json',obs)
            plan=copy.deepcopy(prior.read(source['source_plan']));parent=plan.pop('identity',None);plan.pop('inputs',None)
            plan['files']={**plan['files'],'world.sdf':prior.file_sha256(wp),'obstacles.json':prior.file_sha256(obs)}
            plan.update(parent_source_identity=parent,inputs={str(wp):prior.file_sha256(wp),str(obs):prior.file_sha256(obs),source['source_plan']:prior.file_sha256(source['source_plan'])})
            np=folder/'plan.json';prior.frozen(np,plan)
            frames.append(dict(pair_id=selected['id'],variant=variant,member_id=selected['member_id'],review_ids=[f'{selected["id"]}-{variant}'],
                source_receipt=source['source_receipt'],source_image=source['source_image'],source_world=str(wp),source_plan=str(np),world_name=plan['world_name'],
                actual_pose=selected['actual_pose'],instance_mapping=selected['instance_mapping']))
            paths += [wp,obs,np,Path(source['source_receipt']),Path(source['source_image'])]
    helper=OUT/'gz_visibility_capture_cleanup_fixed';shutil.copy2(DEPTH/helper.name,helper);paths.append(helper)
    return prior.frozen(dest,dict(status='ten_units_frozen_not_captured',frames=frames,max_attempts=3,
        authorization='Five original replays and five physical-lighting captures, explicit full-frame review, then loader-only preflight; no training.',
        inputs={str(x):prior.file_sha256(x) for x in paths}))

def selected_indices(folder):
    for start in range(1,11):
        signatures=[tuple(prior.file_sha256(folder/f'frame-{i}-{k}.bin') for k in ('rgb','mask')) for i in range(start,start+3)]
        if len(set(signatures))==1:return list(range(start,start+3))
    raise ValueError('No three stable consecutive RGB/mask frames')

def certify_pair(variant,rgb_exact,added,lost,delta,unboxed,mask_equal):
    if added or lost or delta>1 or unboxed:raise ValueError('Full-label membership/geometry/coverage conflict')
    if variant=='original' and not rgb_exact:raise ValueError('Original RGB alignment failed')
    if variant=='physical-lighting' and not mask_equal:raise ValueError('Lighting changed visible instance pixel membership')

def analyze(folder,frame,fence):
    indices=selected_indices(folder);mapping=frame['instance_mapping'];original=Image.open(frame['source_image']).convert('RGB')
    old=base.box_map(prior.read(frame['source_receipt'])['raw_truth'],mapping);config=prior.read(Path(frame['source_plan']).parent/'obstacles.json')
    camera=base.ET.parse(frame['source_world']).find(".//model[@name='canonical_camera']/link[@name='research_camera_link']")
    offset=list(map(float,camera.findtext('pose').split()))[:3]
    libs=[s.split('DIAGNOSTIC_LOADED_LIBRARY ',1)[1].strip() for s in (folder/'simulator.log').read_text().splitlines() if 'DIAGNOSTIC_LOADED_LIBRARY ' in s]
    engine=[Path(x) for x in libs if 'libgz-rendering8-ogre2' in x]
    if len(engine)!=1 or engine[0].resolve()!=(DEPTH/'variants/depth/libgz-rendering8-ogre2.8.2.3.dylib').resolve():raise ValueError('Wrong renderer loaded')
    normal_mask=None;normal_record=None
    if frame['variant']=='physical-lighting':
        receipts=sorted((OUT/'replay'/f'{frame["pair_id"]}-original').glob('attempt-*/receipt.json'))
        valid=[rp for rp in receipts if prior.read(rp)['status']=='capture_technical_checks_passed']
        if len(valid)!=1:raise ValueError('Unique aligned original missing')
        nr=prior.read(valid[0]);prior.verify(nr);normal_record=nr['records'][0]
        normal_mask=np.frombuffer((valid[0].parent/f'frame-{normal_record["capture_index"]}-mask.bin').read_bytes(),dtype='u1').reshape(1080,1920,3)[:,:,2]
    records=[]
    for n in indices:
        meta={k:prior.read(folder/f'frame-{n}-{k}.json') for k in ('rgb','depth','mask','pose','boxes','visible-boxes')}
        stamps=[base.message_timestamp(v) for v in meta.values()];skew=max(abs(t-stamps[0]) for t in stamps)*1000
        if min(stamps)<=fence or skew>33.334001:raise ValueError('Clock/synchronization gate failed')
        pose=base.pose_record(meta['pose'])
        if not pose or not base.pose_close(pose,frame['actual_pose']):raise ValueError('Actual pose mismatch')
        base.validate_point(pose['position'],config,role='actual carrier');base.validate_point((np.array(pose['position'])+base.rotate(pose['orientation'],offset)).tolist(),config,role='actual optical')
        for k,w,h,fmt,size in [('rgb',1920,1080,'RGB_INT8',3),('mask',1920,1080,'RGB_INT8',3),('depth',640,360,'R_FLOAT32',4)]:
            m=meta[k]
            if (m['width'],m['height'],m.get('pixelFormatType'))!=(w,h,fmt) or (folder/f'frame-{n}-{k}.bin').stat().st_size!=w*h*size:raise ValueError('Sensor payload/configuration mismatch')
        rgb=np.frombuffer((folder/f'frame-{n}-rgb.bin').read_bytes(),dtype='u1').reshape(1080,1920,3);mask=np.frombuffer((folder/f'frame-{n}-mask.bin').read_bytes(),dtype='u1').reshape(1080,1920,3)
        labels=base.check_mask(meta['mask'],mask,mapping);boxes=base.box_map(meta['boxes'],mapping);visible=base.parse_visible(meta['visible-boxes'],mapping)
        image=Image.fromarray(rgb);image.save(folder/f'frame-{n}-rgb.png');ImageChops.difference(original,image).save(folder/f'frame-{n}-difference.png')
        added=sorted(set(boxes)-set(old));lost=sorted(set(old)-set(boxes));deltas={k:max(abs(a-b) for a,b in zip(old[k],boxes[k])) for k in set(old)&set(boxes)}
        unboxed=sorted({str(k) for k in labels if k not in (0,255)}-set(boxes));exact=image.tobytes()==original.tobytes()
        mask_equal=True if normal_mask is None else np.array_equal(mask[:,:,2],normal_mask)
        certify_pair(frame['variant'],exact,added,lost,max(deltas.values(),default=0),unboxed,mask_equal)
        if normal_record is not None and visible!=normal_record['visible_boxes']:raise ValueError('Visible box mismatch across light conditions')
        counts={k:int((mask[:,:,2]==int(k)).sum()) for k in boxes}
        if any(v==0 for v in counts.values()):raise ValueError('Box with no visible instance pixels; review required')
        records.append(dict(capture_index=n,rgb_exact_vs_original=exact,skew_ms=skew,all_stream_span_ms=(max(stamps)-min(stamps))*1000,full_boxes=boxes,visible_boxes=visible,
            mask_runtime_pixels=counts,mask_equal_vs_aligned_original=mask_equal,historical_deltas=deltas,actual_pose=pose,
            differing_rgb_pixels=int(np.any(rgb!=np.asarray(original),axis=2).sum())))
    if any(r['full_boxes']!=records[0]['full_boxes'] or r['visible_boxes']!=records[0]['visible_boxes'] for r in records):raise ValueError('Boxes not stable')
    return dict(status='capture_technical_checks_passed',variant=frame['variant'],records=records,selected_capture_indices=indices,loaded_libraries=libs,
        original_pixel_visibility_certified=frame['variant']=='original',review_status='explicit_full_frame_review_pending',training_ready=False)

async def run():
    p=freeze();base.guard();replay=base.replay;results=[];paths=[OUT/'protocol.json']
    with (BUILD/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=(base.OUT,base.VARIANT,replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command)
        base.OUT,base.VARIANT=DEPTH,'depth'
        replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command=OUT,base.dual_sensor,base.validate_dual,analyze,PrefixAsync(),clock.command
        try:
            for frame in p['frames']:
                last=None
                for n in range(1,4):
                    clock.CURRENT=OUT/'replay'/frame['review_ids'][0]/f'attempt-{n:02}';rp=clock.CURRENT/'receipt.json'
                    if rp.exists():last=prior.read(rp);prior.verify(last)
                    elif clock.CURRENT.exists():continue
                    else:last=await replay.attempt(frame,n)
                    if not last['process_cleanup_complete']:raise ValueError('Process cleanup incomplete')
                    if last['status']!='technical_failure':break
                if last is None:raise ValueError('No complete attempt within cap')
                paths.append(rp);results.append(dict(pair_id=frame['pair_id'],variant=frame['variant'],receipt=str(rp),status=last['status'],reason=last.get('reason')))
                if last['status']!='capture_technical_checks_passed':break
        finally:(base.OUT,base.VARIANT,replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command)=old
    base.guard()
    prior.frozen(OUT/'capture-receipt.json',dict(status='all_ten_captured_review_pending' if len(results)==10 and all(r['status']=='capture_technical_checks_passed' for r in results) else 'capture_blocked',
        results=results,training_ready=False,training_started=False,inputs={str(x):prior.file_sha256(x) for x in paths}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--capture',action='store_true');a=ap.parse_args()
    if a.capture:asyncio.run(run())
    else:freeze();print('FROZEN_NO_RENDER')
