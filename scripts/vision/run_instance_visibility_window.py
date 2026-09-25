"""Bounded real renderer replay, strictly separate from historical collection."""
import asyncio
import json
import os
import time
import uuid
import shutil
import numpy as np
from PIL import Image, ImageChops
from scripts.vision.instance_visibility_diagnosis import (
    OUT, ROOT, PILOT, prepare, read, save, file_sha256, verify_tree, ET,
    add_sensor, validate_world, command, stop_group, raw_box, Path,
)
from src.vision.canonical.collect import pose_record
from src.vision.canonical.plan import pose_close, rotate
from src.vision.canonical.gates import validate_point, validate_preflight
from src.sensors.gazebo_visual_transport import message_timestamp

def check_mask(metadata, pixels, mapping, segmentation_type='panoptic'):
    if segmentation_type!='panoptic':raise ValueError('Category mask is not panoptic')
    if metadata.get('pixelFormatType',metadata.get('pixel_format_type')) not in ('RGB_INT8',3):raise ValueError('Unexpected mask format')
    if pixels.shape!=(1080,1920,3):raise ValueError('Unexpected mask dimensions')
    labels=set(map(int,np.unique(pixels[:,:,2])))
    # SDF unlabelled visuals are label zero; sky may use 255.
    unknown=labels-{0,255}-{int(k) for k in mapping}
    if unknown:raise ValueError('Unresolved runtime labels: '+str(sorted(unknown)))
    if not labels-{0,255}:raise ValueError('No mapped labels: cannot certify mask encoding')
    foreground=~np.isin(pixels[:,:,2],[0,255])
    if np.any(foreground & np.all(pixels[:,:,:2]==0,axis=2)):
        raise ValueError('Missing panoptic instance count on labelled pixels')
    return sorted(labels)

def certify(rgb_exact, pose_valid, stable, skew_ms, box_delta, mask_valid):
    return bool(rgb_exact and pose_valid and stable>=3 and skew_ms<=33.334+1e-6 and box_delta<=1 and mask_valid)

def analyze_selected(folder, frame, fence):
    records=[];original=Image.open(frame['source_image']).convert('RGB');mapping=frame['instance_mapping']
    config=read(Path(frame['source_plan']).parent/'obstacles.json')
    camera=ET.parse(frame['source_world']).find(".//model[@name='canonical_camera']/link[@name='research_camera_link']")
    offset=list(map(float,camera.findtext('pose').split()))[:3]
    for i in range(1,4):
        prefix=folder/f'frame-{i}'
        metadata={k:read(str(prefix)+f'-{k}.json') for k in ('rgb','depth','mask','pose','boxes')}
        stamps=[message_timestamp(m) for m in metadata.values()]
        if min(stamps)<=fence:raise ValueError('Stale pre-move frame')
        skew=max(abs(t-stamps[0])*1000 for t in stamps)
        if skew>33.334+1e-6:raise ValueError('Unsynchronized frame')
        pose=pose_record(metadata['pose'])
        if not pose or not pose_close(pose,frame['actual_pose']):raise ValueError('Camera outside pose tolerance')
        optical=np.array(pose['position'])+rotate(pose['orientation'],offset)
        validate_point(pose['position'],config,role='replay carrier');validate_point(optical.tolist(),config,role='replay optical center')
        arrays={}
        for k,w,h,dtype,ch,fmt in [('rgb',1920,1080,'u1',3,'RGB_INT8'),('mask',1920,1080,'u1',3,'RGB_INT8'),('depth',640,360,'<f4',1,'R_FLOAT32')]:
            m=metadata[k]
            if (m['width'],m['height'])!=(w,h) or m.get('pixelFormatType')!=fmt:raise ValueError('Unexpected '+k+' sensor metadata')
            raw=Path(str(prefix)+f'-{k}.bin').read_bytes()
            if len(raw)!=w*h*ch*np.dtype(dtype).itemsize:raise ValueError('Payload length mismatch')
            arrays[k]=np.frombuffer(raw,dtype=dtype).reshape((h,w,ch) if ch>1 else (h,w))
        labels=check_mask(metadata['mask'],arrays['mask'],mapping)
        image=Image.fromarray(arrays['rgb']);image.save(str(prefix)+'-rgb.png')
        diff=ImageChops.difference(original,image);diff.save(str(prefix)+'-difference.png')
        exact=original.tobytes()==image.tobytes()
        orig_boxes=read(frame['source_receipt'])['raw_truth']['annotatedBox'];new_boxes=metadata['boxes'].get('annotatedBox',[])
        deltas=[]
        for b in orig_boxes:
            same=[n for n in new_boxes if n.get('label',0)==b.get('label',0)]
            if len(same)!=1:raise ValueError('Missing/duplicate replay box instance')
            deltas.append(max(abs(a-c) for a,c in zip(raw_box(b),raw_box(same[0]))))
        if len(orig_boxes)!=len(new_boxes):raise ValueError('Replay box membership changed')
        targets=[]
        for e in frame['events']:
            mask=arrays['mask'][:,:,2]==e['runtime_label'];ys,xs=np.where(mask)
            target=dict(review_id=e['review_id'],runtime_label=e['runtime_label'],visible_pixel_count=int(mask.sum()),
                visible_bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if xs.size else None,
                component_evidence='unknown',instance_channel_pairs=np.unique(arrays['mask'][mask,:2],axis=0).tolist())
            overlay=arrays['rgb'].copy();overlay[mask]=(overlay[mask].astype(float)*.5+np.array([255,0,255])*.5).astype('u1')
            im=Image.fromarray(overlay);im.save(folder/f'{i}-{e["review_id"]}-overlay.png')
            x1,y1,x2,y2=e['bbox_xyxy'];im.crop((max(0,int(x1)-30),max(0,int(y1)-30),min(1920,int(x2)+31),min(1080,int(y2)+31))).save(folder/f'{i}-{e["review_id"]}-crop.png')
            targets.append(target)
        records.append(dict(frame_index=i,actual_pose=pose,skew_ms=skew,rgb_exact=exact,
            differing_rgb_pixels=int(np.any(arrays['rgb']!=np.asarray(original),axis=2).sum()),
            maximum_box_delta_px=max(deltas,default=0),mask_labels=labels,targets=targets))
    stable=all(r['rgb_exact']==records[0]['rgb_exact'] for r in records) and all(r['targets']==records[0]['targets'] for r in records)
    certified=stable and all(certify(r['rgb_exact'],True,3,r['skew_ms'],r['maximum_box_delta_px'],True) for r in records)
    return dict(status='original_pixel_evidence_certified' if certified else 'alignment_held',records=records,
                stable_frames=3 if stable else 0,mask_encoding='panoptic RGB: first two channels instance count, last runtime label; aggregate by unique equipment runtime label',
                reason='' if certified else 'RGB not exact, box delta >1 pixel or unstable output; original-frame visibility remains unknown')

def analyze(folder, frame, fence):
    # Select by internal stability alone, never by agreement with the original.
    selected = None
    for start in range(1, 11):
        signatures = [tuple(file_sha256(folder/f'frame-{i}-{k}.bin') for k in ('rgb','mask')) for i in range(start,start+3)]
        if len(set(signatures)) == 1:
            selected = list(range(start,start+3))
            break
    if selected is None:
        raise ValueError('No three internally stable RGB/mask frames in twelve-frame window')
    stable=folder/'first-stable-window';stable.mkdir()
    for i,source in enumerate(selected,1):
        for path in folder.glob(f'frame-{source}-*'):
            shutil.copyfile(path,stable/path.name.replace(f'frame-{source}-',f'frame-{i}-',1))
    result=analyze_selected(stable,frame,fence)
    result['selected_capture_indices']=selected
    result['selection_rule']='First three consecutive internally byte-identical RGB and mask; independent of original RGB'
    return result

async def attempt(frame, number):
    folder=OUT/'replay'/frame['review_ids'][0]/f'attempt-{number:02}'
    folder.mkdir(parents=True,exist_ok=False)
    result=dict(status='technical_failure');server=None;cancelled=False;env=dict(os.environ,GZ_PARTITION='visibility-'+uuid.uuid4().hex,GZ_IP='127.0.0.1')
    try:
        plan=read(frame['source_plan']);receipt=read(frame['source_receipt'])
        view=next(v for k in ('calibration_views','pilot_views') for v in plan.get(k,[]) if v['view_id']==receipt['view_id'])
        _,config,_=validate_preflight(plan,Path(frame['source_plan']).parent,[view])
        camera=ET.parse(frame['source_world']).find(".//model[@name='canonical_camera']/link[@name='research_camera_link']")
        offset=list(map(float,camera.findtext('pose').split()))[:3]
        validate_point(frame['actual_pose']['position'],config,role='requested actual carrier')
        validate_point((np.array(frame['actual_pose']['position'])+rotate(frame['actual_pose']['orientation'],offset)).tolist(),config,role='requested actual optical')
        tree=add_sensor(ET.parse(frame['source_world']));wp=folder/'world.sdf';tree.write(wp,encoding='utf-8',xml_declaration=True)
        validate_world(ET.parse(frame['source_world']),ET.parse(wp))
        with (folder/'simulator.log').open('x') as log:
            server=await asyncio.create_subprocess_exec('gz','sim','-s','-r',str(wp),env=env,stdout=log,stderr=asyncio.subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+45;topics=[]
            required={'/research_camera/image','/research_camera/depth','/research_camera/boxes','/diagnostic/instances/labels_map',f'/world/{frame["world_name"]}/pose/info'}
            while time.monotonic()<deadline:
                if server.returncode is not None:raise RuntimeError('Simulator exited during startup')
                topics=(await command('gz','topic','-l',env=env)).splitlines()
                if required<=set(topics):break
                await asyncio.sleep(1)
            else:raise TimeoutError('Required topics not found')
            x,y,z=frame['actual_pose']['position'];qx,qy,qz,qw=frame['actual_pose']['orientation']
            req=f'name:"canonical_camera" position{{x:{x} y:{y} z:{z}}} orientation{{x:{qx} y:{qy} z:{qz} w:{qw}}}'
            response=await command('gz','service','-s',f'/world/{frame["world_name"]}/set_pose','--reqtype','gz.msgs.Pose','--reptype','gz.msgs.Boolean','--timeout','3000','--req',req,env=env)
            if 'true' not in response:raise ValueError('Pose request rejected')
            initial=json.loads(await command('gz','topic','-e','--json-output','-n','1','-t',f'/world/{frame["world_name"]}/pose/info',env=env,timeout=10))
            fence=message_timestamp(initial)
            await command(str(OUT/'gz_visibility_capture_window'),str(folder),frame['world_name'],env=env,timeout=70)
            result=analyze(folder,frame,fence)
    except ValueError as e:result=dict(status='semantic_blocked',reason=str(e))
    except asyncio.CancelledError:
        cancelled=True;result=dict(status='technical_failure',reason='CancelledError: execution cancelled')
    except Exception as e:result=dict(status='technical_failure',reason=f'{type(e).__name__}: {e}')
    finally:await stop_group(server)
    inputs={str(p):file_sha256(p) for p in folder.rglob('*') if p.is_file()}
    for p in (OUT/'protocol.json',Path(__file__),ROOT/'tools/gz_visibility_capture_window.cc',OUT/'gz_visibility_capture_window'):
        inputs[str(p)]=file_sha256(p)
    result.update(inputs=inputs,process_cleanup_complete=server is None or server.returncode is not None,partition=env['GZ_PARTITION'],member_id=frame['member_id'],review_ids=frame['review_ids'])
    save(folder/'receipt.json',result);print(frame['review_ids'],number,result['status'],result.get('reason',''),flush=True)
    if cancelled:raise asyncio.CancelledError
    return result

async def run():
    protocol=prepare();frames=protocol['frames'];pilot=[]
    async def unit(frame):
        root=OUT/'replay'/frame['review_ids'][0]
        for n in range(1,4):
            rp=root/f'attempt-{n:02}'/'receipt.json'
            if rp.exists():
                verify_tree(rp);r=read(rp)
                if r['status']=='alignment_held' and r.get('records') and len(r['records'])==3 and not r['records'][0]['rgb_exact'] and all(x['rgb_exact'] for x in r['records'][1:]):
                    # Explicit startup-window defect recovery; no semantic relabeling.
                    print('RECOVER_STARTUP_WINDOW',frame['review_ids'],str(rp),flush=True)
                elif r['status'] not in ('technical_failure','topics_available'):return r
                continue
            if rp.parent.exists():continue # Incomplete attempt remains forensic evidence.
            r=await attempt(frame,n)
            if r['status']!='technical_failure':return r
        return dict(status='technical_attempts_exhausted',reason='Three attempt directories consumed')
    for rid in PILOT:
        f=next(f for f in frames if rid in f['review_ids']);pilot.append(await unit(f))
    if all(r['status']=='original_pixel_evidence_certified' for r in pilot):
        for f in frames:
            if not set(f['review_ids'])&set(PILOT):await unit(f)
    else:print('EXPANSION_HELD: pilot not fully certified',flush=True)

if __name__=='__main__':asyncio.run(run())
