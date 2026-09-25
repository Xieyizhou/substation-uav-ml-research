"""Standalone Gazebo acquisition. Never imports the PX4 flight runtime."""
import asyncio
from collections import deque
import json
import math
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
import numpy as np
from src.ml.artifacts import file_sha256
from src.sensors.gazebo_rgbd_native import NativeGazeboRgbDepthSource
from src.sensors.gazebo_visual_transport import message_timestamp
from src.vision.collection.gazebo_truth import parse_gazebo_truth_message
from src.vision.training.hard_example_curator import dhash64
from .plan import read_record, write_record, pose_close, CARRIER
from .recovery import acquire_view, resumed_views
from .gates import CHECK_VERSION, target_checks, validate_point, validate_preflight

TARGET_CLASSES = frozenset(("transformer", "switchgear", "capacitor_bank", "reactor"))

def violates_no_target_gate(plan, classes):
    return plan.get('diagnostic_require_no_targets') is True and bool(set(classes) & TARGET_CLASSES)

def valid_annotation_mode(plan):
    """Full-image truth requires explicit per-visual labels on top-level equipment."""
    return (plan.get('annotation_mode') != 'full_2d'
            or (plan.get('label_mode') == 'visual-instance'
                and plan.get('hierarchy_mode') == 'top-level-equipment'))

def allows_diagnostic_absence(plan, mode):
    return (mode == 'calibration' and plan.get('diagnostic_only') is True
            and plan.get('training_admitted') is False
            and plan.get('diagnostic_allow_expected_absence') is True
            and plan.get('diagnostic_require_expected_presence') is not True)

async def command(*args, timeout=5):
    process=await asyncio.create_subprocess_exec(*args,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    try:
        stdout,stderr=await asyncio.wait_for(process.communicate(),timeout)
    except BaseException:
        if process.returncode is None:process.kill()
        await process.communicate();raise
    if process.returncode:raise RuntimeError(stderr.decode(errors='replace')[-1000:])
    return stdout.decode()

async def stop(process):
    if process is None:return
    if process.returncode is None:process.terminate()
    try:await asyncio.wait_for(process.communicate(),3)
    except asyncio.TimeoutError:
        process.kill();await process.communicate()

def pose_record(message):
    for pose in message.get('pose',[]):
        if pose.get('name')==CARRIER:
            p=pose.get('position',{});q=pose.get('orientation',{})
            return {'position':[float(p.get(k,0)) for k in ('x','y','z')],'orientation':[float(q.get(k,0)) for k in ('x','y','z','w')],'timestamp':message_timestamp(message),'raw':message}
    return None

def paired_after_move(rgb_time,depth_time,truth_time,pose_time,fence):
    return min(rgb_time,depth_time,truth_time,pose_time)>fence and max(abs(rgb_time-t) for t in (depth_time,truth_time,pose_time))<=.033334+1e-9

def save_rejected_frame(staging, output, view_id, rgb, depth):
    folder=Path(output)/view_id
    folder.mkdir()
    shutil.copyfile(Path(staging)/rgb.payload_relative_path,folder/'rgb.ppm')
    shutil.copyfile(Path(staging)/depth.payload_relative_path,folder/'depth.bin')
    return {'rgb_path':str(folder/'rgb.ppm'),'depth_path':str(folder/'depth.bin'),
            'image_sha256':file_sha256(folder/'rgb.ppm'),
            'depth_sha256':file_sha256(folder/'depth.bin'),
            'rgb_timestamp':rgb.capture_timestamp,'depth_timestamp':depth.capture_timestamp,
            'rgb_dimensions':[rgb.width,rgb.height],'depth_dimensions':[depth.width,depth.height],
            'training_admitted':False}

async def collect(plan_path, output, mode='calibration', review_path=None, view_id=None, resume_from=None):
    plan_path=Path(plan_path).resolve();plan=read_record(plan_path);base=plan_path.parent
    if not valid_annotation_mode(plan):
        raise ValueError('full_2d collection requires label_mode=visual-instance and hierarchy_mode=top-level-equipment')
    if view_id is not None and (mode!='calibration' or view_id not in {v['view_id'] for v in plan['calibration_views']}):
        raise ValueError('Single-view diagnostics require an existing calibration view')
    for name,identity in plan['files'].items():
        if Path(name).name!=name or file_sha256(base/name)!=identity:raise ValueError('Scene artifact mismatch')
    if mode=='pilot':
        if review_path is None:raise ValueError('Pilot requires reviewed calibration evidence')
        review=read_record(review_path)
        if review.get('plan_identity')!=plan['identity'] or review.get('status')!='accepted':raise ValueError('Calibration review not accepted')
        evidence=read_record(Path(review_path).parent/review['collection_receipt'])
        if evidence['identity']!=review['collection_identity'] or evidence['mode']!='calibration' or evidence['plan_identity']!=plan['identity'] or evidence['status']!='complete_pending_review':raise ValueError('Calibration evidence mismatch')
        required={v['view_id'] for v in plan['calibration_views']}
        accepted={v['view_id'] for v in evidence['views'] if v['status']=='captured'}
        reviewed={v['view_id'] for v in review['views'] if v['decision']=='accepted'}
        if not required<=accepted or not required<=reviewed:raise ValueError('Incomplete calibration review')
    views=plan['calibration_views'] if mode=='calibration' else plan['pilot_views']
    if view_id is not None:views=[v for v in views if v['view_id']==view_id]
    gate_receipt,map_config,instance_map=validate_preflight(plan,base,views)
    recovered,resume_identity=resumed_views(
        resume_from,plan,mode,views,config=map_config,
        check_version=CHECK_VERSION,instance_mapping=instance_map)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    partition='canonical_views_'+plan['identity'][:12]
    old={k:os.environ.get(k) for k in ('GZ_IP','GZ_PARTITION')}
    os.environ.update(GZ_IP='127.0.0.1',GZ_PARTITION=partition)
    server=source=truth_process=pose_process=None;tasks=[];results=list(recovered);fatal=[];snapshot=[]
    completed={row['view_id'] for row in recovered}
    views=[v for v in views if v['view_id'] not in completed]
    truth_history=deque(maxlen=120);poses=deque(maxlen=120);active={};latest_rgb=-1.0
    async def read_truth():
        ordinal=0
        while True:
            line=await truth_process.stdout.readline()
            if not line:raise RuntimeError('Truth stream closed')
            raw=json.loads(line);ordinal+=1
            truth=parse_gazebo_truth_message(raw,topic='/research_camera/boxes',width=1920,height=1080,receive_index=ordinal)
            truth_history.append((truth,raw))
    async def read_pose():
        while True:
            line=await pose_process.stdout.readline()
            if not line:raise RuntimeError('Pose stream closed')
            pose=pose_record(json.loads(line))
            if pose is not None:poses.append(pose)
    async def read_pairs():
        nonlocal latest_rgb
        async for rgb,depth,skew in source.events(timeout_s=30):
            latest_rgb=rgb.capture_timestamp
            if not active or active.get('result') is not None:continue
            token=active['token']
            # Wall-clock delivery grace does not relax simulation timestamp skew.
            for _ in range(50):
                if active.get('token') is not token:break
                if truth_history and poses and truth_history[-1][0].simulation_timestamp>=rgb.capture_timestamp-.033334 and poses[-1]['timestamp']>=rgb.capture_timestamp-.033334:break
                await asyncio.sleep(.01)
            if active.get('token') is not token:continue
            def rejected(reason):
                active['stable']=0
                counts=active['diagnostics'];counts[reason]=counts.get(reason,0)+1
            if not truth_history or not poses:
                rejected('metadata_missing');continue
            truth,raw=min(truth_history,key=lambda p:abs(p[0].simulation_timestamp-rgb.capture_timestamp));pose=min(poses,key=lambda p:abs(p['timestamp']-rgb.capture_timestamp))
            if not paired_after_move(rgb.capture_timestamp,depth.capture_timestamp,truth.simulation_timestamp,pose['timestamp'],active['fence']):
                rejected('stale_or_unsynchronized_metadata');continue
            if not pose_close(pose,active['view']):
                rejected('pose_outside_tolerance');continue
            view=active['view']
            try:
                validate_point(pose['position'],map_config,role='actual carrier reference point')
                optical=[pose['position'][i] + view['camera_position'][i] - view['position'][i]
                         for i in range(3)]
                validate_point(optical,map_config,role='actual camera optical center')
            except ValueError:
                rejected('actual_camera_illegal');continue
            if not truth.valid:
                rejected('invalid_truth');continue
            if (rgb.width,rgb.height,depth.width,depth.height)!=(1920,1080,640,360):raise ValueError('Unexpected sensor dimensions')
            depth_path=source.staging/depth.payload_relative_path
            values=np.fromfile(depth_path,dtype='<f4')
            if values.size!=640*360:raise ValueError('Depth payload size mismatch')
            valid=np.isfinite(values)&(values>=.2)&(values<=100)
            if not valid.any():rejected('invalid_depth');continue
            active['stable']+=1
            if active['stable']<3:continue
            expected=view['category'];classes={o.class_name for o in truth.objects}
            checks=target_checks(view,raw,instance_map)
            diagnostic_absence = allows_diagnostic_absence(plan, mode)
            if violates_no_target_gate(plan, classes):
                evidence=save_rejected_frame(source.staging,output,view['view_id'],rgb,depth)
                active['result']={**evidence,'view_id':view['view_id'],'status':'rejected','reason':'targets_present_in_negative_frame','expected_class':expected,'expected_object_id':view['object_id'],'target_checks':checks,'observed_classes':sorted(classes),'actual_pose':pose,'camera_position':view['camera_position'],'truth':truth.to_record(),'truth_timestamp':truth.simulation_timestamp,'skew_ms':skew,'raw_truth':raw,'check_version':CHECK_VERSION};continue
            if expected in TARGET_CLASSES and checks['planned_instance_present'] is not True and not diagnostic_absence:
                evidence=save_rejected_frame(source.staging,output,view['view_id'],rgb,depth)
                active['result']={**evidence,'view_id':view['view_id'],'status':'rejected','reason':'planned_instance_absent','expected_class':expected,'expected_class_present':checks['category_present'],'expected_object_id':view['object_id'],'target_checks':checks,'observed_classes':sorted(classes),'actual_pose':pose,'camera_position':view['camera_position'],'truth':truth.to_record(),'truth_timestamp':truth.simulation_timestamp,'skew_ms':skew,'raw_truth':raw,'check_version':CHECK_VERSION};continue
            folder=output/view['view_id'];folder.mkdir()
            shutil.copyfile(source.staging/rgb.payload_relative_path,folder/'rgb.ppm');shutil.copyfile(depth_path,folder/'depth.bin')
            row={'view_id':view['view_id'],'status':'captured','map_id':plan['map_id'],'split':'development','family':view['family'],'expected_object_id':view['object_id'],'expected_category':expected,'expected_class_present':checks['category_present'] if expected in TARGET_CLASSES else None,'target_checks':checks,'camera_position':view['camera_position'],'actual_pose':pose,'rgb_timestamp':rgb.capture_timestamp,'depth_timestamp':depth.capture_timestamp,'truth_timestamp':truth.simulation_timestamp,'skew_ms':skew,'truth':truth.to_record(),'raw_truth':raw,'valid_depth_fraction':float(valid.mean()),'rgb_path':str(folder/'rgb.ppm'),'depth_path':str(folder/'depth.bin'),'image_sha256':file_sha256(folder/'rgb.ppm'),'depth_sha256':file_sha256(folder/'depth.bin'),'perceptual_hash':dhash64(folder/'rgb.ppm'),'review_status':'pending','check_version':CHECK_VERSION,'training_admitted':False}
            active['result']=row
    async def supervised(function):
        try:await function()
        except asyncio.CancelledError:raise
        except Exception as error:fatal.append(f'{type(error).__name__}: {error}')
    def check_fatal():
        if fatal:raise RuntimeError('; '.join(fatal))
    error=None
    staging=tempfile.TemporaryDirectory(prefix='canonical-rgbd-')
    try:
        with (output/'simulator.log').open('x') as log:
            stage=staging.name
            server=await asyncio.create_subprocess_exec('gz','sim','-s','-r',str(base/'world.sdf'),stdout=log,stderr=asyncio.subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+60
            while time.monotonic()<deadline:
                if server.returncode is not None:raise RuntimeError('Gazebo exited during startup')
                try:snapshot=(await command('gz','topic','-l')).splitlines()
                except (RuntimeError,asyncio.TimeoutError):snapshot=[]
                if all(t in snapshot for t in ('/research_camera/image','/research_camera/depth','/research_camera/boxes',f'/world/{plan["world_name"]}/pose/info')):break
                await asyncio.sleep(2)
            else:raise RuntimeError('Required topics not discovered within 60 seconds')
            inspections={}
            for topic,kind in [('/research_camera/image','gz.msgs.Image'),('/research_camera/depth','gz.msgs.Image'),('/research_camera/boxes','gz.msgs.AnnotatedAxisAligned2DBox_V'),(f'/world/{plan["world_name"]}/pose/info','gz.msgs.Pose_V')]:
                detail=await command('gz','topic','-i','-t',topic)
                if kind not in detail:raise ValueError('Unexpected topic type: '+topic)
                inspections[topic]=detail
            write_record(output/'transport.json',{'topics':snapshot,'types':inspections,'partition':partition,'GZ_IP':'127.0.0.1'})
            truth_process=await asyncio.create_subprocess_exec('gz','topic','-e','--json-output','-t','/research_camera/boxes',stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL,limit=16*1024*1024)
            pose_process=await asyncio.create_subprocess_exec('gz','topic','-e','--json-output','-t',f'/world/{plan["world_name"]}/pose/info',stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL,limit=16*1024*1024)
            tasks=[asyncio.create_task(supervised(read_truth)),asyncio.create_task(supervised(read_pose))]
            source=NativeGazeboRgbDepthSource(stage,emit_stride=3,preserve_frames=False);await source.start();tasks.append(asyncio.create_task(supervised(read_pairs)))
            start=time.monotonic()
            while not poses or latest_rgb<0:
                check_fatal()
                if time.monotonic()-start>30:raise TimeoutError('No initial pose/RGB-D')
                await asyncio.sleep(.05)
            for view in views:
                print('VIEW',view['view_id'][:12],view['category'],flush=True)
                x,y,z=view['position'];qx,qy,qz,qw=view['orientation']
                request=f'name: "{CARRIER}" position {{ x: {x} y: {y} z: {z} }} orientation {{ x: {qx} y: {qy} z: {qz} w: {qw} }}'
                async def move():
                    return await command('gz','service','-s',f'/world/{plan["world_name"]}/set_pose','--reqtype','gz.msgs.Pose','--reptype','gz.msgs.Boolean','--timeout','3000','--req',request)
                result=await acquire_view(view,move=move,active=active,
                    get_fence=lambda:max(latest_rgb,poses[-1]['timestamp'],truth_history[-1][0].simulation_timestamp if truth_history else -1),
                    check_fatal=check_fatal)
                results.append(result);write_record(output/(view['view_id']+'.json'),result);active.clear()
                if result['status']!='captured':raise RuntimeError('View rejected: '+result['reason'])
    except (Exception,KeyboardInterrupt,asyncio.CancelledError) as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        for task in tasks:task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        if source is not None:await source.stop()
        await stop(truth_process);await stop(pose_process)
        if server is not None:
            try:os.killpg(server.pid,signal.SIGINT)
            except ProcessLookupError:pass
            try:await asyncio.wait_for(server.wait(),10)
            except asyncio.TimeoutError:
                try:os.killpg(server.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                await server.wait()
        staging.cleanup()
        for k,v in old.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    receipt=write_record(output/'collection-receipt.json',{'schema_version':2,'collector_revision':'bounded-recovery-v2','collection_checks':gate_receipt,'actual_annotation_mode':gate_receipt['actual_annotation_mode'],'check_version':CHECK_VERSION,'world_sha256':gate_receipt['world_sha256'],'resumed_from_collection_identity':resume_identity,'resumed_views':len(recovered),'status':'blocked' if error else 'complete_pending_review','plan_identity':plan['identity'],'map_id':plan['map_id'],'mode':mode,'view_count':len(results),'views':results,'error':error,'source_health':source.receipt() if source else {},'training_admitted':False,'promotable':False,'px4_started':False})
    return receipt
