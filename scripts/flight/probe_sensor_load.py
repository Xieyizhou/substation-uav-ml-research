"""Static-vehicle, no-PX4 sensor-load comparison. Does not authorize flight."""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import xml.etree.ElementTree as ET

from scripts.flight.check_px4_shadow import ROOT,PX4,BUILD,stop
from scripts.maps.prepare_research_vehicle import prepare_world_with_vehicle
from scripts.vision.probe_material_shadow import command
from src.sensors.gazebo_lidar import GazeboLidar2DSource
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

BASE=ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1'
OUT=BASE/'sensor-load-probe-v1'


def prepare(profile):
    out=OUT/profile;out.mkdir(parents=True,exist_ok=False)
    model=out/'models/x500_research_probe';model.mkdir(parents=True)
    source=ROOT/'simulation/models/x500_research/model.sdf';tree=ET.parse(source)
    tree.getroot().find('model').set('name','x500_research_probe')
    if profile=='rgb640':
        sensor=tree.getroot().find("model/link[@name='research_camera_link']/sensor[@name='research_rgb']")
        sensor.find('camera/image/width').text='640';sensor.find('camera/image/height').text='360'
    tree.write(model/'model.sdf',encoding='utf-8',xml_declaration=True)
    config=ET.Element('model');ET.SubElement(config,'name').text='x500_research_probe';ET.SubElement(config,'version').text='1';ET.SubElement(config,'sdf',version='1.9').text='model.sdf';ET.ElementTree(config).write(model/'model.config')
    src=BASE/'envelope-aware-scene-v1/world.sdf';world=out/'world.sdf'
    prepare_world_with_vehicle(src,world,model_name='x500_research_probe',entity_name='x500_research_0',pose='-6.1,-7,2,0,0,1.57')
    tree=ET.parse(world);include=tree.getroot().find('world/include');ET.SubElement(include,'static').text='true';tree.write(world,encoding='utf-8',xml_declaration=True)
    paths=[source,src,world,model/'model.sdf',model/'model.config',Path(__file__).resolve()]
    write_record(out/'protocol.json',dict(profile=profile,only_sensor_change='RGB resolution 1920x1080 to 640x360' if profile=='rgb640' else 'none',px4_started=False,static_vehicle=True,lidar_unchanged=True,vision_threads=4,observe_s=60,training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths}))
    return out,world


async def run(profile):
    out,world=prepare(profile)
    os.environ.update(GZ_IP='127.0.0.1',GZ_PARTITION=f'sensor_load_probe_{os.getpid()}',GZ_SIM_RESOURCE_PATH=f'{out}/models:{ROOT}/simulation/models:{PX4}/Tools/simulation/gz/models',GZ_SIM_SYSTEM_PLUGIN_PATH=f'{BUILD}/src/modules/simulation/gz_plugins',GZ_SIM_SERVER_CONFIG_PATH=str(BASE/'magnetic-contract-001/server.config'))
    handles=[];processes=[];source=GazeboLidar2DSource(stale_after_s=.5);error=None;rows=[];events=[]
    async def launch(argv,name):
        f=(out/name).open('x');handles.append(f)
        p=await asyncio.create_subprocess_exec(*map(str,argv),stdout=f,stderr=asyncio.subprocess.STDOUT,start_new_session=True);processes.append(p);return p
    try:
        gz=await launch(['gz','sim','-s','-r',world],'gazebo.log')
        deadline=time.monotonic()+45
        while '/research_camera/image' not in await command('gz','topic','-l'):
            if time.monotonic()>deadline or gz.returncode is not None:raise RuntimeError('Static sensors unavailable')
            await asyncio.sleep(1)
        await source.start();await source.wait_ready(10)
        vision=await launch([sys.executable,'-m','scripts.vision.material_shadow_route','--seconds','90','--output',out/'vision'],'vision.log')
        await asyncio.sleep(15)
        start=time.monotonic();step=0
        while time.monotonic()-start<60:
            now=time.monotonic();scan=source.latest();health=source.health()
            rows.append(dict(monotonic=now,sequence=scan.sequence if scan else None,capture_s=scan.timestamp_s if scan else None,receive_s=scan.received_monotonic_s if scan else None,healthy=health.healthy,message=health.message,age_s=health.last_frame_age_s,dropped=health.dropped_frames))
            if step<4 and now-start>=step*15:
                yaw=step*math.pi/2
                req=f'name: "x500_research_0" position {{ x: -6.1 y: -7 z: 2 }} orientation {{ w: {math.cos(yaw/2)} z: {math.sin(yaw/2)} }}'
                p=await asyncio.create_subprocess_exec('gz','service','-s','/world/substation_simple/set_pose','--reqtype','gz.msgs.Pose','--reptype','gz.msgs.Boolean','--timeout','2000','--req',req,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
                try:stdout,stderr=await asyncio.wait_for(p.communicate(),3)
                finally:
                    if p.returncode is None:p.kill();await p.wait()
                if p.returncode or 'data: true' not in stdout.decode():raise RuntimeError('Static pose update failed: '+stderr.decode())
                events.append(dict(yaw=yaw,monotonic=now));step+=1
            if gz.returncode is not None or vision.returncode is not None:raise RuntimeError('Probe process exited early')
            await asyncio.sleep(.02)
        await asyncio.wait_for(vision.wait(),60)
        if vision.returncode:raise RuntimeError('Vision sidecar failed')
    except BaseException as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        await source.stop()
        for p in reversed(processes):await stop(p)
        for f in handles:f.close()
        scans=[r for i,r in enumerate(rows) if r['sequence'] is not None and (i==0 or rows[i-1]['sequence']!=r['sequence'])]
        gaps=[b['receive_s']-a['receive_s'] for a,b in zip(scans,scans[1:])]
        write_record(out/'samples.json',dict(rows=rows,poses=events))
        write_record(out/'completion.json',dict(status='probe_complete' if error is None else 'failed',error=error,px4_started=False,flight_started=False,max_observed_lidar_receive_gap_s=max(gaps,default=None),unhealthy_samples=sum(not r['healthy'] for r in rows),owned_processes_exited=all(p.returncode is not None for p in processes),limitation='static transport/load experiment; not a successful flight or universal timing guarantee',training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in [out/'protocol.json',out/'samples.json']}))
    print(profile,error or 'probe_complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',choices=('rgb1920','rgb640'),required=True);a=p.parse_args();asyncio.run(run(a.profile))
