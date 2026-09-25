"""Bounded local SITL telemetry/vision concurrency test. Never arm or fly."""
import asyncio
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from scripts.maps.prepare_research_vehicle import prepare_world_with_vehicle
from scripts.vision.probe_material_shadow import command
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

PX4=Path('/Users/xieyizhou/PX4-Autopilot');BUILD=PX4/'build/px4_sitl_default'
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/research/material-shadow-v1/px4-unarmed-concurrency-v1/attempt-001'

def preflight():
    conflict=subprocess.run(['pgrep','-x','px4'],capture_output=True,text=True)
    if conflict.returncode==0:raise RuntimeError('Existing PX4 instance; refusing to share it')
    for kind,port in ((socket.SOCK_DGRAM,14547),(socket.SOCK_DGRAM,14587),(socket.SOCK_STREAM,50177)):
        with socket.socket(socket.AF_INET,kind) as sock:sock.bind(('127.0.0.1',port))
    return BUILD/'bin/px4'

async def stop(p):
    if p is None:return
    if p.returncode is None:
        os.killpg(p.pid,signal.SIGINT)
        try:await asyncio.wait_for(p.wait(),8)
        except asyncio.TimeoutError:os.killpg(p.pid,signal.SIGKILL);await p.wait()

async def main():
    binary=preflight();OUT.mkdir(parents=True,exist_ok=False)
    world=OUT/'world.sdf';src=ROOT/'simulation/worlds/substation_simple.sdf'
    prepare_world_with_vehicle(src,world,model_name='x500_research',entity_name='x500_research_0',pose='-10,-10,0.1,0,0,0')
    os.environ.update(GZ_IP='127.0.0.1',GZ_PARTITION=f'px4_shadow_{os.getpid()}',GZ_SIM_RESOURCE_PATH=f'{ROOT}/simulation/models:{PX4}/Tools/simulation/gz/models',GZ_SIM_SYSTEM_PLUGIN_PATH=f'{BUILD}/src/modules/simulation/gz_plugins',GZ_SIM_SERVER_CONFIG_PATH=str(PX4/'Tools/simulation/gz/server.config'),PX4_SYS_AUTOSTART='4001',PX4_SIM_MODEL='gz_x500',PX4_GZ_STANDALONE='1',PX4_GZ_WORLD='substation_simple',PX4_GZ_MODEL_NAME='x500_research_0',HEADLESS='1')
    processes=[];files=[];tasks=[];drone=None;phase='startup';samples=[];errors=[];failure=None
    async def launch(args,name,cwd=None):
        f=(OUT/name).open('x');files.append(f)
        p=await asyncio.create_subprocess_exec(*map(str,args),cwd=cwd,stdout=f,stderr=asyncio.subprocess.STDOUT,start_new_session=True);processes.append(p);return p
    async def monitor(name,stream):
        try:
            async for value in stream:
                payload=value if isinstance(value,(bool,int,float,str)) else str(value)
                samples.append(dict(stream=name,phase=phase,monotonic=time.monotonic(),value=payload))
                if name in ('armed','in_air') and value:errors.append('Unexpected '+name);return
        except Exception as e:errors.append(f'{name}: {e}')
    async def watch(seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if errors:raise RuntimeError('; '.join(errors))
            if any(p.returncode is not None for p in processes[:2]):raise RuntimeError('SITL or Gazebo exited')
            await asyncio.sleep(.2)
    try:
        gz=await launch(['gz','sim','-s','-r',world],'gazebo.log')
        end=time.monotonic()+45
        while time.monotonic()<end:
            if gz.returncode is not None:raise RuntimeError('Gazebo exited')
            topics=await command('gz','topic','-l')
            if '/research_camera/image' in topics:break
            await asyncio.sleep(1)
        else:raise TimeoutError('Research camera not discovered')
        work=OUT/'px4-work';work.mkdir()
        await launch([binary,str(BUILD/'etc'),'-i','7','-d','-w',work,'-s',BUILD/'etc/init.d-posix/rcS'],'px4.log',work)
        from mavsdk import System
        drone=System(port=50177)
        await asyncio.wait_for(drone.connect(system_address='udpin://127.0.0.1:14547'),20)
        async def connected():
            async for state in drone.core.connection_state():
                if state.is_connected:return
        await asyncio.wait_for(connected(),45)
        for name,stream in [('armed',drone.telemetry.armed()),('in_air',drone.telemetry.in_air()),('position',drone.telemetry.position()),('attitude',drone.telemetry.attitude_euler()),('health',drone.telemetry.health())]:tasks.append(asyncio.create_task(monitor(name,stream)))
        await watch(5)
        if not any(r['stream']=='armed' and r['value'] is False for r in samples):raise RuntimeError('Unarmed state not verified')
        phase='baseline';await watch(15)
        phase='vision'
        sidecar=await launch([sys.executable,'-m','scripts.vision.material_shadow_latest','--mode','live','--seconds','20','--topic','/research_camera/image','--output',OUT/'vision'],'vision.log',ROOT)
        end=time.monotonic()+55
        while sidecar.returncode is None and time.monotonic()<end:await watch(.5)
        if sidecar.returncode!=0:raise RuntimeError('Vision failed or timed out')
        phase='recovery';await watch(10)
        for name in ('position','attitude'):
            for ph in ('baseline','vision','recovery'):
                if sum(r['stream']==name and r['phase']==ph for r in samples)<3:raise RuntimeError('Insufficient telemetry '+name+' '+ph)
    except Exception as e:failure=f'{type(e).__name__}: {e}'
    finally:
        for t in tasks:t.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        if drone is not None and drone._server_process is not None:
            server=drone._server_process
            if server.poll() is None:server.kill()
            server.wait(timeout=5);drone._server_process=None
        for p in reversed(processes):await stop(p)
        for f in files:f.close()
        write_record(OUT/'telemetry.json',dict(samples=samples,training_admitted=False,promotable=False))
        paths=[src,world,binary,OUT/'telemetry.json',Path(__file__).resolve(),ROOT/'simulation/models/x500_research/model.sdf']
        write_record(OUT/'receipt.json',dict(status='blocked' if failure else 'unarmed_sitl_concurrency_complete',error=failure,arm_commands_sent=0,flight_commands_sent=0,flight_tested=False,owned_processes_exited=all(p.returncode is not None for p in processes),partition=os.environ['GZ_PARTITION'],training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths}))
    print(failure or 'UNARMED_SITL_CONCURRENCY_COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,choices=(1,2,3),default=1);args=parser.parse_args()
    OUT=OUT.parent/f'attempt-{args.attempt:03}'
    asyncio.run(main())
