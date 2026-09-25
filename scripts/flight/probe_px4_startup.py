"""Bounded unarmed startup trace; no action or offboard commands exist here."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time
from scripts.flight.check_px4_shadow import ROOT,PX4,BUILD,preflight,stop
from scripts.flight.prepare_lowload_vehicle import OUT as PROFILE,MODEL
from scripts.maps.prepare_research_vehicle import prepare_world_with_vehicle
from scripts.vision.probe_material_shadow import command
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

BASE=ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1'

async def run(attempt):
    binary=preflight();out=BASE/f'px4-startup-probe-v1-{attempt}';out.mkdir(exist_ok=False)
    world=out/'world.sdf';prepare_world_with_vehicle(BASE/'envelope-aware-scene-v1/world.sdf',world,model_name=MODEL,entity_name='x500_research_0',pose='-8.5,-8.5,0.1,0,0,0')
    startup=out/'startup-trace.sh';original=BUILD/'etc/init.d-posix/rcS'
    startup.write_text('#!/bin/sh\nset -x\n. "'+str(original)+'"\n')
    os.environ.update(GZ_IP='127.0.0.1',GZ_PARTITION=f'px4_startup_probe_{os.getpid()}',GZ_SIM_RESOURCE_PATH=f'{PROFILE}/models:{ROOT}/simulation/models:{PX4}/Tools/simulation/gz/models',GZ_SIM_SYSTEM_PLUGIN_PATH=f'{BUILD}/src/modules/simulation/gz_plugins',GZ_SIM_SERVER_CONFIG_PATH=str(BASE/'magnetic-contract-001/server.config'),PX4_SYS_AUTOSTART='4001',PX4_SIM_MODEL='gz_x500',PX4_GZ_STANDALONE='1',PX4_GZ_WORLD='substation_simple',PX4_GZ_MODEL_NAME='x500_research_0',HEADLESS='1',PX4_GZ_MAG_ENU_GAUSS='1',PX4_PARAM_EKF2_DECL_TYPE='3',PX4_PARAM_EKF2_MAG_DECL='0')
    processes=[];handles=[];drone=None;error=None;events=[];armed=[]
    def event(name,**fields):events.append(dict(event=name,monotonic=time.monotonic(),**fields));print(name,fields,flush=True)
    async def launch(argv,name,cwd=None):
        handle=(out/name).open('x');handles.append(handle)
        process=await asyncio.create_subprocess_exec(*map(str,argv),cwd=cwd,stdout=handle,stderr=asyncio.subprocess.STDOUT,start_new_session=True);processes.append(process);return process
    try:
        gz=await launch(['gz','sim','-s','-r',world],'gazebo.log')
        deadline=time.monotonic()+45
        while '/research_camera/image' not in await command('gz','topic','-l'):
            if time.monotonic()>deadline or gz.returncode is not None:raise TimeoutError('Camera startup')
            await asyncio.sleep(.5)
        work=out/'px4-work';work.mkdir()
        px4=await launch([binary,BUILD/'etc','-i','7','-d','-w',work,'-s',startup],'px4.log',work)
        start=time.monotonic();event('px4_started',pid=px4.pid)
        while 'Startup script returned successfully' not in (out/'px4.log').read_text():
            if px4.returncode is not None or gz.returncode is not None:raise RuntimeError('Simulator exited before startup complete')
            if time.monotonic()-start>90:raise TimeoutError('PX4 boot did not complete within 90 seconds')
            await asyncio.sleep(.2)
        event('px4_boot_complete',elapsed_s=time.monotonic()-start)
        from mavsdk import System
        drone=System(port=50177);start=time.monotonic()
        await asyncio.wait_for(drone.connect(system_address='udpin://127.0.0.1:14547'),45)
        event('mavsdk_connected',elapsed_s=time.monotonic()-start)
        async def watch():
            async for value in drone.telemetry.armed():
                armed.append(dict(monotonic=time.monotonic(),armed=value))
                if value:raise RuntimeError('Unexpected armed state in no-arm probe')
        task=asyncio.create_task(watch())
        try:
            await asyncio.sleep(10)
            if task.done():task.result()
            if not armed:raise RuntimeError('No disarmed telemetry')
        finally:task.cancel();await asyncio.gather(task,return_exceptions=True)
        event('unarmed_probe_complete')
    except BaseException as exc:error=f'{type(exc).__name__}: {exc}';event('probe_failed',error=error)
    finally:
        if drone is not None and drone._server_process is not None:
            process=drone._server_process
            if process.poll() is None:process.kill()
            process.wait(timeout=5);drone._server_process=None
        for process in reversed(processes):await stop(process)
        for handle in handles:handle.close()
        inputs=[world,startup,original,binary,out/'px4.log',out/'gazebo.log',PROFILE/'protocol.json',Path(__file__).resolve()]
        write_record(out/'completion.json',dict(status='unarmed_startup_verified' if error is None else 'failed',error=error,events=events,armed_samples=armed,arm_commands_sent=0,flight_commands_sent=0,owned_processes_exited=all(p.returncode is not None for p in processes),training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in inputs}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,choices=(1,2,3),required=True);args=parser.parse_args();asyncio.run(run(args.attempt))
