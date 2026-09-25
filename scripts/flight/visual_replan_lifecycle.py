"""Bounded semantic SITL lifecycle derived from the verified LiDAR lifecycle.

Kept separate so historical flight receipts continue to bind their original code.
The visual service owns bounded RGB-D/pose inference; flight guards and landing
remain the same. New visual flights require independent evidence.
"""
import asyncio
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from scripts.flight.check_px4_shadow import preflight,stop,PX4,BUILD,ROOT
from scripts.maps.prepare_research_vehicle import prepare_world_with_vehicle
from scripts.vision.probe_material_shadow import command
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256
from src.sandbox.live_replan_stop import check_stop
from src.flight.startup_gate import wait_px4_startup

OUT=ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1/pillar-route-001/runtime'

def check_spawn(config,x=-8.5,y=-8.5,radius=1.):
    ox,oy,_=config['gazebo_world_origin_m'];resolution=config['resolution_m'];u=x-ox;v=y-oy
    if not(radius<=u<=config['width']*resolution-radius and radius<=v<=config['height']*resolution-radius):raise ValueError('Envelope outside map')
    for b in config['obstacles']:
        if (b['x_min']*resolution-radius<=u<=(b['x_max']+1)*resolution+radius and b['y_min']*resolution-radius<=v<=(b['y_max']+1)*resolution+radius):raise ValueError('Envelope intersects '+b['name'])

def guard(state,now,origin,max_height_m=1.8,max_drift_m=1.):
    for name in ('local','attitude'):
        if name not in state or now-state[name][0]>1.:raise RuntimeError('Stale '+name)
    p=state['local'][1];a=state['attitude'][1]
    if not all(math.isfinite(v) for v in [now,*origin.values(),*p.values(),*a.values()]):raise RuntimeError('Nonfinite telemetry')
    height=origin['down']-p['down'];drift=math.hypot(p['north']-origin['north'],p['east']-origin['east'])
    if height>max_height_m or height<-.4 or drift>max_drift_m:raise RuntimeError('Position envelope exceeded')
    if abs(a['roll'])>20 or abs(a['pitch'])>20:raise RuntimeError('Attitude envelope exceeded')
    return height,drift

async def main(fly=False, preflight_hook=None, takeoff_altitude=1., post_hover_hook=None, server_config=None, post_hover_timeout_s=5., source_world=None, visual_service=None):
    if visual_service is None:raise ValueError('Explicit visual service required')
    if takeoff_altitude not in (1.,2.):raise ValueError('Only bounded 1m or 2m trials supported')
    if not 0<post_hover_timeout_s<=180:raise ValueError('Post-hover timeout must be bounded')
    binary=preflight();config_path=ROOT/'config/substation_obstacles.json';check_spawn(json.loads(config_path.read_text()))
    if shutil.disk_usage(ROOT).free<1024**3:raise RuntimeError('Need1GiB free for bounded telemetry and visual evidence')
    if source_world is None:raise ValueError('Explicit frozen scenario world required')
    src=Path(source_world).resolve(strict=True)
    from scripts.flight.prepare_avoidance_route import collision_boxes,inside
    if any(inside((1.5,1.5),b,1.) for b in collision_boxes(src)):raise ValueError('Scenario spawn envelope blocked')
    OUT.mkdir(parents=True,exist_ok=False);world=OUT/'world.sdf'
    from scripts.flight.prepare_lowload_vehicle import OUT as profile, MODEL
    identity=json.loads((profile/'protocol.json').read_text())
    for path,digest in identity['inputs'].items():
        if file_sha256(path)!=digest:raise ValueError('Flight vehicle profile changed: '+path)
    prepare_world_with_vehicle(src,world,model_name=MODEL,entity_name='x500_research_0',pose='-8.5,-8.5,0.1,0,0,0')
    os.environ.update(GZ_IP='127.0.0.1',GZ_PARTITION=f'px4_hover_{os.getpid()}',GZ_SIM_RESOURCE_PATH=f'{ROOT}/simulation/models:{PX4}/Tools/simulation/gz/models',GZ_SIM_SYSTEM_PLUGIN_PATH=f'{BUILD}/src/modules/simulation/gz_plugins',GZ_SIM_SERVER_CONFIG_PATH=str(PX4/'Tools/simulation/gz/server.config'),PX4_SYS_AUTOSTART='4001',PX4_SIM_MODEL='gz_x500',PX4_GZ_STANDALONE='1',PX4_GZ_WORLD='substation_simple',PX4_GZ_MODEL_NAME='x500_research_0',HEADLESS='1')
    os.environ['GZ_SIM_RESOURCE_PATH']=str(profile/'models')+':'+os.environ['GZ_SIM_RESOURCE_PATH']
    if server_config is not None:
        os.environ['GZ_SIM_SERVER_CONFIG_PATH']=str(Path(server_config).resolve(strict=True))
    processes=[];files=[];tasks=[];state={};samples=[];events=[];errors=[];phase='startup';drone=None;origin=None;failure=None;armed_requested=False;landed=False;vision=None
    log=(OUT/'telemetry.jsonl').open('x')
    def event(name,**kw):
        r=dict(event=name,monotonic=time.monotonic(),phase=phase,**kw);events.append(r);print(name,flush=True)
    async def launch(args,name,cwd=None):
        f=(OUT/name).open('x');files.append(f);p=await asyncio.create_subprocess_exec(*map(str,args),stdout=f,stderr=asyncio.subprocess.STDOUT,cwd=cwd,start_new_session=True);processes.append(p);return p
    async def monitor(name,stream,convert):
        try:
            async for value in stream:
                now=time.monotonic();value=convert(value);state[name]=(now,value);r=dict(stream=name,monotonic=now,phase=phase,value=value);samples.append(r);log.write(json.dumps(r)+'\n');log.flush()
        except Exception as e:errors.append(f'{name}: {e}')
    async def wait_until(predicate,timeout,stable=0.,enforce=False):
        end=time.monotonic()+timeout;since=None
        while time.monotonic()<end:
            if phase not in ('landing','failsafe_landing'):check_stop(OUT.parent)
            if errors:raise RuntimeError('; '.join(errors))
            if phase=='vision_startup' and vision is not None and vision.returncode is not None:raise RuntimeError('Vision startup failed; see vision.log')
            if any(p.returncode is not None for p in processes[:2]):raise RuntimeError('Simulator or PX4 exited')
            if enforce:
                guard(state,time.monotonic(),origin,max_height_m=takeoff_altitude+.8,max_drift_m=6. if phase=='post_hover_trial' else 1.)
                if vision.returncode is not None:raise RuntimeError('Vision exited in flight')
            if predicate():
                since=since or time.monotonic()
                if time.monotonic()-since>=stable:return
            else:since=None
            await asyncio.sleep(.1)
        raise TimeoutError('Phase timed out: '+phase)
    def grounded():
        if origin is None or 'local' not in state:return False
        p=state['local'][1]
        return state.get('in_air',(0,None))[1] is False and abs(origin['down']-p['down'])<.25 and abs(p['vd'])<.15 and time.monotonic()-state['local'][0]<1
    try:
        await launch(['gz','sim','-s','-r',world],'gazebo.log')
        end=time.monotonic()+45
        while '/research_camera/image' not in await command('gz','topic','-l'):
            if time.monotonic()>end:raise TimeoutError('No camera')
            await asyncio.sleep(1)
        work=OUT/'px4-work';work.mkdir()
        startup=OUT/'startup-trace.sh';startup.write_text('#!/bin/sh\nset -x\n. "'+str(BUILD/'etc/init.d-posix/rcS')+'"\n')
        await launch([binary,BUILD/'etc','-i','7','-d','-w',work,'-s',startup],'px4.log',work)
        phase='px4_startup_gate';elapsed=await wait_px4_startup(OUT/'px4.log',processes[:2],check_cancel=lambda:check_stop(OUT.parent));event('px4_startup_verified',elapsed_s=elapsed)
        check_stop(OUT.parent)
        from mavsdk import System
        phase='mavsdk_connect';event('mavsdk_connect_started')
        drone=System(port=50177);await asyncio.wait_for(drone.connect(system_address='udpin://127.0.0.1:14547'),45)
        phase='vehicle_connect';event('mavsdk_connected')
        async def connect():
            async for c in drone.core.connection_state():
                if c.is_connected:return
        await asyncio.wait_for(connect(),50);event('vehicle_connected')
        sources=[('armed',drone.telemetry.armed(),bool),('in_air',drone.telemetry.in_air(),bool),
            ('local',drone.telemetry.position_velocity_ned(),lambda v:dict(north=v.position.north_m,east=v.position.east_m,down=v.position.down_m,vn=v.velocity.north_m_s,ve=v.velocity.east_m_s,vd=v.velocity.down_m_s)),
            ('attitude',drone.telemetry.attitude_euler(),lambda v:dict(roll=v.roll_deg,pitch=v.pitch_deg,yaw=v.yaw_deg)),
            ('health',drone.telemetry.health(),lambda v:dict(local=v.is_local_position_ok,global_position=v.is_global_position_ok,home=v.is_home_position_ok,armable=v.is_armable)),
            ('mode',drone.telemetry.flight_mode(),str)]
        tasks=[asyncio.create_task(monitor(*x)) for x in sources]
        phase='preflight'
        def ready():
            return all(state.get('health',(0,{}))[1].values()) and len(state.get('health',(0,{}))[1])==4 and state.get('armed',(0,None))[1] is False and state.get('in_air',(0,None))[1] is False and 'local' in state and 'attitude' in state and time.monotonic()-state['local'][0]<1
        await wait_until(ready,45,stable=3);origin=dict(state['local'][1]);guard(state,time.monotonic(),origin)
        if preflight_hook is not None:
            await preflight_hook(drone,state,origin)
        if not fly:
            await wait_until(ready,20,stable=10)
            for topic in ('estimator_status','vehicle_global_position','vehicle_gps_position','failsafe_flags'):
                result=await command(str(BUILD/'bin/px4-listener'),'--instance','7',topic,'-n','1')
                (OUT/(topic+'.txt')).write_text(result)
            event('unarmed_health_check_passed')
            return
        phase='vision_startup'
        vision=visual_service
        await asyncio.wait_for(vision.start(drone),30)
        await wait_until(vision.ready.is_set,20)
        await asyncio.wait_for(drone.action.set_takeoff_altitude(takeoff_altitude),5)
        altitude=await asyncio.wait_for(drone.action.get_takeoff_altitude(),5)
        if not math.isfinite(altitude) or abs(altitude-takeoff_altitude)>.01:raise ValueError('Takeoff altitude mismatch')
        if not ready():raise RuntimeError('Readiness lost')
        check_stop(OUT.parent)
        phase='arming';armed_requested=True;event('arm_requested');await asyncio.wait_for(drone.action.arm(),5)
        await wait_until(lambda:state.get('armed',(0,False))[1] is True,5)
        phase='takeoff';event('takeoff_requested',altitude_m=takeoff_altitude);await asyncio.wait_for(drone.action.takeoff(),5)
        def hover():
            height,drift=guard(state,time.monotonic(),origin,max_height_m=takeoff_altitude+.8);return takeoff_altitude-.25<=height<=takeoff_altitude+.25 and abs(state['local'][1]['vd'])<.2 and state.get('in_air',(0,False))[1] is True
        await wait_until(hover,25,stable=2,enforce=True)
        phase='hover';event('hover_started');await wait_until(hover,20,stable=10,enforce=True);event('hover_complete')
        if post_hover_hook is not None:
            phase='post_hover_trial'
            hook=asyncio.create_task(post_hover_hook(drone,state,origin))
            try:
                await wait_until(hook.done,post_hover_timeout_s,enforce=True)
                await hook
            finally:
                if not hook.done():hook.cancel()
                await asyncio.gather(hook,return_exceptions=True)
        phase='landing';event('land_requested');await asyncio.wait_for(drone.action.land(),5)
        await wait_until(grounded,40,stable=3);landed=True;event('landing_confirmed')
        if state.get('armed',(0,False))[1]:await asyncio.wait_for(drone.action.disarm(),5)
        await wait_until(lambda:state.get('armed',(0,True))[1] is False,8);event('disarmed_confirmed')
        phase='postflight'
        await vision.stop()
        if vision.error:raise RuntimeError('Vision failed: '+vision.error)
    except BaseException as e:
        failure=f'{type(e).__name__}: {e}';event('trial_failed',error=failure)
        if drone is not None and armed_requested and not landed:
            phase='failsafe_landing'
            try:
                await asyncio.wait_for(drone.action.land(),5);await wait_until(grounded,40,stable=3);landed=True
                if state.get('armed',(0,False))[1]:await asyncio.wait_for(drone.action.disarm(),5)
                await wait_until(lambda:state.get('armed',(0,True))[1] is False,8)
                event('failsafe_landing_confirmed')
            except Exception as landing_error:event('landing_unconfirmed',error=str(landing_error))
    finally:
        if vision is not None:
            try:await vision.stop()
            except Exception as cleanup_error:
                failure=failure or ('Visual cleanup failed: '+str(cleanup_error))
        for t in tasks:t.cancel()
        await asyncio.gather(*tasks,return_exceptions=True);log.close()
        if drone is not None and drone._server_process is not None:
            server=drone._server_process
            if server.poll() is None:server.kill()
            server.wait(timeout=5);drone._server_process=None
        for p in reversed(processes):await stop(p)
        for f in files:f.close()
        paths=[world,src,binary,config_path,OUT/'telemetry.jsonl',Path(__file__).resolve(),OUT/'px4.log']
        paths += [p for p in [OUT/'startup-trace.sh'] if p.exists()]
        write_record(OUT/'receipt.json',dict(status='failed' if failure else ('sitl_hover_landed_disarmed' if fly else 'unarmed_health_check_passed'),error=failure,events=events,origin=origin,landing_confirmed=landed,final_armed=state.get('armed',(0,None))[1],simulation_only=True,flight_requested=fly,vision_control_authority='bounded_semantic_request',owned_processes_exited=all(p.returncode is not None for p in processes),training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths}))
    print(failure or 'SITL_HOVER_LANDED_DISARMED',flush=True)
