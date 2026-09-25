"""Explicit SITL-only static-wall braking experiment; never physical flight."""
import asyncio
import json
import math
import os
from pathlib import Path
import time
from mavsdk.offboard import VelocityNedYaw
from scripts.flight import fly_short_route as route
from scripts.flight import fly_px4_shadow_hover as trial
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256

OUT=route.OUT.parent/'obstacle-stop-001'

def nearest(source):
    scan=source.latest()
    if scan is None or not source.health().healthy or scan.age_s(time.monotonic())>.5:
        raise RuntimeError('Missing/stale LiDAR')
    values=scan.ranges_m
    if len(values)<100 or any(math.isnan(v) or v<scan.range_min_m for v in values):
        raise RuntimeError('Malformed or below-minimum scan')
    valid=[v for v in values if math.isfinite(v) and v<=scan.range_max_m]
    if not valid:raise RuntimeError('No finite returns')
    return min(valid)

async def fixture():
    # Spawn is (-8.5,-8.5); near wall face is 1.7 m east of spawn.
    sdf='<sdf version="1.9"><model name="braking_wall"><static>true</static><pose>-6.7 -8.5 2 0 0 0</pose><link name="wall"><collision name="collision"><geometry><box><size>0.2 2 4</size></box></geometry></collision><visual name="visual"><geometry><box><size>0.2 2 4</size></box></geometry></visual></link></model></sdf>'
    (OUT/'fixture.sdf').write_text(sdf)
    process=await asyncio.create_subprocess_exec('/opt/homebrew/bin/gz','service','-s','/world/substation_simple/create','--reqtype','gz.msgs.EntityFactory','--reptype','gz.msgs.Boolean','--timeout','3000','--req','sdf: '+json.dumps(sdf),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    try:
        stdout,stderr=await asyncio.wait_for(process.communicate(),5)
    finally:
        if process.returncode is None:process.kill();await process.wait()
    write_record(OUT/'fixture-receipt.json',dict(returncode=process.returncode,stdout=stdout.decode(),stderr=stderr.decode(),sha256=file_sha256(OUT/'fixture.sdf')))
    if process.returncode or 'true' not in stdout.decode():raise RuntimeError('Fixture creation failed')

async def main():
    prior=route.OUT
    if json.loads((prior/'completion.json').read_text())['status']!='short_route_verified':raise RuntimeError('Short route prerequisite missing')
    config=OUT.parent/'magnetic-contract-001/server.config'
    old=json.loads((prior/'protocol.json').read_text())
    for p in (config,trial.BUILD/'bin/px4'):
        if file_sha256(p)!=old['inputs'][str(p)]:raise RuntimeError('Verified input changed')
    route.validate_contract(config)
    OUT.mkdir(exist_ok=False)
    write_record(OUT/'protocol.json',dict(simulation_only=True,speed_command_m_s=.2,stop_range_m=1.5,stopped_speed_m_s=.1,stopped_stable_s=1.,stop_timeout_s=4.,takeoff_m=2.,horizontal_envelope_m=1.,max_approach_s=6.,vision_control_authority='none',training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in (Path(__file__).resolve(),Path(trial.__file__).resolve(),config,trial.BUILD/'bin/px4',prior/'completion.json')}))
    source=route.GazeboLidar2DSource(stale_after_s=.5)
    async def preflight(drone,state,origin):
        await route.prearm.gate(drone,state,origin)
        await fixture()
        await source.start();await source.wait_ready(10)
        await asyncio.sleep(1)
        if not 1.5<nearest(source)<2.2:raise RuntimeError('Fixture not detected at expected safe initial range')
    async def braking(drone,state,origin):
        await route.airborne.post_hover(drone,state,origin)
        rows=[];active=False;trigger=None;stable=None;error=None;moving=False
        yaw=state['attitude'][1]['yaw'];deadline=time.monotonic()+6
        try:
            await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,yaw))
            await asyncio.wait_for(drone.offboard.start(),3);active=True
            while True:
                now=time.monotonic();trial.guard(state,now,origin,max_height_m=2.8)
                distance=nearest(source);p=state['local'][1].copy();speed=math.hypot(p['vn'],p['ve'])
                moving=moving or speed>=.15
                if trigger is None and distance<=1.5:
                    if not moving:raise RuntimeError('Obstacle triggered before verified motion')
                    trigger=now;print('obstacle_stop_triggered',distance,flush=True)
                if trigger is None and now>deadline:raise TimeoutError('Obstacle not triggered within approach budget')
                if trigger is not None and now-trigger>4:raise TimeoutError('Actual stop not confirmed')
                ve=.2 if trigger is None else 0.
                await asyncio.wait_for(drone.offboard.set_velocity_ned(VelocityNedYaw(0,ve,0,yaw)),1)
                rows.append(dict(monotonic=now,position=p,lidar_nearest_m=distance,speed_m_s=speed,command_east_m_s=ve,triggered=trigger is not None))
                if trigger is not None and speed<.1:
                    stable=stable or now
                    if now-stable>=1:break
                else:stable=None
                await asyncio.sleep(.05)
        except BaseException as exc:error=f'{type(exc).__name__}: {exc}';raise
        finally:
            try:
                if active:
                    await asyncio.wait_for(drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,yaw)),1)
                    await asyncio.wait_for(drone.offboard.stop(),3)
            finally:write_record(OUT/'braking.json',dict(status='failed' if error else 'actual_stop_observed',error=error,trigger_monotonic=trigger,motion_observed=moving,samples=rows,training_admitted=False,promotable=False))
    os.environ.update(PX4_GZ_MAG_ENU_GAUSS='1',PX4_PARAM_EKF2_DECL_TYPE='3',PX4_PARAM_EKF2_MAG_DECL='0')
    route.prearm.OUT=route.airborne.OUT=OUT;trial.OUT=OUT/'runtime'
    try:await trial.main(fly=True,preflight_hook=preflight,takeoff_altitude=2.,post_hover_hook=braking,server_config=config,post_hover_timeout_s=20.)
    finally:await source.stop()

if __name__=='__main__':asyncio.run(main())
