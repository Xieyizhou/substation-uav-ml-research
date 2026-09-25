"""Bounded SITL-only offboard route with live LiDAR stop protection."""
import asyncio
import json
import math
import os
from pathlib import Path
import time
from mavsdk.offboard import VelocityNedYaw
from scripts.flight import fly_px4_shadow_hover as trial
from scripts.flight import retest_sim_heading as prearm
from scripts.flight import check_final_heading as airborne
from scripts.flight.retest_magnetic_contract import validate_contract
from src.planner.local_frame import LocalFrame
from src.sensors.gazebo_lidar import GazeboLidar2DSource
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256

OUT=trial.ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1/short-route-001'


def lidar_clearance(source):
    scan=source.latest();health=source.health()
    if scan is None or not health.healthy or scan.age_s(time.monotonic())>.5:raise RuntimeError('LiDAR missing or stale')
    values=scan.ranges_m
    if len(values)<100 or any(math.isnan(v) or v<0 for v in values):raise RuntimeError('Malformed scan')
    valid=[v for v in values if math.isfinite(v) and scan.range_min_m<=v<=scan.range_max_m]
    # Known populated scene must provide usable returns; do not accept empty scans.
    if not valid:raise RuntimeError('No valid LiDAR returns')
    nearest=min(valid)
    if nearest<1.5:raise RuntimeError('LiDAR obstacle inside conservative 1.5m stop envelope')
    return nearest


def velocity(north_error,east_error):
    n,e=.7*north_error,.7*east_error
    scale=max(1.,math.hypot(n,e)/.3)
    return n/scale,e/scale


async def main():
    prior=OUT.parent/'enu-magnetic-flight-001'
    receipt=json.loads((prior/'runtime/receipt.json').read_text())
    alignment=json.loads((prior/'inair-alignment.json').read_text())
    protocol=json.loads((prior/'protocol.json').read_text())
    config=OUT.parent/'magnetic-contract-001/server.config'
    if receipt['status']!='sitl_hover_landed_disarmed' or not alignment['horizontal_flight_ready']:raise ValueError('Prior flight not passed')
    for p in (trial.BUILD/'bin/px4',config):
        if file_sha256(p)!=protocol['inputs'][str(p)]:raise ValueError('Verified input changed: '+str(p))
    validate_contract(config)
    OUT.mkdir(parents=True,exist_ok=False)
    write_record(OUT/'protocol.json',dict(simulation_only=True,takeoff_altitude_m=2.,speed_cap_m_s=.3,
        map_targets_east_north_m=[[2.,1.5],[2.,2.],[1.5,1.5]],horizontal_envelope_m=1.,
        lidar_stop_distance_m=1.5,lidar_max_age_s=.5,waypoint_tolerance_m=.12,waypoint_stable_s=.6,
        timeout_s=45.,vision_control_authority='none',training_admitted=False,promotable=False,
        inputs={str(p):file_sha256(p) for p in [prior/'runtime/receipt.json',prior/'inair-alignment.json',config,Path(__file__).resolve(),Path(trial.__file__).resolve(),trial.BUILD/'bin/px4']}))
    source=GazeboLidar2DSource(stale_after_s=.5)
    async def preflight(drone,state,origin):
        await prearm.gate(drone,state,origin)
        await source.start();await source.wait_ready(10);lidar_clearance(source)

    async def route(drone,state,origin):
        await airborne.post_hover(drone,state,origin)
        calibration=json.loads((OUT/'inair-alignment.json').read_text())
        frame=LocalFrame.from_mapping(calibration['local_frame'])
        targets=[frame.to_local(e,n) for e,n in ((2.,1.5),(2.,2.),(1.5,1.5))]
        yaw=state['attitude'][1]['yaw'];active=False;rows=[];error=None;reached=[]
        try:
            lidar_clearance(source)
            await drone.offboard.set_velocity_ned(VelocityNedYaw(0.,0.,0.,yaw))
            await asyncio.wait_for(drone.offboard.start(),5);active=True
            for index,(east,north) in enumerate(targets):
                deadline=time.monotonic()+12;since=None
                while time.monotonic()<deadline:
                    now=time.monotonic();trial.guard(state,now,origin,max_height_m=2.8)
                    nearest=lidar_clearance(source);p=state['local'][1]
                    ne,ee=north-p['north'],east-p['east'];distance=math.hypot(ne,ee)
                    if distance<.12 and math.hypot(p['vn'],p['ve'])<.12:
                        since=since or now
                        if now-since>=.6:break
                    else:since=None
                    vn,ve=velocity(ne,ee);vd=max(-.2,min(.2,.7*(origin['down']-2.-p['down'])))
                    await asyncio.wait_for(drone.offboard.set_velocity_ned(VelocityNedYaw(vn,ve,vd,yaw)),1.)
                    rows.append(dict(monotonic=now,index=index,target_east=east,target_north=north,position=p.copy(),error_m=distance,lidar_nearest_m=nearest,command_ned=[vn,ve,vd]))
                    await asyncio.sleep(.05)
                else:raise TimeoutError('Waypoint timeout '+str(index))
                reached.append(index);print('waypoint_reached',index,flush=True)
        except BaseException as exc:
            error=f'{type(exc).__name__}: {exc}';raise
        finally:
            try:
                if active:
                    await asyncio.wait_for(drone.offboard.set_velocity_ned(VelocityNedYaw(0.,0.,0.,yaw)),1)
                    await asyncio.wait_for(drone.offboard.stop(),3)
            finally:
                write_record(OUT/'route.json',dict(status='failed' if error else 'route_completed',error=error,reached=reached,samples=rows,training_admitted=False,promotable=False))
    os.environ.update(PX4_GZ_MAG_ENU_GAUSS='1',PX4_PARAM_EKF2_DECL_TYPE='3',PX4_PARAM_EKF2_MAG_DECL='0')
    prearm.OUT=airborne.OUT=OUT;trial.OUT=OUT/'runtime'
    try:
        await trial.main(fly=True,preflight_hook=preflight,takeoff_altitude=2.,post_hover_hook=route,server_config=config,post_hover_timeout_s=45.)
    finally:await source.stop()


if __name__=='__main__':asyncio.run(main())
