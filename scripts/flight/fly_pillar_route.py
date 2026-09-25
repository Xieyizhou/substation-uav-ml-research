"""Explicit, bounded SITL pillar avoidance; no height-based obstacle removal."""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import time
from mavsdk.offboard import VelocityNedYaw
from scripts.flight import route_trial_lifecycle as trial
from scripts.flight import retest_sim_heading as prearm
from scripts.flight import check_final_heading as airborne
from scripts.flight.fly_obstacle_stop import nearest
from scripts.flight.prepare_avoidance_route import collision_boxes,build,inside
from scripts.flight.retest_magnetic_contract import validate_contract
from src.planner.local_frame import LocalFrame
from src.perception.lidar_detector import LidarRiskDetector
from src.sensors.gazebo_lidar import GazeboLidar2DSource
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

OUT=trial.ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1/pillar-route-001'
PILLAR=dict(name='diagnostic_pillar',x0=1.15,x1=1.45,y0=3.65,y1=3.95,z0=0.,z1=4.)

def segment_distance(point,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1];length2=dx*dx+dy*dy
    if length2==0:return math.dist(point,a)
    t=max(0.,min(1.,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/length2))
    return math.dist(point,(a[0]+t*dx,a[1]+t*dy))

def angle_error(a,b):return (a-b+180)%360-180

def cruise_guard(state,origin,frame,a,b,now):
    height,_=trial.guard(state,now,origin,max_height_m=2.8,max_drift_m=6.)
    if not 1.75<=height<=2.25:raise RuntimeError('Cruise altitude excursion')
    p=state['local'][1];att=state['attitude'][1];point=frame.to_map(p['east'],p['north'])
    if max(abs(att['roll']),abs(att['pitch']))>5:raise RuntimeError('Cruise tilt excursion')
    distance=segment_distance(point,a,b)
    if distance>.18:raise RuntimeError('Route corridor excursion')
    if math.hypot(p['vn'],p['ve'])>.35:raise RuntimeError('Actual speed excursion')
    return point,distance

def freeze():
    world=trial.ROOT/'simulation/worlds/substation_simple.sdf'
    boxes=collision_boxes(world);points=build(boxes+[PILLAR],goal=(1.3,6.))
    # Prove direct travel is physically blocked (not just an inflated near miss).
    if not any(inside(tuple(a+(b-a)*i/1000 for a,b in zip(points[0],points[-1])),PILLAR) for i in range(1001)):
        raise ValueError('Fixture does not block the direct line')
    OUT.mkdir(exist_ok=False)
    config=OUT.parent/'magnetic-contract-001/server.config';validate_contract(config)
    sources=[world,trial.ROOT/'simulation/models/x500_research/model.sdf',config,trial.BUILD/'bin/px4',Path(__file__).resolve(),Path(trial.__file__).resolve(),trial.ROOT/'scripts/vision/material_shadow_route.py',trial.ROOT/'scripts/flight/prepare_avoidance_route.py',trial.ROOT/'src/planner/astar_grid.py',OUT.parent/'lidar-registration-001/completion.json',OUT.parent/'vehicle-envelope-002/envelope.json']
    for name in ('lidar-registration-001/completion.json','vehicle-envelope-002/envelope.json'):
        v=json.loads((OUT.parent/name).read_text())
        for p,d in v['inputs'].items():
            if file_sha256(p)!=d:raise ValueError('Prerequisite changed: '+p)
    write_record(OUT/'protocol.json',dict(simulation_only=True,map_points=points,pillar=PILLAR,collision_boxes=boxes,height_filter_enabled=False,planning_clearance_m=1.8,lidar_stop_m=1.5,cruise_relative_altitude_m=[1.75,2.25],nominal_altitude_m=2.,command_speed_m_s=.2,actual_speed_limit_m_s=.35,corridor_m=.18,yaw_gate_deg=5.,waypoint_tolerance_m=.1,stable_s=.5,timeout_s=180.,maximum_displacement_m=6.,vision_control_authority='none',training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in sources}))
    print('frozen_route',points,flush=True)

async def create_pillar():
    sdf='<sdf version="1.9"><model name="diagnostic_pillar"><static>true</static><pose>-8.7 -6.2 2 0 0 0</pose><link name="pillar"><collision name="collision"><geometry><box><size>0.3 0.3 4</size></box></geometry></collision><visual name="visual"><geometry><box><size>0.3 0.3 4</size></box></geometry></visual></link></model></sdf>'
    (OUT/'pillar.sdf').write_text(sdf)
    p=await asyncio.create_subprocess_exec('/opt/homebrew/bin/gz','service','-s','/world/substation_simple/create','--reqtype','gz.msgs.EntityFactory','--reptype','gz.msgs.Boolean','--timeout','3000','--req','sdf: '+json.dumps(sdf),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    try:stdout,stderr=await asyncio.wait_for(p.communicate(),5)
    finally:
        if p.returncode is None:p.kill();await p.wait()
    write_record(OUT/'pillar-receipt.json',dict(returncode=p.returncode,stdout=stdout.decode(),stderr=stderr.decode(),sha256=file_sha256(OUT/'pillar.sdf')))
    if p.returncode or 'data: true' not in stdout.decode():raise RuntimeError('Pillar creation failed')

async def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for p,d in protocol['inputs'].items():
        if file_sha256(p)!=d:raise ValueError('Frozen input changed: '+p)
    source=GazeboLidar2DSource(stale_after_s=.5)
    async def gate(drone,state,origin):
        await prearm.gate(drone,state,origin);await create_pillar()
        await source.start();await source.wait_ready(10);await asyncio.sleep(1)
        if nearest(source)<=1.5:raise RuntimeError('Unsafe prearm scan')
        frame=LocalFrame.from_mapping(json.loads((OUT/'prearm-alignment.json').read_text())['local_frame'])
        detector=LidarRiskDetector(source,resolution_m=.1,local_frame=frame,sensor_forward_m=-.1)
        scan=source.latest();p=state['local'][1];att=state['attitude'][1];hits=[]
        for i,d in enumerate(scan.ranges_m):
            if not math.isfinite(d) or not scan.range_min_m<=d<=scan.range_max_m:continue
            _,_,n,e=detector._global_cell(d,scan.angle_at(i),p['north'],p['east'],att['yaw'])
            point=frame.to_map(e,n)
            if inside(point,PILLAR,.05):hits.append(point)
        write_record(OUT/'pillar-scan.json',dict(hits=hits,scan_timestamp_s=scan.timestamp_s))
        if len(hits)<5:raise RuntimeError('Pillar scan registration not confirmed')

    async def follow(drone,state,origin):
        await airborne.post_hover(drone,state,origin)
        frame=LocalFrame.from_mapping(json.loads((OUT/'inair-alignment.json').read_text())['local_frame'])
        rows=[];reached=[];active=False;error=None;yaw=state['attitude'][1]['yaw']
        try:
            await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,yaw))
            await asyncio.wait_for(drone.offboard.start(),3);active=True
            points=protocol['map_points']
            for index,(a,b) in enumerate(zip(points,points[1:])):
                yaw=math.degrees(math.atan2(b[0]-a[0],b[1]-a[1]))
                target_e,target_n=frame.to_local(*b);deadline=time.monotonic()+math.dist(a,b)/.2*2+20
                aligned_since=None;arrival_since=None
                while time.monotonic()<deadline:
                    now=time.monotonic();point,cross=cruise_guard(state,origin,frame,a,b,now)
                    distance=nearest(source)
                    if distance<=1.5:raise RuntimeError('LiDAR emergency stop')
                    p=state['local'][1].copy();att=state['attitude'][1]
                    ne,ee=target_n-p['north'],target_e-p['east'];err=math.hypot(ne,ee)
                    heading=abs(angle_error(att['yaw'],yaw))
                    if heading<=5:aligned_since=aligned_since or now
                    else:aligned_since=None
                    vn=ve=0.
                    if err<.1:
                        if math.hypot(p['vn'],p['ve'])<.08:
                            arrival_since=arrival_since or now
                            if now-arrival_since>=.5:break
                        else:arrival_since=None
                    elif aligned_since is not None and now-aligned_since>=.5:
                        vn,ve=.7*ne,.7*ee;scale=max(1.,math.hypot(vn,ve)/.2);vn/=scale;ve/=scale
                        if abs(angle_error(math.degrees(math.atan2(ve,vn)),att['yaw']))>90:raise RuntimeError('Command outside forward scan sector')
                    vd=max(-.15,min(.15,.7*(origin['down']-2-p['down'])))
                    await asyncio.wait_for(drone.offboard.set_velocity_ned(VelocityNedYaw(vn,ve,vd,yaw)),1)
                    rows.append(dict(monotonic=now,index=index,map_position=point,position=p,cross_track_m=cross,target_error_m=err,yaw=att['yaw'],yaw_target=yaw,yaw_error_deg=heading,lidar_nearest_m=distance,command_ned=[vn,ve,vd]))
                    await asyncio.sleep(.05)
                else:raise TimeoutError('Segment timeout '+str(index))
                reached.append(index);print('segment_reached',index,flush=True)
        except BaseException as exc:error=f'{type(exc).__name__}: {exc}';raise
        finally:
            try:
                if active:
                    await asyncio.wait_for(drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,yaw)),1)
                    await asyncio.wait_for(drone.offboard.stop(),3)
            finally:write_record(OUT/'route.json',dict(status='failed' if error else 'route_completed',error=error,reached=reached,samples=rows,training_admitted=False,promotable=False))
    os.environ.update(PX4_GZ_MAG_ENU_GAUSS='1',PX4_PARAM_EKF2_DECL_TYPE='3',PX4_PARAM_EKF2_MAG_DECL='0')
    prearm.trial=airborne.trial=trial;prearm.OUT=airborne.OUT=OUT;trial.OUT=OUT/'runtime'
    try:await trial.main(fly=True,preflight_hook=gate,takeoff_altitude=2.,post_hover_hook=follow,server_config=OUT.parent/'magnetic-contract-001/server.config',post_hover_timeout_s=180.)
    finally:await source.stop()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fly',action='store_true');parser.add_argument('--attempt',type=int,choices=(1,2,3),default=1);args=parser.parse_args()
    OUT=OUT.parent/f'pillar-route-{args.attempt:03}'
    if args.fly:asyncio.run(main())
    else:freeze()
