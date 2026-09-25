"""Sensor-triggered stop/map/replan/resume experiment, PX4 SITL only."""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import time
from mavsdk.offboard import VelocityNedYaw, PositionNedYaw
from src.flight.turn_hold import slew_heading
from scripts.flight import route_trial_lifecycle as trial
from scripts.flight import retest_sim_heading as prearm
from scripts.flight import check_final_heading as airborne
from scripts.flight import replan_repeat_fixture as fixture_harness
from scripts.flight.fly_pillar_route import cruise_guard,angle_error
from scripts.flight.prepare_avoidance_route import collision_boxes,inside
from src.flight.live_obstacle_map import LiveObstacleMap,GazeboScanGate
from src.sensors.gazebo_lidar import GazeboLidar2DSource
from src.planner.local_frame import LocalFrame
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

OUT=trial.ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1/live-replan-001'
GOAL=(1.3,6.5)
INITIAL_ROUTE=[(1.5,1.5),(1.2,.8),(1.2,6.5),GOAL]
CASE='baseline'

def freeze():
    world=trial.ROOT/'simulation/worlds/substation_simple.sdf';boxes=collision_boxes(world)
    for a,b in zip(INITIAL_ROUTE,INITIAL_ROUTE[1:]):
        for i in range(101):
            p=tuple(x+(y-x)*i/100 for x,y in zip(a,b))
            if any(inside(p,box,1.8) for box in boxes):raise ValueError('Initial static route blocked')
    prior=OUT.parent/'pillar-route-002/completion.json';r=json.loads(prior.read_text())
    if r['status']!='bounded_static_pillar_route_verified':raise ValueError('Static flight prerequisite missing')
    OUT.mkdir(exist_ok=False)
    sources=[Path(__file__).resolve(),Path(trial.__file__).resolve(),Path(fixture_harness.__file__).resolve(),trial.ROOT/'src/flight/live_obstacle_map.py',trial.ROOT/'scripts/flight/prepare_avoidance_route.py',trial.ROOT/'src/planner/astar_grid.py',trial.ROOT/'src/perception/lidar_detector.py',trial.ROOT/'scripts/vision/material_shadow_route.py',world,trial.BUILD/'bin/px4',OUT.parent/'magnetic-contract-001/server.config',prior]
    sources.extend([trial.ROOT/'src/flight/turn_hold.py',OUT.parent/'replan-repeat-v1/protocol.json'])
    write_record(OUT/'protocol.json',dict(simulation_only=True,turn_policy='fixed local position hold; heading setpoint slew 20 deg/s, dt capped 0.1s',goal=GOAL,initial_route=INITIAL_ROUTE,static_boxes=boxes,unregistered_fixture_in_planner=False,fixture_policy='harness inserts pillar only after measured northward speed >=0.15 m/s; controller receives no fixture pose',map_policy='live endpoints only, accumulate, no return-based clearing, 0.10 m observed-bound padding; unobserved shape not certified',waypoint_tolerance_m=.03,stop_speed_m_s=.08,stop_stable_s=1.,planning_clearance_m=1.8,lidar_emergency_stop_m=1.5,max_replans=3,route_timeout_s=180,vision_control_authority='none',training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in sources}))

async def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    campaign=json.loads((OUT.parent/'replan-repeat-v1/protocol.json').read_text())
    unit=next((u for u in campaign['units'] if u['directory']==OUT.name),None)
    if unit is None or unit['case']!=CASE:raise ValueError('Unregistered repeat-test unit')
    for p,h in protocol['inputs'].items():
        if file_sha256(p)!=h:raise ValueError('Frozen input changed: '+p)
    source=GazeboLidar2DSource(stale_after_s=.5)
    scan_gate=GazeboScanGate()
    nearest=scan_gate.nearest
    async def gate(drone,state,origin):
        await prearm.gate(drone,state,origin)
        await source.start();await source.wait_ready(10)
        if nearest(source)<=1.5:raise RuntimeError('Unsafe initial scan')

    async def follow(drone,state,origin):
        await airborne.post_hover(drone,state,origin)
        frame=LocalFrame.from_mapping(json.loads((OUT/'inair-alignment.json').read_text())['local_frame'])
        mapping=LiveObstacleMap(protocol['static_boxes'],frame)
        p=state['local'][1];route=[frame.to_map(p['east'],p['north'])]+[tuple(v) for v in protocol['initial_route'][1:]]
        index=0;rows=[];events=[];plans=[];error=None;active=False;spawn=None;stage='following';stopped_since=None;aligned_since=None;arrived_since=None;last_yaw=None;last_time=None;yaw=state['attitude'][1]['yaw']
        turn_anchor=(p['north'],p['east']);command_yaw=yaw;command_time=time.monotonic()
        def event(name,**kwargs):
            events.append(dict(event=name,monotonic=time.monotonic(),**kwargs));print(name,kwargs,flush=True)
        try:
            await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,yaw));await asyncio.wait_for(drone.offboard.start(),3);active=True
            while index<len(route)-1:
                now=time.monotonic();a,b=route[index:index+2]
                point,cross=cruise_guard(state,origin,frame,a,b,now)
                distance=nearest(source)
                if distance<=1.5:raise RuntimeError('LiDAR emergency stop')
                p=state['local'][1].copy();att=state['attitude'][1].copy();speed=math.hypot(p['vn'],p['ve'])
                if spawn is None and fixture_harness.ready(CASE,point,p):
                    event('harness_spawn_requested',measured_speed_m_s=speed)
                    spawn=asyncio.create_task(fixture_harness.insert(OUT,CASE))
                if spawn is not None and spawn.done():spawn.result()
                yaw_rate=0. if last_yaw is None else abs(angle_error(att['yaw'],last_yaw))/max(now-last_time,.001)
                last_yaw=att['yaw'];last_time=now
                scan=source.latest();added=0
                # Receive-time pairing is bounded. Suppress map updates during
                # fast turns; raw LiDAR emergency stopping remains active.
                if yaw_rate<5 and max(abs(att['roll']),abs(att['pitch']))<=1 and max(abs(scan.received_monotonic_s-state[k][0]) for k in ('local','attitude'))<=.033334:
                    added=mapping.ingest(scan,p,att,now)
                remaining=[point]+route[index+1:]
                if stage=='following' and mapping.route_blocked(remaining):
                    stage='braking';stopped_since=None;event('obstacle_detected_stop_requested',speed_m_s=speed,lidar_m=distance,observed_returns=len(mapping.points))
                vn=ve=0.
                if stage=='braking':
                    if speed<.08:
                        stopped_since=stopped_since or now
                        if now-stopped_since>=1:
                            if len(plans)>=3:raise RuntimeError('Replan budget exhausted')
                            event('actual_stop_confirmed',speed_m_s=speed)
                            write_record(OUT/f'map-before-replan-{len(plans)+1}.json',dict(boxes=mapping.boxes(),points=mapping.points,source='live_lidar_only'))
                            route=await asyncio.wait_for(asyncio.to_thread(mapping.replan,point,tuple(protocol['goal'])),3.)
                            plans.append(dict(points=route,observed_boxes=mapping.boxes(),monotonic=time.monotonic()))
                            index=0;stage='following';aligned_since=None;arrived_since=None
                            turn_anchor=(p['north'],p['east'])
                            event('route_replaced',replan=len(plans),waypoints=route)
                    else:stopped_since=None
                else:
                    yaw=math.degrees(math.atan2(b[0]-a[0],b[1]-a[1]))
                    target_e,target_n=frame.to_local(*b);ne,ee=target_n-p['north'],target_e-p['east'];err=math.hypot(ne,ee)
                    if abs(angle_error(att['yaw'],yaw))<=5:aligned_since=aligned_since or now
                    else:aligned_since=None
                    if err<.03 and speed<.08:
                        arrived_since=arrived_since or now
                        if now-arrived_since>=.5:
                            index+=1;arrived_since=None;aligned_since=None;event('waypoint_reached',index=index)
                            turn_anchor=(p['north'],p['east'])
                    else:
                        arrived_since=None
                        if err>=.03 and aligned_since is not None and now-aligned_since>=.5:
                            vn,ve=.7*ne,.7*ee;scale=max(1.,math.hypot(vn,ve)/.2);vn/=scale;ve/=scale
                            if abs(angle_error(math.degrees(math.atan2(ve,vn)),att['yaw']))>90:raise RuntimeError('Motion outside forward scan sector')
                vd=max(-.15,min(.15,.7*(origin['down']-2-p['down'])))
                command_yaw=slew_heading(command_yaw,yaw,now-command_time);command_time=now
                hold=stage=='following' and (aligned_since is None or now-aligned_since<.5)
                if hold:
                    await asyncio.wait_for(drone.offboard.set_position_ned(PositionNedYaw(*turn_anchor,origin['down']-2,command_yaw)),1)
                else:
                    await asyncio.wait_for(drone.offboard.set_velocity_ned(VelocityNedYaw(vn,ve,vd,command_yaw)),1)
                rows.append(dict(monotonic=now,stage=stage,index=index,map_position=point,position=p,attitude=att,scan_sequence=scan.sequence,scan_timestamp_s=scan.timestamp_s,new_returns=added,cross_track_m=cross,lidar_m=distance,command_ned=[vn,ve,vd],control_mode='position_hold' if hold else 'velocity',turn_anchor=turn_anchor,yaw_command=command_yaw,yaw_target=yaw))
                await asyncio.sleep(.05)
            if spawn is None or not plans:raise RuntimeError('No actual injected-obstacle replan occurred')
            await asyncio.wait_for(spawn,5);event('goal_reached_after_replan')
        except BaseException as exc:error=f'{type(exc).__name__}: {exc}';raise
        finally:
            try:
                if active:
                    await asyncio.wait_for(drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,yaw)),1)
                    await asyncio.wait_for(drone.offboard.stop(),3)
            finally:
                if spawn is not None:
                    if not spawn.done():spawn.cancel()
                    await asyncio.gather(spawn,return_exceptions=True)
                write_record(OUT/'route.json',dict(status='failed' if error else 'sensor_replan_route_completed',error=error,events=events,replans=plans,samples=rows,training_admitted=False,promotable=False))
    prearm.trial=airborne.trial=trial;prearm.OUT=airborne.OUT=OUT;trial.OUT=OUT/'runtime'
    os.environ.update(PX4_GZ_MAG_ENU_GAUSS='1',PX4_PARAM_EKF2_DECL_TYPE='3',PX4_PARAM_EKF2_MAG_DECL='0')
    try:await trial.main(fly=True,preflight_hook=gate,takeoff_altitude=2.,post_hover_hook=follow,server_config=OUT.parent/'magnetic-contract-001/server.config',post_hover_timeout_s=180.)
    finally:await source.stop()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fly',action='store_true');p.add_argument('--case',choices=tuple(fixture_harness.CASES),required=True);p.add_argument('--repeat',type=int,choices=(1,2),required=True);args=p.parse_args();CASE=args.case;OUT=OUT.parent/f'replan-repeat-v1-{CASE}-{args.repeat}'
    if args.fly:asyncio.run(main())
    else:freeze()
