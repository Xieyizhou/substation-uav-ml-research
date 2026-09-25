"""YOLO RGB-D equipment-triggered stand-off route replacement, PX4 SITL only."""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import time
from mavsdk.offboard import VelocityNedYaw, PositionNedYaw
from src.flight.turn_hold import slew_heading
from src.flight.visual_replan_service import VisualReplanService
from src.flight.visual_target_route import plan_visual_standoff
from src.flight.live_obstacle_map import GazeboScanGate
from src.flight.envelope_route import EnvelopeMap as ClearanceMap
from src.flight.tracking_envelope import check_route
from src.flight.replan_evidence import EvidenceJournal
from src.flight.arrival_capture import ArrivalCapture
from src.sensors.gazebo_lidar_observed import ObservedLidarSource as GazeboLidar2DSource
from src.planner.local_frame import LocalFrame
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

from src.flight import sitl_lifecycle
from src.flight.sitl_support import alignment, cruise_guard, angle_error
from src.sandbox.live_replan_stop import check_stop
async def run(root, px4, assets, out, model_run):
    OUT = Path(out)
    protocol=json.loads((OUT/'protocol.json').read_text())
    check_stop(OUT)
    for p,h in protocol['inputs'].items():
        if file_sha256(p)!=h:raise ValueError('Frozen input changed: '+p)
    vision=VisualReplanService(model_run,OUT/'vision')
    source=GazeboLidar2DSource(stale_after_s=.5)
    scan_gate=GazeboScanGate()
    nearest=scan_gate.nearest
    async def gate(drone,state,origin):
        await alignment(OUT)
        await source.start();await source.wait_ready(10)
        if nearest(source)<=1.5:raise RuntimeError('Unsafe initial scan')

    async def follow(drone,state,origin):
        await alignment(OUT, airborne=True)
        frame=LocalFrame.from_mapping(json.loads((OUT/'inair-alignment.json').read_text())['local_frame'])
        mapping=ClearanceMap(protocol['static_boxes'],frame)
        evidence=EvidenceJournal(OUT,frame)
        arrival=ArrivalCapture()
        vision.begin_mission()
        p=state['local'][1];route=[frame.to_map(p['east'],p['north'])]+[tuple(v) for v in protocol['initial_route'][1:]]
        index=0;rows=[];events=[];plans=[];error=None;active=False;target_pending=None;stage='following';stopped_since=None;aligned_since=None;arrived_since=None;last_yaw=None;last_time=None;yaw=state['attitude'][1]['yaw']
        turn_anchor=(p['north'],p['east']);command_yaw=yaw;command_time=time.monotonic()
        def event(name,**kwargs):
            events.append(dict(event=name,monotonic=time.monotonic(),**kwargs));print(name,kwargs,flush=True)
        try:
            await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,yaw));await asyncio.wait_for(drone.offboard.start(),3);active=True
            while index<len(route)-1:
                check_stop(OUT)
                now=time.monotonic();a,b=route[index:index+2]
                point,cross=cruise_guard(state,origin,frame,a,b,now)
                distance=nearest(source)
                if distance<=1.5:raise RuntimeError('LiDAR emergency stop')
                p=state['local'][1].copy();att=state['attitude'][1].copy();speed=math.hypot(p['vn'],p['ve'])
                candidate=vision.current(now)
                yaw_rate=0. if last_yaw is None else abs(angle_error(att['yaw'],last_yaw))/max(now-last_time,.001)
                last_yaw=att['yaw'];last_time=now
                scan=source.latest();added=0
                # Receive-time pairing is bounded. Suppress map updates during
                # fast turns; raw LiDAR emergency stopping remains active.
                eligible=yaw_rate<5 and max(abs(att['roll']),abs(att['pitch']))<=1 and max(abs(scan.received_monotonic_s-state[k][0]) for k in ('local','attitude'))<=.033334
                if eligible:
                    added=mapping.ingest(scan,p,att,now)
                evidence.append(scan,state,now,yaw_rate,eligible,added)
                remaining=[point]+route[index+1:]
                if mapping.route_blocked(remaining):raise RuntimeError('LiDAR route became blocked; abort semantic trial')
                if stage=='following' and not plans and speed>=.1 and candidate is not None:
                    target_pending=candidate
                    stage='braking';stopped_since=None
                    await drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,command_yaw))
                    snapshot=evidence.snapshot('route_blocked',mapping,state,scan,route,index)
                    event('visual_target_stop_requested',speed_m_s=speed,lidar_m=distance,request=candidate,map_snapshot=snapshot)
                vn=ve=0.;capture=False
                if stage=='braking':
                    if speed<.08:
                        stopped_since=stopped_since or now
                        if now-stopped_since>=1:
                            if len(plans)>=1:
                                evidence.snapshot('replan_budget_exhausted',mapping,state,scan,route,index)
                                raise RuntimeError('Replan budget exhausted')
                            event('actual_stop_confirmed',speed_m_s=speed)
                            write_record(OUT/f'map-before-replan-{len(plans)+1}.json',dict(boxes=mapping.boxes(),points=mapping.points,source='live_lidar_only'))
                            candidate=vision.current(time.monotonic())
                            if candidate is None or candidate['class_name']!=target_pending['class_name']:raise RuntimeError('Visual target lost while stopped')
                            snapshot=await vision.snapshot(candidate)
                            planned=await asyncio.wait_for(asyncio.to_thread(plan_visual_standoff,candidate,now=time.monotonic(),frame=frame,start=point,boxes=protocol['static_boxes']+mapping.boxes()),3.)
                            corroborated=vision.current(time.monotonic())
                            if corroborated is None or corroborated['class_name']!=candidate['class_name'] or math.dist(tuple(corroborated['local_position'].values()),tuple(candidate['local_position'].values()))>.3:raise RuntimeError('Visual target changed during planning')
                            route=planned['route']
                            plans.append(dict(points=route,visual_plan=planned,request_snapshot=snapshot,corroborating_request=corroborated,observed_boxes=mapping.boxes(),monotonic=time.monotonic()))
                            index=0;stage='following';aligned_since=None;arrived_since=None
                            turn_anchor=(p['north'],p['east'])
                            arrival.reset()
                            event('route_replaced',replan=len(plans),waypoints=route)
                    else:stopped_since=None
                else:
                    yaw=math.degrees(math.atan2(b[0]-a[0],b[1]-a[1]))
                    target_e,target_n=frame.to_local(*b);ne,ee=target_n-p['north'],target_e-p['east'];err=math.hypot(ne,ee)
                    capture=arrival.update(err)
                    if abs(angle_error(att['yaw'],yaw))<=5:aligned_since=aligned_since or now
                    else:aligned_since=None
                    if err<.03 and speed<.08:
                        arrived_since=arrived_since or now
                        if now-arrived_since>=.5:
                            index+=1;arrived_since=None;aligned_since=None;event('waypoint_reached',index=index)
                            turn_anchor=(p['north'],p['east'])
                            arrival.reset();capture=False
                    else:
                        arrived_since=None
                        if not capture and err>=.03 and aligned_since is not None and now-aligned_since>=.5:
                            vn,ve=.7*ne,.7*ee;scale=max(1.,math.hypot(vn,ve)/.2);vn/=scale;ve/=scale
                            if abs(angle_error(math.degrees(math.atan2(ve,vn)),att['yaw']))>90:raise RuntimeError('Motion outside forward scan sector')
                vd=max(-.15,min(.15,.7*(origin['down']-2-p['down'])))
                command_yaw=slew_heading(command_yaw,yaw,now-command_time);command_time=now
                hold=stage=='following' and (aligned_since is None or now-aligned_since<.5)
                if capture:
                    await asyncio.wait_for(drone.offboard.set_position_ned(PositionNedYaw(target_n,target_e,origin['down']-2,command_yaw)),1)
                elif hold:
                    await asyncio.wait_for(drone.offboard.set_position_ned(PositionNedYaw(*turn_anchor,origin['down']-2,command_yaw)),1)
                else:
                    await asyncio.wait_for(drone.offboard.set_velocity_ned(VelocityNedYaw(vn,ve,vd,command_yaw)),1)
                rows.append(dict(monotonic=now,stage=stage,index=index,map_position=point,position=p,attitude=att,scan_sequence=scan.sequence,scan_timestamp_s=scan.timestamp_s,new_returns=added,cross_track_m=cross,lidar_m=distance,command_ned=[vn,ve,vd],control_mode='arrival_capture' if capture else 'position_hold' if hold else 'velocity',capture_target=[target_n,target_e] if capture else None,turn_anchor=turn_anchor,yaw_command=command_yaw,yaw_target=yaw))
                await asyncio.sleep(.05)
            if not plans:raise RuntimeError('No actual visual replan occurred')
            event('visual_standoff_reached_after_replan')
        except BaseException as exc:
            error=f'{type(exc).__name__}: {exc}'
            write_record(OUT/'sensor-failure.json',dict(error=error,monotonic=time.monotonic(),diagnostics=source.diagnostics()))
            try:evidence.snapshot(error,mapping,state,source.latest(),route,index)
            except Exception as diagnostic_error:event('snapshot_write_failed',error=str(diagnostic_error))
            raise
        finally:
            try:
                if active:
                    await asyncio.wait_for(drone.offboard.set_velocity_ned(VelocityNedYaw(0,0,0,yaw)),1)
                    await asyncio.wait_for(drone.offboard.stop(),3)
            finally:
                evidence.close()
                write_record(OUT/'sensor-diagnostics.json',source.diagnostics())
                write_record(OUT/'route.json',dict(status='failed' if error else 'visual_standoff_route_completed',error=error,events=events,replans=plans,samples=rows,training_admitted=False,promotable=False))
    os.environ.update(PX4_GZ_MAG_ENU_GAUSS='1',PX4_PARAM_EKF2_DECL_TYPE='3',PX4_PARAM_EKF2_MAG_DECL='0')
    try:await sitl_lifecycle.run(root, px4, assets, OUT/'runtime', fly=True,preflight_hook=gate,takeoff_altitude=2.,post_hover_hook=follow,server_config=assets/'server.config',post_hover_timeout_s=180.,source_world=assets/'world.sdf',visual_service=vision)
    finally:await source.stop()
