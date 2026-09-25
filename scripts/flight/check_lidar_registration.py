"""Unarmed SITL check of planar scan mounting against a known wall."""
import asyncio
import json
import math
import os
from pathlib import Path
import statistics
import time
from scripts.flight import fly_obstacle_stop as wall
from scripts.flight import fly_px4_shadow_hover as trial
from scripts.flight import retest_sim_heading as prearm
from scripts.flight.retest_magnetic_contract import validate_contract
from src.perception.lidar_detector import LidarRiskDetector
from src.sensors.gazebo_lidar import GazeboLidar2DSource
from src.planner.local_frame import LocalFrame
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

OUT=trial.ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1/lidar-registration-001'

async def main():
    config=OUT.parent/'magnetic-contract-001/server.config';validate_contract(config)
    OUT.mkdir(exist_ok=False)
    write_record(OUT/'protocol.json',dict(simulation_only=True,arming_authorized=False,frames=30,wall_map_east_m=3.2,max_endpoint_error_m=.05,max_receive_time_pair_difference_s=.033334,static_speed_limit_m_s=.02,tilt_limit_deg=1.,dynamic_registration_certified=False,training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in (Path(__file__).resolve(),Path(trial.__file__).resolve(),Path(wall.__file__).resolve(),config,trial.BUILD/'bin/px4',trial.ROOT/'src/perception/lidar_detector.py')}))
    source=GazeboLidar2DSource(stale_after_s=.5)
    async def check(drone,state,origin):
        await prearm.gate(drone,state,origin)
        await wall.fixture()
        await source.start();await source.wait_ready(10);await asyncio.sleep(1)
        frame=LocalFrame.from_mapping(json.loads((OUT/'prearm-alignment.json').read_text())['local_frame'])
        corrected=LidarRiskDetector(source,resolution_m=.1,local_frame=frame,sensor_forward_m=-.1)
        legacy=LidarRiskDetector(source,resolution_m=.1,local_frame=frame)
        records=[];seen=set();deadline=time.monotonic()+15
        while len(records)<30 and time.monotonic()<deadline:
            scan=source.latest();now=time.monotonic()
            p=state['local'][1];a=state['attitude'][1]
            if state['armed'][1] is not False:raise RuntimeError('Unexpected arm state')
            if scan is None or scan.sequence in seen or scan.age_s(now)>.2:
                await asyncio.sleep(.01);continue
            if max(abs(scan.received_monotonic_s-state[k][0]) for k in ('local','attitude'))>.033334:
                await asyncio.sleep(.01);continue
            if math.hypot(p['vn'],p['ve'])>.02 or max(abs(a['roll']),abs(a['pitch']))>1:raise RuntimeError('Static planar assumption failed')
            endpoints=[]
            for i,d in enumerate(scan.ranges_m):
                angle=scan.angle_at(i)
                if abs(angle)>.2 or not math.isfinite(d) or not scan.range_min_m<=d<=scan.range_max_m:continue
                _,_,n,e=corrected._global_cell(d,angle,p['north'],p['east'],a['yaw'])
                _,_,ln,le=legacy._global_cell(d,angle,p['north'],p['east'],a['yaw'])
                me,mn=frame.to_map(e,n);old_e,_=frame.to_map(le,ln)
                endpoints.append(dict(index=i,range_m=d,angle_rad=angle,map_east=me,map_north=mn,error_m=me-3.2,uncorrected_error_m=old_e-3.2))
            if len(endpoints)<50:raise RuntimeError('Insufficient central wall returns')
            records.append(dict(sequence=scan.sequence,scan_timestamp_s=scan.timestamp_s,scan_received=scan.received_monotonic_s,local_received=state['local'][0],attitude_received=state['attitude'][0],position=p.copy(),attitude=a.copy(),endpoints=endpoints));seen.add(scan.sequence)
            await asyncio.sleep(.01)
        write_record(OUT/'observations.json',dict(frames=records))
        errors=[abs(e['error_m']) for r in records for e in r['endpoints']]
        old=[abs(e['uncorrected_error_m']) for r in records for e in r['endpoints']]
        passed=len(records)==30 and max(errors,default=math.inf)<=.05
        write_record(OUT/'registration.json',dict(passed=passed,frames=len(records),max_error_m=max(errors,default=None),median_error_m=statistics.median(errors) if errors else None,uncorrected_median_error_m=statistics.median(old) if old else None,role='static receive-time paired check; not dynamic timestamp certification',training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in (OUT/'observations.json',OUT/'protocol.json',OUT/'prearm-alignment.json',OUT/'fixture.sdf')}))
        if not passed:raise RuntimeError('Static LiDAR registration failed')
        print('static_lidar_registration_passed',max(errors),flush=True)
    prearm.OUT=wall.OUT=OUT;trial.OUT=OUT/'runtime'
    os.environ.update(PX4_GZ_MAG_ENU_GAUSS='1',PX4_PARAM_EKF2_DECL_TYPE='3',PX4_PARAM_EKF2_MAG_DECL='0')
    try:await trial.main(fly=False,preflight_hook=check,server_config=config)
    finally:await source.stop()

if __name__=='__main__':asyncio.run(main())
