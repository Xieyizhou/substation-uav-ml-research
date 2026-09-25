"""Postflight controller/groundtruth audit, never starts flight."""
import argparse
import bisect
import json
import math
from pathlib import Path
from scripts.flight.fly_pillar_route import OUT
from scripts.flight.diagnose_gps_health import read_samples
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from src.planner.local_frame import LocalFrame

def point_box_distance(point,box):
    return math.sqrt(sum(max(lo-v,0.,v-hi)**2 for v,lo,hi in zip(point,(box['x0'],box['y0'],box['z0']),(box['x1'],box['y1'],box['z1']))))

def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for p,h in protocol['inputs'].items():
        if file_sha256(p)!=h:raise ValueError('Changed protocol input: '+p)
    runtime=json.loads((OUT/'runtime/receipt.json').read_text())
    route=json.loads((OUT/'route.json').read_text())
    vision_path=OUT/'runtime/vision/completion.json'
    vision=json.loads(vision_path.read_text())
    for path,digest in vision['inputs'].items():
        if file_sha256(path)!=digest:raise ValueError('Changed vision evidence: '+path)
    predictions=[json.loads(line) for line in (OUT/'runtime/vision/detections.jsonl').read_text().splitlines()]
    events={r['event']:r['monotonic'] for r in runtime['events']}
    covered=bool(predictions) and predictions[0]['result_monotonic']<=events['arm_requested'] and predictions[-1]['result_monotonic']>=events['disarmed_confirmed']
    log=next((OUT/'runtime/px4-work/log').rglob('*.ulg'))
    all_positions=list(read_samples(log,{'vehicle_local_position_groundtruth','vehicle_local_position'}))
    truth=[v for n,m,v in all_positions if n=='vehicle_local_position_groundtruth' and m==0]
    if not truth:raise ValueError('Missing groundtruth')
    boxes=protocol['collision_boxes']+[protocol['pillar']]
    sphere=json.loads((OUT.parent/'vehicle-envelope-002/envelope.json').read_text())['collision_sphere_radius_m']
    clearance=min(point_box_distance((v['y']+10,v['x']+10,-v['z']),b)-sphere for v in truth for b in boxes)
    # Ordered closest visits are descriptive independent diagnostics, not a
    # post-hoc replacement tolerance for controller acceptance.
    visits=[];start=0
    for target in protocol['map_points'][1:]:
        index=min(range(start,len(truth)),key=lambda i:math.dist((truth[i]['y']+10,truth[i]['x']+10),target))
        v=truth[index];visits.append(dict(target=target,timestamp_us=v['timestamp'],distance_m=math.dist((v['y']+10,v['x']+10),target)));start=index
    rows=route['samples']
    calibration=json.loads((OUT/'inair-alignment.json').read_text())
    frame=LocalFrame.from_mapping(calibration['local_frame'])
    stamps=[v['timestamp'] for v in truth];alignment=[]
    for name,m,v in all_positions:
        if name!='vehicle_local_position' or m!=0:continue
        i=bisect.bisect_left(stamps,v['timestamp'])
        choices=[truth[j] for j in (i-1,i) if 0<=j<len(truth)]
        t=min(choices,key=lambda x:abs(x['timestamp']-v['timestamp']))
        skew=abs(t['timestamp']-v['timestamp'])
        if skew>33334 or -t['z']<1.6:continue
        alignment.append(dict(skew_us=skew,horizontal_error_m=math.dist(frame.to_map(v['y'],v['x']),(t['y']+10,t['x']+10)),heading_error_deg=abs(math.degrees((v['heading']-t['heading']+math.pi)%(2*math.pi)-math.pi))))
    moving=[r for r in rows if math.hypot(*r['command_ned'][:2])>1e-6]
    metrics=dict(reached=route['reached'],ordered_truth_closest_visits=visits,whole_run_min_collision_sphere_clearance_m=clearance,max_measured_speed_m_s=max((math.hypot(r['position']['vn'],r['position']['ve']) for r in rows),default=None),min_lidar_m=min((r['lidar_nearest_m'] for r in rows),default=None),max_cross_track_m=max((r['cross_track_m'] for r in rows),default=None),max_yaw_error_when_commanded_moving_deg=max((r['yaw_error_deg'] for r in moving),default=None),landing_confirmed=runtime['landing_confirmed'],final_armed=runtime['final_armed'],owned_processes_exited=runtime['owned_processes_exited'])
    metrics['airborne_alignment_diagnostic']=dict(pairs=len(alignment),max_position_error_m=max((v['horizontal_error_m'] for v in alignment),default=None),max_heading_error_deg=max((v['heading_error_deg'] for v in alignment),default=None),max_skew_us=max((v['skew_us'] for v in alignment),default=None),role='postflight diagnostic above 1.6 m true model-origin height, including turns; not the static prearm gate')
    metrics['vision']=dict(inferred_frames=len(predictions),covers_arm_through_disarm=covered,maximum_result_gap_s=max((b['result_monotonic']-a['result_monotonic'] for a,b in zip(predictions,predictions[1:])),default=None),control_authority='none')
    passed=route['status']=='route_completed' and route['reached']==list(range(len(protocol['map_points'])-1)) and runtime['status']=='sitl_hover_landed_disarmed' and metrics['landing_confirmed'] and metrics['final_armed'] is False and metrics['owned_processes_exited'] and clearance>0
    passed=passed and covered and vision['status']=='live_complete_not_flight_certified'
    paths=[OUT/'protocol.json',OUT/'route.json',OUT/'prearm-alignment.json',OUT/'inair-alignment.json',OUT/'pillar.sdf',OUT/'pillar-receipt.json',OUT/'pillar-scan.json',OUT/'runtime/receipt.json',OUT/'runtime/telemetry.jsonl',log,Path(__file__).resolve()]
    paths.append(vision_path)
    write_record(OUT/'completion.json',dict(status='bounded_static_pillar_route_verified' if passed else 'needs_investigation',metrics=metrics,simulation_only=True,dynamic_replanning_verified=False,physical_flight_certified=False,training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths}))
    print(json.dumps(dict(passed=passed,metrics=metrics),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,choices=(1,2,3),default=1);args=p.parse_args();OUT=OUT.parent/f'pillar-route-{args.attempt:03}';main()
