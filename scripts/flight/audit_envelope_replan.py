"""Verify sensor-driven event chain and actual resumed flight, not just plans."""
import argparse
import json
import math
from pathlib import Path
from scripts.flight.fly_px4_shadow_hover import ROOT
from scripts.flight.prepare_avoidance_route import collision_boxes
from scripts.flight.summarize_pillar_route import point_box_distance
from scripts.flight.diagnose_gps_health import read_samples
from src.flight.envelope_route import EnvelopeMap as LiveObstacleMap
from src.flight.tracking_envelope import check_route
from scripts.flight.measure_mapping_error import measure
from src.planner.local_frame import LocalFrame
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();out=args.directory.resolve()
    protocol=json.loads((out/'protocol.json').read_text());r=json.loads((out/'route.json').read_text());runtime=json.loads((out/'runtime/receipt.json').read_text())
    for file,digest in protocol['inputs'].items():
        if file_sha256(file)!=digest:raise ValueError('Frozen input changed: '+file)
    required=['harness_spawn_requested','obstacle_detected_stop_requested','actual_stop_confirmed','route_replaced','goal_reached_after_replan']
    events=r['events'];indices=[];cursor=0
    for name in required:
        matches=[i for i,e in enumerate(events) if i>=cursor and e['event']==name]
        if not matches:raise ValueError('Missing ordered event: '+name)
        indices.append(matches[0]);cursor=matches[0]+1
    if events[indices[0]]['measured_speed_m_s']<.15:raise ValueError('No actual motion before injection')
    plan_replays=[];map_files=[]
    if not check_route(protocol['initial_route'],protocol['static_boxes'],map_uncertainty=.20)['passed']:raise ValueError('Initial tracking envelope invalid')
    for i,plan in enumerate(r['replans'],1):
        mp=out/f'map-before-replan-{i}.json';map_files.append(mp);evidence=json.loads(mp.read_text())
        if evidence['source']!='live_lidar_only':raise ValueError('Non-sensor map input')
        m=LiveObstacleMap(protocol['static_boxes'],LocalFrame());m.points=evidence['points']
        replay=m.replan(plan['points'][0],protocol['goal'])
        if [list(v) for v in replay]!=plan['points']:raise ValueError('Replan cannot be reproduced from sensor points')
        if not check_route(plan['points'],protocol['static_boxes']+evidence['boxes'],map_uncertainty=.20)['passed']:raise ValueError('Planned tracking envelope invalid')
        plan_replays.append(True)
    replanned_at=events[indices[3]]['monotonic'];samples=r['samples']
    if not 1<=len(r['replans'])<=protocol['max_replans']:raise ValueError('Replan count outside frozen budget')
    stops=[e for e in events if e['event']=='actual_stop_confirmed']
    if len(stops)!=len(r['replans']) or any(e['speed_m_s']>=protocol['stop_speed_m_s'] for e in stops):raise ValueError('Stop evidence inconsistent')
    if any(s['cross_track_m']>.18 or s['lidar_m']<=protocol['lidar_emergency_stop_m'] or math.hypot(s['position']['vn'],s['position']['ve'])>.35 for s in samples):raise ValueError('Recorded flight envelope violated')
    final_map_gap=math.dist(samples[-1]['map_position'],protocol['goal'])
    if final_map_gap>=protocol['waypoint_tolerance_m']:raise ValueError('Final waypoint not reached')
    resumed=[s for s in samples if s['monotonic']>replanned_at and math.hypot(s['position']['vn'],s['position']['ve'])>.1 and math.hypot(*s['command_ned'][:2])>.1]
    if not resumed:raise ValueError('No actual resumed flight')
    log=next((out/'runtime/px4-work/log').rglob('*.ulg'));truth=[v for n,m,v in read_samples(log,{'vehicle_local_position_groundtruth'}) if m==0]
    geometry=protocol['static_boxes']+collision_boxes(out/'pillar.sdf')
    envelope_path=out.parent/'vehicle-envelope-002/envelope.json';radius=json.loads(envelope_path.read_text())['collision_sphere_radius_m']
    clearance=min(point_box_distance((v['y']+10,v['x']+10,-v['z']),b)-radius for v in truth for b in geometry)
    goal_gap=min(math.dist((v['y']+10,v['x']+10),protocol['goal']) for v in truth)
    vision_path=out/'runtime/vision/completion.json';vision=json.loads(vision_path.read_text())
    for f,digest in vision['inputs'].items():
        if file_sha256(f)!=digest:raise ValueError('Vision evidence changed: '+f)
    if vision['status']!='live_complete_not_flight_certified' or vision['counts']['inferred']<=0:raise ValueError('Vision evidence incomplete')
    registration,registration_inputs=measure(out)
    if registration['unknown_sequences'] or registration['max_pose_projection_budget_m']>.20:raise ValueError('Empirical registration budget failed')
    write_record(out/'map-registration.json',dict(**registration,status='within_frozen_empirical_budget_not_universal_certificate',inputs={str(p):file_sha256(p) for p in registration_inputs}))
    metrics=dict(ordered_events=[events[i] for i in indices],replan_count=len(r['replans']),sensor_plan_replay_verified=all(plan_replays),resumed_samples=len(resumed),min_lidar_m=min(s['lidar_m'] for s in samples),max_speed_m_s=max(math.hypot(s['position']['vn'],s['position']['ve']) for s in samples),max_cross_track_m=max(s['cross_track_m'] for s in samples),final_estimated_goal_gap_m=final_map_gap,route_duration_s=samples[-1]['monotonic']-samples[0]['monotonic'],truth_goal_closest_m=goal_gap,whole_run_collision_sphere_clearance_m=clearance,landing_confirmed=runtime['landing_confirmed'],final_armed=runtime['final_armed'],owned_processes_exited=runtime['owned_processes_exited'],vision_frames=vision['counts']['inferred'])
    passed=r['status']=='sensor_replan_route_completed' and runtime['status']=='sitl_hover_landed_disarmed' and runtime['landing_confirmed'] and runtime['final_armed'] is False and runtime['owned_processes_exited'] and clearance>0 and bool(resumed)
    paths=[out/'protocol.json',out/'route.json',out/'runtime/receipt.json',out/'runtime/telemetry.jsonl',out/'pillar.sdf',out/'pillar-receipt.json',out/'prearm-alignment.json',out/'inair-alignment.json',log,vision_path,envelope_path,Path(__file__).resolve(),out/'scan-pose.jsonl',out/'evidence-replay.json',out/'map-registration.json']+map_files+sorted(out.glob('event-map-*.json'))
    write_record(out/'completion.json',dict(status='sensor_stop_replan_resume_goal_verified' if passed else 'blocked',metrics=metrics,scope='single newly inserted stationary pillar in PX4 SITL; no fixture coordinates supplied to planner',dynamic_object_tracking_verified=False,physical_flight_certified=False,training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths}))
    print(json.dumps(dict(passed=passed,metrics=metrics),indent=2))

if __name__=='__main__':main()
