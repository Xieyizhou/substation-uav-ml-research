"""Read-only evidence aggregation for the bounded wall stop trial."""
import json
from scripts.flight.fly_obstacle_stop import OUT
from scripts.flight.diagnose_gps_health import read_samples
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for path,digest in protocol['inputs'].items():
        if file_sha256(path)!=digest:raise ValueError('Input changed: '+path)
    receipt=json.loads((OUT/'runtime/receipt.json').read_text())
    b=json.loads((OUT/'braking.json').read_text());rows=b['samples'];t=b['trigger_monotonic']
    after=[r for r in rows if r['monotonic']>=t]
    log=next((OUT/'runtime/px4-work/log').rglob('*.ulg'))
    truth=[v for n,m,v in read_samples(log,{'vehicle_local_position_groundtruth'})]
    # Ground-truth NED y is world ENU x. Wall near face world x=-6.8.
    center_gap=-6.8-max(v['y'] for v in truth)
    metrics=dict(max_speed_m_s=max(r['speed_m_s'] for r in rows),trigger_range_m=after[0]['lidar_nearest_m'],min_range_m=min(r['lidar_nearest_m'] for r in rows),trigger_speed_m_s=after[0]['speed_m_s'],first_below_0_1_host_seconds=next(r['monotonic']-t for r in after if r['speed_m_s']<.1),east_overshoot_m=max(r['position']['east'] for r in after)-after[0]['position']['east'],final_speed_m_s=after[-1]['speed_m_s'],whole_run_truth_center_wall_gap_min_m=center_gap)
    stable=[r for r in after if r['monotonic']>=after[-1]['monotonic']-1.]
    passed=(b['status']=='actual_stop_observed' and b['motion_observed'] and after[-1]['monotonic']-t>=1 and all(r['speed_m_s']<.1 for r in stable) and center_gap>.7 and receipt['status']=='sitl_hover_landed_disarmed' and receipt['landing_confirmed'] and receipt['final_armed'] is False and receipt['owned_processes_exited'])
    paths=[OUT/'protocol.json',OUT/'fixture.sdf',OUT/'fixture-receipt.json',OUT/'braking.json',OUT/'runtime/receipt.json',OUT/'runtime/telemetry.jsonl',OUT/'prearm-alignment.json',OUT/'inair-alignment.json',log]
    write_record(OUT/'completion.json',dict(status='bounded_obstacle_stop_verified' if passed else 'needs_investigation',metrics=metrics,simulation_only=True,avoidance_verified=False,training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths}))
    print(json.dumps(dict(passed=passed,metrics=metrics),indent=2))

if __name__=='__main__':main()
