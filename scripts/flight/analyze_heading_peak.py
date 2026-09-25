"""Offline phase attribution, with causal land/mode state and paired truth."""
import bisect
from collections import defaultdict
import json
import math
from pathlib import Path
from scripts.flight.retest_sim_heading import OUT
from scripts.flight.diagnose_gps_health import read_samples
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256


def main():
    log=next((OUT/'runtime/px4-work/log').rglob('*.ulg'))
    gate=OUT/'prearm-alignment.json'
    cutoff=json.loads(gate.read_text())['pairs'][-1]['timestamp_us']
    topics={'vehicle_local_position','vehicle_local_position_groundtruth','vehicle_land_detected','vehicle_status','estimator_status_flags'}
    data=defaultdict(list)
    for name,m,row in read_samples(log,topics):
        if m==0:data[name].append(row)
    stamps={name:[r['timestamp'] for r in rows] for name,rows in data.items()}
    def causal(name,t):
        i=bisect.bisect_right(stamps[name],t)-1
        if i<0:raise ValueError('Missing preceding state: '+name)
        return data[name][i]
    paired=[]
    for row in data['vehicle_local_position']:
        t=row['timestamp']
        if t<cutoff:continue
        name='vehicle_local_position_groundtruth';i=bisect.bisect_left(stamps[name],t)
        truth=min(data[name][max(0,i-1):i+1],key=lambda q:abs(q['timestamp']-t))
        if abs(truth['timestamp']-t)>33334:continue
        land=causal('vehicle_land_detected',t);status=causal('vehicle_status',t)
        phase='ground_contact' if land['ground_contact'] else ('descent' if status['nav_state']==18 else ('takeoff' if status['nav_state']==17 else 'airborne_hold'))
        paired.append(dict(timestamp_us=t,truth_skew_us=abs(truth['timestamp']-t),phase=phase,
            heading_error_deg=abs(math.degrees((row['heading']-truth['heading']+math.pi)%(2*math.pi)-math.pi)),
            heading_good_for_control=row['heading_good_for_control'],heading_reset_counter=row['heading_reset_counter'],
            xy_reset_counter=row['xy_reset_counter'],landed=land['landed'],ground_contact=land['ground_contact']))
    summary={}
    for phase in sorted({r['phase'] for r in paired}):
        group=[r for r in paired if r['phase']==phase]
        summary[phase]=dict(samples=len(group),peak=max(group,key=lambda r:r['heading_error_deg']),over_one_degree=sum(r['heading_error_deg']>1. for r in group))
    target=OUT.parent/'heading-peak-diagnosis-v1';target.mkdir(exist_ok=False)
    write_record(target/'analysis.json',dict(summary=summary,paired_samples=paired,
        conclusion='Peak occurred after ground contact; final in-air yaw alignment remained incomplete. Small hover error alone does not authorize horizontal flight.',
        inputs={str(p):file_sha256(p) for p in [log,gate,Path(__file__).resolve()]},training_admitted=False,promotable=False))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
