"""Offline empirical registration-error budget; never supplies truth to flight."""
import bisect
import json
import math
from pathlib import Path
from scripts.flight.diagnose_gps_health import read_samples
from scripts.flight.fly_px4_shadow_hover import ROOT
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

BASE=ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1'
OUT=BASE/'mapping-error-budget-v1'


def interpolate(rows,times,t):
    i=bisect.bisect_left(times,t)
    if i==0 or i==len(rows):return None
    a,b=rows[i-1],rows[i]
    if b['timestamp']-a['timestamp']>40000:return None
    f=(t-a['timestamp'])/(b['timestamp']-a['timestamp'])
    h=a['heading']+f*((b['heading']-a['heading']+math.pi)%(2*math.pi)-math.pi)
    return dict(x=a['x']+f*(b['x']-a['x']),y=a['y']+f*(b['y']-a['y']),heading=h)


def measure(directory):
    journal=directory/'scan-pose.jsonl'
    log=next((directory/'runtime/px4-work/log').rglob('*.ulg'))
    rows=[r for n,m,r in read_samples(log,{'vehicle_local_position_groundtruth'}) if m==0]
    times=[r['timestamp'] for r in rows];values=[];unknown=[]
    with journal.open() as stream:
        for line in stream:
            r=json.loads(line)
            if r['new_returns']==0:continue
            t=r['scan']['timestamp_s']*1e6;truth=interpolate(rows,times,t)
            if truth is None:
                unknown.append(r['scan']['sequence']);continue
            p=r['local'][1];a=r['attitude'][1];f=r['local_frame']
            if f['rotation_deg']!=0:raise ValueError('Only frozen zero-rotation frame supported')
            position_error=math.hypot(p['east']+f['east_offset_m']-(truth['y']+10),p['north']+f['north_offset_m']-(truth['x']+10))
            yaw_error=abs((math.radians(a['yaw'])-truth['heading']+math.pi)%(2*math.pi)-math.pi)
            # Worst planar heading displacement over the accepted <=3m rays
            # plus .1m horizontal sensor lever arm. The .5m vertical lever
            # and <=1deg roll/pitch are added as a conservative approximation.
            tilt=math.radians(math.hypot(a['roll'],a['pitch']))
            bound=position_error+2*3.1*math.sin(yaw_error/2)+.5*math.sin(tilt)+3.1*(1-math.cos(tilt))
            values.append(dict(sequence=r['scan']['sequence'],timestamp_s=t/1e6,position_error_m=position_error,yaw_error_deg=math.degrees(yaw_error),pose_projection_budget_m=bound))
    if not values:raise ValueError('No eligible empirical evidence')
    return dict(run=directory.name,accepted_pairs=len(values),unknown_sequences=unknown,max_position_error_m=max(v['position_error_m'] for v in values),max_yaw_error_deg=max(v['yaw_error_deg'] for v in values),max_pose_projection_budget_m=max(v['pose_projection_budget_m'] for v in values),samples=values),[journal,log,directory/'evidence-replay.json']


def main():
    results=[];files=[Path(__file__).resolve()]
    for run in ('replan-stable-v2-baseline-1','replan-capture-v4-baseline-1'):
        value,inputs=measure(BASE/run);results.append(value);files.extend(inputs)
    OUT.mkdir(exist_ok=False)
    maximum=max(r['max_pose_projection_budget_m'] for r in results)
    # Freeze a diagnostic budget only if the observations fit it; this is not
    # a universal error bound and every later flight must be checked again.
    write_record(OUT/'measurement.json',dict(status='empirical_measurement_only',results=results,proposed_additional_map_budget_m=.20,all_measured_pairs_within_budget=maximum<=.20,unknown_pair_count=sum(len(r['unknown_sequences']) for r in results),scope='pose/time-registration component of <=3m planar rays; not a hidden-surface, sensor-range, or universal accuracy certificate',training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in files}))
    print(json.dumps([{k:v for k,v in r.items() if k!='samples'} for r in results],indent=2))


if __name__=='__main__':main()
