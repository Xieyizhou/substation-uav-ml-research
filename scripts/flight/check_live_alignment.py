"""Timestamp-paired PX4 estimate and Gazebo groundtruth; never arms."""
import asyncio
import math
import re
import statistics
from pathlib import Path
from scripts.flight import fly_px4_shadow_hover as trial
from scripts.vision.probe_material_shadow import command
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256

OUT=trial.ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1/alignment-002'


def parse_samples(text):
    rows=[]
    for block in re.split(r'TOPIC: ',text)[1:]:
        row={}
        for key,value in re.findall(r'^\s+([a-z_]+):\s+([^\s]+)',block,re.M):
            try:row[key]=float(value)
            except ValueError:pass
        if all(k in row for k in ('timestamp','x','y','heading')):rows.append(row)
    return rows


def evaluate(estimated,truth):
    if any(not all(math.isfinite(r[k]) for k in ('timestamp','x','y','heading')) for r in estimated+truth):
        raise ValueError('Nonfinite alignment input')
    pairs=[];used=set()
    for a in estimated:
        choices=[(abs(a['timestamp']-b['timestamp']),i,b) for i,b in enumerate(truth) if i not in used]
        if not choices:break
        dt,i,b=min(choices,key=lambda v:v[0])
        if dt>33334:continue
        used.add(i)
        yaw=math.degrees((a['heading']-b['heading']+math.pi)%(2*math.pi)-math.pi)
        pairs.append(dict(timestamp_us=a['timestamp'],skew_us=dt,east_offset_m=b['y']+10-a['y'],north_offset_m=b['x']+10-a['x'],heading_error_deg=yaw))
    if len(pairs)<10:raise ValueError(f'Only {len(pairs)} synchronized pairs')
    east=statistics.median(p['east_offset_m'] for p in pairs);north=statistics.median(p['north_offset_m'] for p in pairs)
    residual=max(math.hypot(p['east_offset_m']-east,p['north_offset_m']-north) for p in pairs)
    heading=max(abs(p['heading_error_deg']) for p in pairs)
    stable_resets=all(all(k in r for r in estimated) and len({r[k] for r in estimated})==1 for k in ('xy_reset_counter','heading_reset_counter'))
    passed=residual<=.05 and heading<=1. and stable_resets and abs(east-1.5)<=.25 and abs(north-1.5)<=.25
    return dict(passed=passed,pairs=pairs,local_frame=dict(east_offset_m=east,north_offset_m=north,rotation_deg=0.),max_position_residual_m=residual,max_heading_error_deg=heading,reset_counters_stable=stable_resets,
        interpretation='Heading error is not absorbed into map rotation: ENU/NED axes are fixed by simulator. Static registration does not verify moving-flight alignment.')


async def check(drone,state,origin):
    topics=('vehicle_local_position','vehicle_local_position_groundtruth')
    raw=['','']
    for _ in range(25):
        snapshots=await asyncio.gather(*(command(str(trial.BUILD/'bin/px4-listener'),'--instance','7',t,'-n','1') for t in topics))
        for i,s in enumerate(snapshots):raw[i]+=s
        await asyncio.sleep(.1)
    paths=[]
    for topic,text in zip(topics,raw):
        p=OUT/(topic+'.txt');p.write_text(text);paths.append(p)
    result=evaluate(*(parse_samples(s) for s in raw))
    write_record(OUT/'alignment.json',dict(**result,training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths+[Path(__file__).resolve()]}))
    print(result['max_heading_error_deg'],result['max_position_residual_m'],flush=True)
    if not result['passed']:raise RuntimeError('Live coordinate/heading alignment gate failed; no flight authorized by gate')


if __name__=='__main__':
    trial.OUT=OUT/'runtime'
    asyncio.run(trial.main(fly=False,preflight_hook=check))
