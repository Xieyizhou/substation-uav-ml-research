"""Timestamp-bracketed moving-flight diagnostic, not a replacement flight gate."""
import argparse
import bisect
import json
import math
from pathlib import Path
from scripts.flight.diagnose_gps_health import read_samples
from src.planner.local_frame import LocalFrame
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def interpolate(truth,timestamp,max_gap_us=40000):
    stamps=[v['timestamp'] for v in truth]
    if any(b<=a for a,b in zip(stamps,stamps[1:])):raise ValueError('Non-increasing truth timestamps')
    i=bisect.bisect_left(stamps,timestamp)
    if i<len(truth) and stamps[i]==timestamp:return dict(truth[i],bracket_gap_us=0)
    if i==0 or i==len(truth):raise ValueError('Unbracketed timestamp')
    a,b=truth[i-1],truth[i];gap=b['timestamp']-a['timestamp']
    if gap>max_gap_us:raise ValueError('Truth bracket gap too large')
    f=(timestamp-a['timestamp'])/gap
    result={k:a[k]+f*(b[k]-a[k]) for k in ('x','y','z')}
    delta=(b['heading']-a['heading']+math.pi)%(2*math.pi)-math.pi
    result.update(heading=a['heading']+f*delta,timestamp=timestamp,bracket_gap_us=gap)
    if not all(math.isfinite(v) for v in result.values()):raise ValueError('Nonfinite truth')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();out=args.directory
    log=next((out/'runtime/px4-work/log').rglob('*.ulg'));all_rows=list(read_samples(log,{'vehicle_local_position','vehicle_local_position_groundtruth'}))
    truth=[v for n,m,v in all_rows if n=='vehicle_local_position_groundtruth' and m==0]
    frame=LocalFrame.from_mapping(json.loads((out/'inair-alignment.json').read_text())['local_frame'])
    pairs=[];unknown=[]
    for n,m,v in all_rows:
        if n!='vehicle_local_position' or m!=0:continue
        try:t=interpolate(truth,v['timestamp'])
        except ValueError as e:unknown.append(dict(timestamp_us=v['timestamp'],reason=str(e)));continue
        if -t['z']<1.6:continue
        pairs.append(dict(timestamp_us=v['timestamp'],bracket_gap_us=t['bracket_gap_us'],horizontal_error_m=math.dist(frame.to_map(v['y'],v['x']),(t['y']+10,t['x']+10)),heading_error_deg=abs(math.degrees((v['heading']-t['heading']+math.pi)%(2*math.pi)-math.pi))))
    result=dict(role='postflight interpolated diagnostic; raw nearest-timestamp metrics remain unchanged; interpolation is approximate between truth samples',pair_count=len(pairs),unknown=unknown,max_heading_error_deg=max(v['heading_error_deg'] for v in pairs),max_horizontal_error_m=max(v['horizontal_error_m'] for v in pairs),max_bracket_gap_us=max(v['bracket_gap_us'] for v in pairs),pairs=pairs,training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in (log,out/'inair-alignment.json',Path(__file__).resolve())})
    write_record(out/'moving-alignment-diagnostic.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','inputs')},indent=2))

if __name__=='__main__':main()
