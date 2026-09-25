"""Isolated SITL magnetic-declination consistency trial, gated before arm."""
import asyncio
import math
import os
import statistics
from pathlib import Path
from scripts.flight import fly_px4_shadow_hover as trial
from scripts.flight.diagnose_gps_health import read_samples
from scripts.flight.check_live_alignment import evaluate
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256

OUT=trial.ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1/heading-profile-003'
PRIOR=trial.ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1/alignment-002/runtime/px4-work/log'


def measured_declination(path):
    rows=list(read_samples(path,{'vehicle_magnetometer','vehicle_attitude_groundtruth'}))
    end=max(r['timestamp'] for _,_,r in rows)
    attitudes=[r for n,m,r in rows if n=='vehicle_attitude_groundtruth' and m==0 and r['timestamp']>end-5000000]
    values=[]
    for name,m,row in rows:
        if name!='vehicle_magnetometer' or m!=0 or row['timestamp']<=end-5000000:continue
        a=min(attitudes,key=lambda a:abs(a['timestamp']-row['timestamp']))
        if abs(a['timestamp']-row['timestamp'])>33334:continue
        w,x,y,z=a['q'];bx,by,bz=row['magnetometer_ga']
        # Body FRD magnetic vector rotated to groundtruth NED.
        north=(1-2*(y*y+z*z))*bx+2*(x*y-w*z)*by+2*(x*z+w*y)*bz
        east=2*(x*y+w*z)*bx+(1-2*(x*x+z*z))*by+2*(y*z-w*x)*bz
        values.append(math.degrees(math.atan2(east,north)))
    if len(values)<20 or not all(math.isfinite(v) for v in values):raise ValueError('Insufficient magnetic evidence')
    return statistics.median(values),len(values)


async def gate(drone,state,origin):
    # Logger is already running; inspect complete records only in this live snapshot.
    await asyncio.sleep(10)
    logs=list((trial.OUT/'px4-work/log').rglob('*.ulg'))
    if len(logs)!=1:raise ValueError('Ambiguous live log')
    rows=list(read_samples(logs[0],{'vehicle_local_position','vehicle_local_position_groundtruth'},allow_incomplete_tail=True))
    end=max(r['timestamp'] for _,_,r in rows)
    a=[r for n,m,r in rows if n=='vehicle_local_position' and m==0 and r['timestamp']>=end-5000000]
    b=[r for n,m,r in rows if n=='vehicle_local_position_groundtruth' and m==0 and r['timestamp']>=end-5000000]
    result=evaluate(a,b)
    write_record(OUT/'prearm-alignment.json',dict(**result,training_admitted=False,promotable=False))
    print('heading_error_deg',result['max_heading_error_deg'],flush=True)
    if not result['passed']:raise RuntimeError('Alignment gate failed; refusing arm')


async def main():
    OUT.mkdir(parents=True,exist_ok=False)
    prior=next(PRIOR.rglob('*.ulg'));declination,count=measured_declination(prior)
    write_record(OUT/'protocol.json',dict(simulation_only=True,declination_deg=declination,magnetic_samples=count,
        purpose='Match estimator declination to measured simulator magnetic field; not an earth-field calibration',
        parameters={'EKF2_DECL_TYPE':0,'EKF2_MAG_DECL':declination},heading_gate_deg=1.,position_gate_m=.05,
        inputs={str(p):file_sha256(p) for p in [prior,Path(__file__).resolve()]},training_admitted=False,promotable=False))
    os.environ.update(PX4_PARAM_EKF2_DECL_TYPE='0',PX4_PARAM_EKF2_MAG_DECL=str(declination))
    trial.OUT=OUT/'runtime'
    await trial.main(fly=True,preflight_hook=gate)


if __name__=='__main__':asyncio.run(main())
