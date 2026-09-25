"""Bounded vertical-only 2m SITL check of final in-air heading alignment."""
import asyncio
import json
import os
from pathlib import Path
from scripts.flight import fly_px4_shadow_hover as trial
from scripts.flight import retest_sim_heading as profile
from scripts.flight.diagnose_gps_health import read_samples
from scripts.flight.check_live_alignment import evaluate
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256

OUT=trial.ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1/final-heading-001'


def assess(a,b):
    result=evaluate(a,b)
    result['heading_good_for_control_all']=bool(a) and all(r.get('heading_good_for_control') is True for r in a)
    result['horizontal_flight_ready']=result['passed'] and result['heading_good_for_control_all']
    return result


async def post_hover(drone,state,origin):
    log=next((trial.OUT/'px4-work/log').rglob('*.ulg'))
    rows=list(read_samples(log,{'vehicle_local_position','vehicle_local_position_groundtruth'},allow_incomplete_tail=True))
    end=max(r['timestamp'] for _,_,r in rows)
    a=[r for n,m,r in rows if n=='vehicle_local_position' and m==0 and r['timestamp']>=end-3000000]
    b=[r for n,m,r in rows if n=='vehicle_local_position_groundtruth' and m==0 and r['timestamp']>=end-3000000]
    result=assess(a,b)
    write_record(OUT/'inair-alignment.json',dict(**result,training_admitted=False,promotable=False))
    print('inair_heading_max',result['max_heading_error_deg'],'final_aligned',result['heading_good_for_control_all'],flush=True)
    if not result['horizontal_flight_ready']:raise RuntimeError('In-air final alignment failed; landing without horizontal flight')


async def main():
    OUT.mkdir(parents=True,exist_ok=False)
    prior=profile.OUT/'protocol.json';p=json.loads(prior.read_text())
    write_record(OUT/'protocol.json',dict(takeoff_altitude_m=2.,hover_s=10,max_altitude_m=2.8,max_drift_m=1.,
        horizontal_motion_authorized=False,heading_gate_deg=1.,position_residual_gate_m=.05,
        parameters=p['parameters'],inputs={str(q):file_sha256(q) for q in [prior,Path(__file__).resolve(),Path(trial.__file__).resolve()]},training_admitted=False,promotable=False))
    os.environ.update(PX4_PARAM_EKF2_DECL_TYPE='0',PX4_PARAM_EKF2_MAG_DECL=str(p['declination_deg']))
    profile.OUT=OUT;trial.OUT=OUT/'runtime'
    await trial.main(fly=True,preflight_hook=profile.gate,takeoff_altitude=2.,post_hover_hook=post_hover)


if __name__=='__main__':asyncio.run(main())
