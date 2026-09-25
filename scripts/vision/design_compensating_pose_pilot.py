"""Freeze one deterministic compensation source; do not change existing data."""
import copy,shutil
from pathlib import Path
from scripts.vision.screen_compensating_instance_poses import OUT as SCREEN,prior
from scripts.vision.design_full_scene_poses import OUT as BASE,validate_preflight,write_record

OUT=BASE.parent/'compensating-pose-pilot-v1'


def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():p=prior.read(dest);prior.verify(p);return p
    sp=SCREEN/'result.json';s=prior.read(sp);prior.verify(s)
    if not s['compensating_candidates']:raise ValueError('No bounded candidate')
    selected=dict(s['compensating_candidates'][0]['view'],probe_id='C01')
    src=BASE/'plan/plan.json';p=prior.read(src)
    OUT.mkdir(exist_ok=True);folder=OUT/'plan';folder.mkdir(exist_ok=True)
    paths=[sp,src,Path(__file__)];plan=copy.deepcopy(p);plan.pop('identity',None)
    for name,sha in plan['files'].items():
        ip=src.parent/name
        if prior.file_sha256(ip)!=sha:raise ValueError('Source configuration changed')
        shutil.copy2(ip,folder/name);paths += [ip,folder/name]
    plan['calibration_views']=[selected];plan['pilot_views']=[]
    write_record(folder/'plan.json',plan);validate_preflight(plan,folder,[selected]);paths.append(folder/'plan.json')
    return prior.frozen(dest,dict(status='one_compensation_source_frozen',selected=[selected],plan_path=str(folder/'plan.json'),
        geometry_expectation=s['compensating_candidates'][0]['projected_counts'],
        policy='First candidate from frozen geometric ordering, not image/model ranking. No material changes. Full original-world labels required.',
        training_ready=False,training_started=False,inputs={str(p):prior.file_sha256(p) for p in paths}))


if __name__=='__main__':freeze();print('ONE_SOURCE_FROZEN_NO_CAPTURE')
