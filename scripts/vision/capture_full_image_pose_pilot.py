"""Capture eight original-only, all-instance-screened views before any variants."""
import asyncio
import copy
import fcntl
import xml.etree.ElementTree as ET
from pathlib import Path
from scripts.vision.supplement_full_image_poses import OUT as SCREEN,SOURCE,read,save,file_sha256,verify_tree,ROOT
from scripts.vision.capture_appearance_recovery_pilot import transform,assert_world
from src.ml.artifacts import object_sha256
from src.vision.canonical.plan import read_record,write_record
from src.vision.canonical.gates import validate_preflight
from src.vision.canonical.recovery import resumed_views
from src.vision.canonical.collect import collect

OUT=SCREEN/'original-pilot-v1'

def prepare():
    path=OUT/'protocol.json'
    if path.exists():verify_tree(path);return read(path)
    verify_tree(SCREEN/'screen.json');s=read(SCREEN/'screen.json')
    if s['status']!='geometric_candidates_ready_for_original_only_pilot' or len(s['selected'])!=8:raise ValueError('Incomplete geometric candidate coverage')
    source=read_record(SOURCE);folder=OUT/'plan';folder.mkdir(parents=True,exist_ok=False)
    for name,digest in source['files'].items():
        p=SOURCE.parent/name
        if file_sha256(p)!=digest:raise ValueError('Source hash changed')
        (folder/name).write_bytes(p.read_bytes())
    tree=ET.parse(folder/'world.sdf');transform(tree.getroot(),'original',{},source['objects']);ET.indent(tree,space='  ');tree.write(folder/'world.sdf',encoding='utf-8',xml_declaration=True)
    assert_world(SOURCE.parent/'world.sdf',folder/'world.sdf','original',{},source['objects'])
    views=[]
    for chosen in s['selected']:
        v=copy.deepcopy(chosen['view']);sid=v['view_id'];v.update(source_view_id=sid,pose_id=sid,view_id=object_sha256({'source':sid,'experiment':'full-image-original-pilot-v1'}),variant='original',derivation_group='full-image-pose:'+sid,data_role='new_training_candidate',training_admitted=False,promotable=False);views.append(v)
    plan=copy.deepcopy(source);plan.pop('identity');plan.update(annotation_mode='full_2d',label_mode='visual-instance',hierarchy_mode='top-level-equipment',calibration_views=views,pilot_views=[],diagnostic_only=True,diagnostic_require_expected_presence=True,automatic_training=False,training_admitted=False,promotable=False)
    plan['files']['world.sdf']=file_sha256(folder/'world.sdf');plan=write_record(folder/'plan.json',plan);gate,_,_=validate_preflight(plan,folder,views)
    paths=[SCREEN/'screen.json',folder/'plan.json',Path(__file__),ROOT/'scripts/vision/capture_appearance_recovery_pilot.py']
    return save(path,dict(status='frozen_original_only',plan_path=str(folder/'plan.json'),gate=gate,expected_frames=8,review_policy='Every full-image supervised instance requires explicit visual review; use exact-alignment replay when uncertain. No variants or training before the entire pilot passes. No area-only admission.',inputs={str(p):file_sha256(p) for p in paths}))

async def main():
    p=prepare()
    with (OUT/'capture.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);pp=Path(p['plan_path']);plan=read_record(pp);out=OUT/'capture';rp=out/'collection-receipt.json'
        gate,config,mapping=validate_preflight(plan,pp.parent,plan['calibration_views'])
        if out.exists():
            if not rp.exists():raise ValueError('Incomplete prior attempt retained; investigate before retry')
            restored,_=resumed_views(rp,plan,'calibration',plan['calibration_views'],config=config,check_version=gate['check_version'],instance_mapping=mapping)
            if len(restored)!=8:raise ValueError('Prior capture incomplete; no silent retry reset')
            r=read_record(rp)
        else:r=await collect(pp,out,mode='calibration')
        n=sum(v['status']=='captured' for v in r['views'])
        save(OUT/'progress.json',dict(status='complete_pending_full_image_review' if n==8 and r['status']=='complete_pending_review' else 'capture_held',captured=n,expected=8,inputs={str(rp):file_sha256(rp),str(OUT/'protocol.json'):file_sha256(OUT/'protocol.json')}))
        print('ORIGINAL_PILOT_CAPTURED',n,r['status'],flush=True)

if __name__=='__main__':asyncio.run(main())
