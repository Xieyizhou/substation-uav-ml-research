"""Capture a genuinely different simple world package for development recheck."""
import asyncio,copy,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from src.vision.canonical.plan import read_record,write_record
from src.vision.canonical.collect import collect

async def main():
 base=ROOT/'data/research/ml_training_recovery_v1/simple-independent-scene-v1';base.mkdir(exist_ok=True)
 srcp=ROOT/'data/research/canonical_views_v1/simple/plan.json';src=read_record(srcp);folder=base/'plan';folder.mkdir(exist_ok=True)
 for n,digest in src['files'].items():
  p=srcp.parent/n;assert file_sha256(p)==digest;(folder/n).write_bytes(p.read_bytes())
 tree=ET.parse(folder/'world.sdf');s=[x for x in tree.iter('sensor') if x.get('type')=='boundingbox_camera'];assert len(s)==1;s[0].find('camera/box_type').text='full_2d';tree.write(folder/'world.sdf',encoding='utf-8',xml_declaration=True)
 plan={k:copy.deepcopy(v) for k,v in src.items() if k!='identity'};plan.update(annotation_mode='full_2d',calibration_views=src['calibration_views'],pilot_views=[],diagnostic_only=True,diagnostic_allow_expected_absence=True,training_admitted=False,automatic_training=False,diagnostic_purpose='independent_simple_scene_negative_ab',source_plan_path=str(srcp),source_plan_sha256=file_sha256(srcp));plan['files']['world.sdf']=file_sha256(folder/'world.sdf');plan=write_record(folder/'plan.json',plan)
 cap=base/'capture';receipt=await collect(folder/'plan.json',cap,mode='calibration');assert receipt['status']=='complete_pending_review';write_json(base/'manifest.json',{'source_plan':str(srcp),'plan_path':str(folder/'plan.json'),'receipt_path':str(cap/'collection-receipt.json'),'requested':len(plan['calibration_views']),'captured':sum(v['status']=='captured' for v in receipt['views']),'world_sha256':plan['files']['world.sdf'],'training_admitted':False})
 print(json.dumps({'captured':sum(v['status']=='captured' for v in receipt['views']),'world_sha256':plan['files']['world.sdf']}))
if __name__=='__main__':asyncio.run(main())
