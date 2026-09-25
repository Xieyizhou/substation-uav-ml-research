"""Capture the unused pilot views from alternate world snapshots."""
import asyncio,copy,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from src.vision.canonical.plan import read_record,write_record
from src.vision.canonical.collect import collect
SOURCES={m:ROOT/f'data/research/canonical_views_v1/{m}-clean-view-calibration-plan-v1/plan.json' for m in ('complex','medium','simple')}

async def main():
 base=ROOT/'data/research/ml_training_recovery_v1/cross-scene-additional-v1';base.mkdir(exist_ok=True);runs=[]
 for m,srcp in SOURCES.items():
  src=read_record(srcp);folder=base/m/'plan';folder.mkdir(parents=True,exist_ok=True)
  for n,digest in src['files'].items():
   p=srcp.parent/n;assert file_sha256(p)==digest;(folder/n).write_bytes(p.read_bytes())
  tree=ET.parse(folder/'world.sdf');s=[x for x in tree.iter('sensor') if x.get('type')=='boundingbox_camera'];assert len(s)==1;s[0].find('camera/box_type').text='full_2d';tree.write(folder/'world.sdf',encoding='utf-8',xml_declaration=True)
  plan={k:copy.deepcopy(v) for k,v in src.items() if k!='identity'};views=src['pilot_views'][:]
  plan.update(annotation_mode='full_2d',calibration_views=views,pilot_views=[],diagnostic_only=True,diagnostic_allow_expected_absence=True,training_admitted=False,automatic_training=False,diagnostic_purpose='cross_scene_additional_views',source_plan_path=str(srcp),source_plan_sha256=file_sha256(srcp));plan['files']['world.sdf']=file_sha256(folder/'world.sdf');plan=write_record(folder/'plan.json',plan)
  cap=base/m/'capture';rp=cap/'collection-receipt.json';receipt=read_record(rp) if rp.exists() else await collect(folder/'plan.json',cap,mode='calibration');assert receipt['plan_identity']==plan['identity']
  runs.append({'map_id':m,'plan_path':str(folder/'plan.json'),'receipt_path':str(rp),'requested':len(views),'captured':sum(x['status']=='captured' for x in receipt['views']),'status':receipt['status'],'world_sha256':plan['files']['world.sdf']});write_json(base/'progress.json',{'runs':runs,'training_admitted':False});print('DONE',runs[-1],flush=True)
 manifest={'runs':runs,'purpose':'additional_cross_scene_development_recheck','training_admitted':False};manifest['identity']=object_sha256(manifest);write_json(base/'manifest.json',manifest);print(json.dumps({'captured':sum(x['captured'] for x in runs),'identity':manifest['identity']}))
if __name__=='__main__':asyncio.run(main())
