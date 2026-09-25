"""Capture diagnostic full_2d views from alternate scene packages."""
import asyncio
import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from src.vision.canonical.plan import read_record,write_record
from src.vision.canonical.collect import collect

SOURCES={
 'complex':ROOT/'data/research/canonical_views_v1/complex-clean-view-calibration-plan-v1/plan.json',
 'medium':ROOT/'data/research/canonical_views_v1/medium-clean-view-calibration-plan-v1/plan.json',
 'simple':ROOT/'data/research/canonical_views_v1/simple-clean-view-calibration-plan-v1/plan.json',
}

async def main():
    base=ROOT/'data/research/ml_training_recovery_v1/cross-scene-recheck-v1';base.mkdir(exist_ok=True)
    runs=[]
    for map_id,source_path in SOURCES.items():
        source=read_record(source_path);source_world=source_path.parent/'world.sdf';assert file_sha256(source_world)==source['files']['world.sdf']
        folder=base/map_id/'plan';folder.mkdir(parents=True,exist_ok=True)
        for name,digest in source['files'].items():
            p=source_path.parent/name;assert file_sha256(p)==digest;(folder/name).write_bytes(p.read_bytes())
        tree=ET.parse(folder/'world.sdf');sensors=[s for s in tree.iter('sensor') if s.get('type')=='boundingbox_camera'];assert len(sensors)==1
        sensors[0].find('camera/box_type').text='full_2d';tree.write(folder/'world.sdf',encoding='utf-8',xml_declaration=True)
        views=list(source['calibration_views'])[:]
        plan={k:copy.deepcopy(v) for k,v in source.items() if k!='identity'}
        plan.update(annotation_mode='full_2d',calibration_views=views,pilot_views=[],diagnostic_only=True,diagnostic_allow_expected_absence=True,
                    training_admitted=False,automatic_training=False,diagnostic_purpose='cross_scene_negative_ab_recheck',source_plan_path=str(source_path),source_plan_sha256=file_sha256(source_path))
        plan['files']['world.sdf']=file_sha256(folder/'world.sdf');plan=write_record(folder/'plan.json',plan)
        capture=base/map_id/'capture';receipt_path=capture/'collection-receipt.json'
        if receipt_path.exists():receipt=read_record(receipt_path)
        else:
            print('START',map_id,len(views),flush=True);receipt=await collect(folder/'plan.json',capture,mode='calibration')
        assert receipt['plan_identity']==plan['identity']
        runs.append({'map_id':map_id,'source_plan':str(source_path),'alternate_world_sha256':plan['files']['world.sdf'],'plan_path':str(folder/'plan.json'),'receipt_path':str(receipt_path),'requested':len(views),'captured':sum(v['status']=='captured' for v in receipt['views']),'status':receipt['status']})
        write_json(base/'progress.json',{'runs':runs,'training_admitted':False})
        print('DONE',runs[-1],flush=True)
        if receipt['status']!='complete_pending_review':break
    manifest={'runs':runs,'source_maps':sorted(SOURCES),'training_admitted':False,'purpose':'independent_cross_scene_development_recheck'}
    manifest['identity']=object_sha256(manifest);write_json(base/'manifest.json',manifest)
    print(json.dumps({'runs':len(runs),'captured':sum(r['captured'] for r in runs),'identity':manifest['identity']}))

if __name__=='__main__':asyncio.run(main())
