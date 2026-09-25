"""Execute the frozen eighty-view diagnostic matrix, preserving absent targets."""
import asyncio
import copy
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from src.vision.canonical.plan import read_record,write_record,rotate
from src.vision.canonical.collect import collect


async def main():
    base=ROOT/'data/research/ml_training_recovery_v1'
    matrix_path=base/'control-matrix-v1/matrix.json';matrix=read_record(matrix_path)
    out=base/'control-matrix-capture-v1';out.mkdir(exist_ok=True)
    groups=defaultdict(list)
    for pair in matrix['pairs']:
        if pair['eligible_for_capture']:
            for view in pair['views']:groups[(pair['source_plan_path'],view['occlusion_proxy'])].append((pair,view))
    runs=[]
    for (source_path,stratum),members in sorted(groups.items()):
        source_path=Path(source_path);source=read_record(source_path)
        assert file_sha256(source_path)==matrix['inputs'][str(source_path)]
        folder=out/f'{source["map_id"]}-{stratum}'/'plan';folder.mkdir(parents=True,exist_ok=True)
        for name,digest in source['files'].items():
            p=source_path.parent/name
            assert file_sha256(p)==digest
            (folder/name).write_bytes(p.read_bytes())
        tree=ET.parse(folder/'sensor_source.sdf')
        link=tree.find('.//link[@name="research_camera_link"]')
        extrinsic=list(map(float,link.findtext('pose').split()))[:3]
        assert abs(float(tree.find('.//sensor[@name="research_rgb"]/camera/horizontal_fov').text)-1.466)<1e-9
        views=[]
        for pair,v in members:
            delta=rotate(v['orientation'],extrinsic)
            row={'map_id':pair['map_id'],'object_id':v['object_id'],'category':v['category'],
                 'camera_position':v['camera_position'],'orientation':v['orientation'],
                 'position':[a-b for a,b in zip(v['camera_position'],delta)],
                 'family':f'{pair["identity"]}:{v["object_id"]}',
                 'matrix_pair_identity':pair['identity'],'occlusion_proxy':stratum,
                 'diagnostic_projected_aabb':v['diagnostic_projected_aabb']}
            row['view_id']=object_sha256(row);views.append(row)
        plan={k:copy.deepcopy(v) for k,v in source.items() if k!='identity'}
        plan.update(calibration_views=views,pilot_views=[],diagnostic_only=True,training_admitted=False,
                    diagnostic_allow_expected_absence=True,automatic_training=False,
                    diagnostic_purpose='occlusion_stratified_control_matrix',matrix_sha256=file_sha256(matrix_path))
        plan=write_record(folder/'plan.json',plan)
        capture=folder.parent/'capture';receipt_path=capture/'collection-receipt.json'
        if receipt_path.exists():
            receipt=read_record(receipt_path)
            assert receipt['plan_identity']==plan['identity']
        else:
            print('START',folder.parent.name,len(views),flush=True)
            receipt=await collect(folder/'plan.json',capture,mode='calibration')
        runs.append({'name':folder.parent.name,'plan_path':str(folder/'plan.json'),'receipt_path':str(receipt_path),
                     'status':receipt['status'],'captured':sum(r['status']=='captured' for r in receipt['views']),
                     'requested':len(views),'error':receipt.get('error')})
        write_json(out/'progress.json',{'runs':runs,'matrix_sha256':file_sha256(matrix_path),'training_admitted':False})
        print('DONE',runs[-1],flush=True)
        if receipt['status']!='complete_pending_review':break


if __name__=='__main__':asyncio.run(main())
