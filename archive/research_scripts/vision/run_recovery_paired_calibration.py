#!/usr/bin/env python3
"""Run isolated, non-training annotation-mode calibration at bound poses."""
import argparse
import asyncio
import copy
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from src.vision.canonical.plan import read_record,write_record,quaternion,rotate,visible
from src.vision.canonical.collect import collect

BASE=ROOT/'data/research/ml_training_recovery_v1'
OUT=BASE/'paired-calibration-v1'


def controls(plan,folder):
    """Select two common bearings with equal range/height in unchanged scenes."""
    objects=plan['objects']
    config=json.loads((folder/'obstacles.json').read_text())
    origin=config['gazebo_world_origin_m']
    link=ET.parse(folder/'sensor_source.sdf').find('.//link[@name="research_camera_link"]')
    extrinsic=[float(v) for v in link.findtext('pose').split()][:3]
    def view(obj,bearing,distance,height):
        b=obj['bounds'];target=[(b[0]+b[1])/2,(b[2]+b[3])/2,(b[4]+b[5])/2]
        a=math.radians(bearing);camera=[target[0]+distance*math.cos(a),target[1]+distance*math.sin(a),height]
        if not (origin[0]+.25<camera[0]<origin[0]+config['width']-.25 and origin[1]+.25<camera[1]<origin[1]+config['height']-.25):return None
        if not visible(camera,target,objects,obj['name']):return None
        if any(o['bounds'][0]-.25<=camera[0]<=o['bounds'][1]+.25 and o['bounds'][2]-.25<=camera[1]<=o['bounds'][3]+.25 and o['bounds'][4]-.25<=height<=o['bounds'][5]+.25 for o in objects):return None
        q=quaternion(0,math.atan2(height-target[2],distance),math.atan2(target[1]-camera[1],target[0]-camera[0]))
        delta=rotate(q,extrinsic)
        row={'map_id':plan['map_id'],'object_id':obj['name'],'category':obj['category'],
             'bearing':bearing,'distance':distance,'height':height,'offset':0,
             'position':[x-y for x,y in zip(camera,delta)],'orientation':q,'camera_position':camera,
             'family':f'{plan["map_id"]}:{obj["name"]}:{bearing}'}
        row['view_id']=object_sha256(row)
        return row
    cabinets=sorted((o for o in objects if o['category']=='cabinet'),key=lambda o:o['name'])
    switches=sorted((o for o in objects if o['category']=='switchgear'),key=lambda o:o['name'])
    for cab in cabinets:
        for switch in switches:
            for distance in (6,10,16):
                for height in (1.5,2.5,4):
                    pairs=[]
                    for bearing in range(0,360,45):
                        a,b=view(cab,bearing,distance,height),view(switch,bearing,distance,height)
                        if a and b:pairs.append((a,b))
                    if len(pairs)>=2:return [v for pair in pairs[:2] for v in pair]
    raise ValueError('No two common unobstructed cabinet/switchgear control bearings')


def prepare():
    OUT.mkdir(exist_ok=True)
    hold_path=BASE/'annotation-compatibility-audit-v1/canonical-scope-hold.json'
    held=read_record(hold_path)
    inventory=json.loads((BASE/'canonical-increment-audit-v1/source-inventory.json').read_text())
    plans={r['plan_identity']:Path(r['plan_path']) for r in inventory['collections']}
    groups=defaultdict(list)
    for row in held['selected']:groups[row['plan_identity']].append(row)
    runs=[];controlled_maps=set();control_inventory=[]
    for identity,rows in sorted(groups.items()):
        source_path=plans[identity];source=read_record(source_path)
        source_views={v['view_id']:v for v in source['pilot_views']}
        views=[source_views[r['view_id']] for r in rows]
        control_views=[]
        if source['map_id'] not in controlled_maps:
            control_views=controls(source,source_path.parent)
            controlled_maps.add(source['map_id'])
            control_inventory.append({'map_id':source['map_id'],'source_plan_identity':identity,'views':control_views,
                                      'matched':'same source scene, horizontal range, absolute height and bearing; pitch may differ with target height'})
        for mode in ('visible_2d','full_2d'):
            name=f'{source["map_id"]}-{identity[:8]}-{mode}'
            folder=OUT/name/'plan'
            if folder.exists():
                plan=read_record(folder/'plan.json')
            else:
                folder.mkdir(parents=True)
                for filename,digest in source['files'].items():
                    payload=(source_path.parent/filename).read_bytes()
                    if file_sha256(source_path.parent/filename)!=digest:raise ValueError('Changed source snapshot')
                    (folder/filename).write_bytes(payload)
                world=folder/'world.sdf';tree=ET.parse(world)
                sensors=[s for s in tree.iter('sensor') if s.get('type')=='boundingbox_camera']
                if len(sensors)!=1:raise ValueError('Expected one box sensor')
                sensors[0].find('camera/box_type').text=mode
                tree.write(world,encoding='utf-8',xml_declaration=True)
                plan={k:copy.deepcopy(v) for k,v in source.items() if k!='identity'}
                unique={v['view_id']:v for v in views+control_views}
                plan.update(parent_plan_identity=identity,annotation_mode=mode,
                            calibration_views=list(unique.values()),pilot_views=[],
                            diagnostic_only=True,training_admitted=False,automatic_training=False,
                            diagnostic_purpose='paired_annotation_extent_and_cabinet_switchgear_controls',
                            source_plan_path=str(source_path),source_plan_sha256=file_sha256(source_path))
                plan['files']['world.sdf']=file_sha256(world)
                plan=write_record(folder/'plan.json',plan)
            runs.append({'name':name,'map_id':source['map_id'],'source_plan_identity':identity,'mode':mode,
                         'plan_path':str(folder/'plan.json'),'plan_identity':plan['identity'],
                         'requested_views':len(plan['calibration_views']),'held_target_view_ids':[r['view_id'] for r in rows],
                         'control_view_ids':[v['view_id'] for v in control_views]})
    manifest={'schema_version':1,'source_hold_sha256':file_sha256(hold_path),'runs':runs,'controls':control_inventory,
              'training_admitted':False,'px4_started':False}
    manifest['identity']=object_sha256(manifest)
    path=OUT/'plan-manifest.json'
    if path.exists() and read_record(path)!=manifest:raise ValueError('Existing calibration manifest differs')
    write_json(path,manifest)
    return manifest


async def execute(manifest):
    results=[]
    for run in manifest['runs']:
        folder=OUT/run['name']/'capture'
        receipt_path=folder/'collection-receipt.json'
        if receipt_path.exists():
            receipt=read_record(receipt_path)
            if receipt['plan_identity']!=run['plan_identity']:raise ValueError('Existing capture plan mismatch')
        else:
            print('START',run['name'],run['requested_views'],flush=True)
            receipt=await collect(run['plan_path'],folder,mode='calibration')
        results.append({**run,'receipt_path':str(receipt_path),'collection_identity':receipt['identity'],
                        'status':receipt['status'],'captured':sum(v['status']=='captured' for v in receipt['views']),
                        'error':receipt.get('error')})
        write_json(OUT/'execution-progress.json',{'runs':results,'training_admitted':False})
        print('DONE',run['name'],receipt['status'],receipt.get('error'),flush=True)
        if receipt['status']!='complete_pending_review':
            print('STOP: bounded capture failed; inspect evidence before continuing',flush=True)
            break


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    manifest=prepare()
    print(json.dumps({'runs':len(manifest['runs']),'requested_capture_frames':sum(r['requested_views'] for r in manifest['runs'])}))
    if args.execute:asyncio.run(execute(manifest))
