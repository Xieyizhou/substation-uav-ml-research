#!/usr/bin/env python3
"""Compare captured annotation modes, conditioning conclusions on RGB equality."""
import json
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from src.vision.canonical.plan import read_record,pose_close
from src.vision.canonical.visible_admission import adapt_visible_truth
from scripts.vision.verify_pixel_duplicates import rgb_digest

BASE=ROOT/'data/research/ml_training_recovery_v1/paired-calibration-v1'


def iou(a,b):
    intersect=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda x:max(0,x[2]-x[0])*max(0,x[3]-x[1])
    union=area(a)+area(b)-intersect
    return intersect/union if union else 0


def indexed(objects):
    groups=defaultdict(list)
    for o in objects:
        label=int(re.search(r'-instance-(\d+)-',o['annotation_id'])[1])
        groups[label].append(o)
    return groups


def main():
    inputs={}
    def bind(path):
        path=Path(path);inputs[str(path)]=file_sha256(path);return path
    bind(__file__)
    manifest=read_record(bind(BASE/'plan-manifest.json'))
    captures={};worlds={};plan_views={}
    for run in manifest['runs']:
        plan=read_record(bind(run['plan_path']))
        world=bind(Path(run['plan_path']).parent/'world.sdf')
        if plan['identity']!=run['plan_identity'] or file_sha256(world)!=plan['files']['world.sdf']:
            raise ValueError('Changed planned world')
        tree=ET.parse(world)
        for sensor in tree.iter('sensor'):
            if sensor.get('type')=='boundingbox_camera':
                node=sensor.find('camera/box_type')
                if node.text!=run['mode']:raise ValueError('Box mode mismatch')
                node.text='MODE_REMOVED_FOR_EQUALITY_CHECK'
        worlds[(run['source_plan_identity'],run['mode'])]=ET.tostring(tree.getroot())
        receipt=read_record(bind(BASE/run['name']/'capture/collection-receipt.json'))
        if receipt['plan_identity']!=plan['identity'] or receipt['status']!='complete_pending_review':
            raise ValueError('Calibration capture incomplete or mismatched')
        expected={v['view_id']:v for v in plan['calibration_views']}
        if {v['view_id'] for v in receipt['views']}!=set(expected):raise ValueError('Missing planned view')
        for row in receipt['views']:
            if row['status']!='captured' or not pose_close(row['actual_pose'],expected[row['view_id']]):
                raise ValueError('Invalid captured pose')
            rgb=bind(row['rgb_path']);depth=bind(row['depth_path'])
            if file_sha256(rgb)!=row['image_sha256'] or file_sha256(depth)!=row['depth_sha256']:
                raise ValueError('Changed captured bytes')
            pixels,size=rgb_digest(rgb.read_bytes())
            key=(run['source_plan_identity'],row['view_id'])
            captures.setdefault(key,{})[run['mode']]={**row,'pixel_sha256':pixels,'size':size}
            plan_views[key]=expected[row['view_id']]
    for identity,_ in worlds:
        if worlds[(identity,'visible_2d')]!=worlds[(identity,'full_2d')]:
            raise ValueError('Mode pairs changed more than the box sensor type')
    held={(r['source_plan_identity'],v) for r in manifest['runs'] for v in r['held_target_view_ids']}
    pairs=[];object_pairs=[]
    for (identity,view_id),modes in sorted(captures.items()):
        v,f=modes['visible_2d'],modes['full_2d']
        adapted=adapt_visible_truth(v['truth'],annotation_mode='visible_2d')
        vi,fi=indexed(adapted['objects']),indexed(f['truth']['objects'])
        same_pixels=v['pixel_sha256']==f['pixel_sha256']
        same_pose=pose_close(v['actual_pose'],f['actual_pose'])
        comparisons=[]
        for label in sorted(vi.keys()&fi.keys()):
            if len(vi[label])!=1 or len(fi[label])!=1:continue
            a,b=vi[label][0],fi[label][0]
            if a['class_name']!=b['class_name']:raise ValueError('Paired instance class differs')
            vb,fb=a['bbox_xyxy'],b['bbox_xyxy']
            item={'source_plan_identity':identity,'view_id':view_id,'instance_label':label,
                  'class_name':a['class_name'],'visible_half_open_bbox':vb,'full_bbox':fb,
                  'iou':iou(vb,fb),'max_endpoint_difference_px':max(abs(x-y) for x,y in zip(vb,fb)),
                  'same_rgb_pixels':same_pixels,'same_pose_within_tolerance':same_pose,
                  'held_target_view':(identity,view_id) in held}
            comparisons.append(item);object_pairs.append(item)
        pairs.append({'source_plan_identity':identity,'view_id':view_id,'map_id':v['map_id'],
                      'expected_object_id':v['expected_object_id'],'expected_category':v['expected_category'],
                      'held_target_view':(identity,view_id) in held,'same_rgb_pixels':same_pixels,
                      'same_pose_within_tolerance':same_pose,'visible_object_count':sum(map(len,vi.values())),
                      'full_object_count':sum(map(len,fi.values())),
                      'visible_only_instance_labels':sorted(vi.keys()-fi.keys()),'full_only_instance_labels':sorted(fi.keys()-vi.keys()),
                      'ambiguous_instance_labels':sorted(k for k in vi.keys()|fi.keys() if len(vi.get(k,[]))>1 or len(fi.get(k,[]))>1),
                      'visible_rgb_path':v['rgb_path'],'full_rgb_path':f['rgb_path'],
                      'visible_image_sha256':v['image_sha256'],'full_image_sha256':f['image_sha256'],
                      'object_comparisons':comparisons})
    all_matched=[r for r in object_pairs if r['same_rgb_pixels'] and r['same_pose_within_tolerance']]
    def stats(rows):
        values=[r['iou'] for r in rows]
        return {'matched_objects':len(rows),'iou_min':min(values,default=None),
                'iou_median':statistics.median(values) if values else None,
                'iou_below_0_5':sum(x<.5 for x in values),'iou_below_0_9':sum(x<.9 for x in values),
                'endpoint_difference_gt_1px':sum(r['max_endpoint_difference_px']>1 for r in rows)}
    out=BASE/'analysis';out.mkdir(exist_ok=True)
    files={}
    for name,rows in [('pairs.jsonl',pairs),('object-comparisons.jsonl',object_pairs)]:
        path=out/name;path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows));files[name]=file_sha256(path)
    sheets=ROOT/'outputs/research/ml_training_recovery_v1/paired-calibration'
    sheets.mkdir(parents=True,exist_ok=True)
    # Paired overlays show all labels; control sheets show source-bound intent.
    display=sorted(pairs,key=lambda r:(not r['held_target_view'],min((x['iou'] for x in r['object_comparisons']),default=1)))
    evidence={}
    for start in range(0,len(display),4):
        page=Image.new('RGB',(1280,1600),'#202020');draw=ImageDraw.Draw(page)
        for n,row in enumerate(display[start:start+4]):
            modes=captures[(row['source_plan_identity'],row['view_id'])]
            for col,mode in enumerate(('visible_2d','full_2d')):
                image_row=modes[mode];x,y=col*640,n*400
                with Image.open(image_row['rgb_path']) as im:page.paste(im.resize((640,360)),(x,y+40))
                draw.text((x+4,y+3),f'{start+n+1} {mode} {row["map_id"]} {row["view_id"][:8]}',fill='white')
                draw.text((x+4,y+20),f'Expected: {row["expected_object_id"]} | RGB equal: {row["same_rgb_pixels"]}',fill='white')
                objs=adapt_visible_truth(image_row['truth'],annotation_mode='visible_2d')['objects'] if mode=='visible_2d' else image_row['truth']['objects']
                for o in objs:
                    a,b,c,d=o['bbox_xyxy'];box=(x+a/3,y+40+b/3,x+c/3,y+40+d/3)
                    draw.rectangle(box,outline='#40ffff' if mode=='visible_2d' else '#ffb347',width=2)
                    draw.text((box[0],max(y+40,box[1]-12)),o['class_name'][0].upper(),fill='white')
        path=sheets/f'page-{start//4+1:02}.jpg';page.save(path,quality=95);evidence[str(path)]=file_sha256(path)
    report={'schema_version':1,'status':'paired_capture_comparison_complete','inputs':inputs,'files':files,'visual_evidence':evidence,
            'captured_runs':len(manifest['runs']),'paired_views':len(pairs),'held_target_pairs':sum(r['held_target_view'] for r in pairs),
            'control_pairs':sum(not r['held_target_view'] for r in pairs),
            'same_rgb_pixel_pairs':sum(r['same_rgb_pixels'] for r in pairs),
            'same_pose_pairs':sum(r['same_pose_within_tolerance'] for r in pairs),
            'world_pairs_differ_only_in_box_mode':True,
            'all_same_image_matched_objects':stats(all_matched),
            'held_target_same_image_matched_objects':stats([r for r in all_matched if r['held_target_view']]),
            'pairs_with_instance_inventory_difference':sum(bool(r['visible_only_instance_labels'] or r['full_only_instance_labels']) for r in pairs),
            'training_admitted':False,
            'limits':['Metrics are diagnostic descriptions, not acceptance thresholds.',
                      'Direct annotation-only conclusions apply to equal RGB pixels and matched poses.',
                      'New captures do not certify historical full_2d settings or historical image equality.',
                      'Cabinet/switchgear controls use matching range, height and bearing; target geometry, pitch and background differ.',
                      'No human review acceptance, model confusion measurement, training or taxonomy change is inferred.']}
    report['identity']=object_sha256(report);write_json(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','files','visual_evidence')},indent=2))


if __name__=='__main__':main()
