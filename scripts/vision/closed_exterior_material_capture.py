"""Closed-asset exterior coverage continuation; no automatic admission/training."""
import argparse
import asyncio
import copy
import fcntl
import itertools
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from scripts.vision import source_isolated_material_capture as old
from scripts.vision.design_full_scene_poses import screen
from src.vision.canonical.expansion import _pose
from src.vision.canonical.plan import read_record,write_record,object_sha256,pose_close
from src.vision.canonical.gates import validate_view_pose,validate_preflight
from src.vision.canonical.collect import collect

prior=old.prior
OUT=old.OUT/'closed-exterior-coverage-v1'


def candidates(plan,cfg,layout,previous):
    result=[];rejected={}
    for obj in plan['objects']:
        if obj['category'] not in old.COUNTS:continue
        b=obj['bounds'];center=[(b[i]+b[i+1])/2 for i in (0,2,4)]
        for bearing,distance,height in itertools.product(range(0,360,15),(3.,4.,5.,6.,7.5,9.,10.,12.5),(2.5,3.,4.,5.,7.,9.)):
            angle=math.radians(bearing);cam=[center[0]+distance*math.cos(angle),center[1]+distance*math.sin(angle),height]
            pos,q=_pose(cam,center,0)
            v=dict(map_id=layout,object_id=obj['name'],category=obj['category'],bearing=bearing,distance=distance,height=height,
                offset=0,camera_position=cam,position=pos,orientation=q,family='closed-exterior-v1:'+layout+':'+obj['name'])
            v['view_id']=object_sha256(v)
            try:
                validate_view_pose(v,cfg)
                if any(pose_close(v,p) for p in previous):raise ValueError('previous_frozen_pose_excluded')
                projected=screen(v,plan['objects'])
            except ValueError as ex:
                rejected[str(ex)]=rejected.get(str(ex),0)+1;continue
            result.append(dict(view=v,projected_objects=sorted(projected)))
    return result,rejected


def select(rows,layout):
    chosen=[]
    for category,count in old.COUNTS.items():
        options=[r['view'] for r in rows if r['view']['category']==category]
        # Front-panel normals are -Y on these unchanged assets. Select one frontal and one oblique front.
        if category=='switchgear':
            buckets=[[v for v in options if v['bearing'] in (255,270,285)],
                     [v for v in options if v['bearing'] in (210,225,240,300,315,330)]]
        else:
            distances=sorted({v['distance'] for v in options})
            if len(distances)<count:raise ValueError('distance_coverage_infeasible:'+layout+':'+category)
            if count==1:targets=[distances[len(distances)//2]]
            elif count==2:targets=[distances[0],distances[-1]]
            else:targets=[distances[0],distances[len(distances)//2],distances[-1]]
            buckets=[[v for v in options if v['distance']==d] for d in targets]
        for index,bucket in enumerate(buckets):
            bucket=sorted(bucket,key=lambda v:(v['height'],v['distance'],v['object_id'],v['bearing'],v['view_id']))
            if not bucket:raise ValueError('exterior_orientation_infeasible:'+layout+':'+category+':'+str(index))
            v=copy.deepcopy(bucket[0]);v.update(pair_id=layout+':closed:'+category+':'+str(index+1),
                coverage_goal='front_panel' if category=='switchgear' and index==0 else 'oblique_front' if category=='switchgear' else 'closed_body_scale_and_base',
                data_role='new_training_candidate')
            chosen.append(v)
    return chosen


def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():p=prior.read(dest);prior.verify(p);return p
    pold=old.freeze();review=prior.read(old.OUT/'reviewed-pilot-completion.json');prior.verify(review)
    layouts=[];prepared=[];paths=[old.OUT/'protocol.json',old.OUT/'reviewed-pilot-completion.json',Path(__file__)]
    for layout in ('layout-A','layout-B'):
        ancestor=old.OUT/'plans'/layout/'original';plan=read_record(ancestor/'plan.json')
        cfg=json.loads((ancestor/'obstacles.json').read_text())
        previous=next(l['selected'] for l in pold['layouts'] if l['layout_id']==layout)
        rows,rejections=candidates(plan,cfg,layout,previous);replacement=None;failures=[];changes=[]
        try:views=select(rows,layout)
        except ValueError as ex:
            failures.append(dict(attempt=0,reason=str(ex)))
            ancestor_plan=read_record(old.SOURCE);ancestor_cfg=json.loads((old.SOURCE.parent/'obstacles.json').read_text())
            ancestor_tree=ET.parse(old.SOURCE.parent/'world.sdf').getroot()
            for attempt in range(1,21):
                try:
                    objects,cfg,replacement,changes=old.relocate(ancestor_plan,ancestor_cfg,ancestor_tree,'closed-exterior-v1:'+layout+':'+str(attempt))
                    plan=dict(plan,objects=objects)
                    rows,rejections=candidates(plan,cfg,layout,previous);views=select(rows,layout)
                    replacement.find('.//sensor[@type="boundingbox_camera"]/camera/box_type').text='full_2d'
                    break
                except ValueError as err:
                    failures.append(dict(attempt=attempt,reason=str(err)))
                    print('GEOMETRY_REJECT',layout,attempt,str(err),flush=True)
            else:raise ValueError('bounded_closed_layout_search_exhausted:'+layout)
        layouts.append(dict(layout_id=layout,selected=views,geometric_candidates=rows,rejections=rejections,
                            rejected_layouts=failures,new_layout_translations=changes,prior_layout_reused=replacement is None))
        prepared.append((layout,views,replacement,cfg,plan['objects']))
    OUT.mkdir(parents=True,exist_ok=True);runs=[];units=[]
    for layout,views,replacement,cfg,objects in prepared:
        for variant in old.PALETTES:
            source=old.OUT/'plans'/layout/variant;plan=read_record(source/'plan.json')
            folder=OUT/'plans'/layout/variant;folder.mkdir(parents=True,exist_ok=False)
            for name,h in plan['files'].items():
                if prior.file_sha256(source/name)!=h:raise ValueError('source_changed')
                shutil.copy2(source/name,folder/name);paths.append(source/name)
            if replacement is not None:
                world=copy.deepcopy(replacement);color=old.PALETTES[variant]
                names={o['name'] for o in objects if o['category'] in old.COUNTS}
                if color:
                    for n in old.material_nodes(world,names).values():n.text=color
                old.check_only_materials(replacement,world,names)
                ET.ElementTree(world).write(folder/'world.sdf',encoding='utf-8',xml_declaration=True)
                (folder/'obstacles.json').write_text(json.dumps(cfg,indent=2))
            derived={k:copy.deepcopy(v) for k,v in plan.items() if k!='identity'}
            derived.update(objects=objects,calibration_views=views,pilot_views=[],diagnostic_purpose='closed_exterior_coverage',
                data_role='new_training_candidate',training_admitted=False,promotable=False)
            derived['files']={name:prior.file_sha256(folder/name) for name in plan['files']}
            record=write_record(folder/'plan.json',derived);validate_preflight(record,folder,views)
            paths.extend(folder.iterdir());key=layout+':'+variant
            runs.append(dict(key=key,plan_path=str(folder/'plan.json'),layout_id=layout,variant=variant))
            for v in views:units.append(dict(key=v['pair_id']+':'+variant,run_key=key,pair_id=v['pair_id'],variant=variant,
                plan_path=str(folder/'plan.json'),view_id=v['view_id'],category=v['category'],object_id=v['object_id']))
    return prior.frozen(dest,dict(status='closed_exterior_64_frozen_not_reviewed',runs=runs,units=units,layouts=layouts,
        authorization='User requested supplementation and continuous progression until training begins, conditional on all quality and fixed-budget gates.',
        scope='Closed assets unchanged. Cover exterior proportions/base, panel-facing views and scales; no exterior capacitor-cylinder claim.',
        training_started=False,training_ready=False,training_admitted=False,promotable=False,
        training_budget=dict(total=2700,preserved_original=2184,preserved_negative=486,replaceable_variant_slots=30,
            all_64_members_exposure_not_promised=True),
        inputs={str(p):prior.file_sha256(p) for p in paths}))


async def run():
    p=freeze();dest=OUT/'capture-completion.json'
    if dest.exists():prior.verify(prior.read(dest));return
    with (OUT/'capture.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);results=[];paths=[OUT/'protocol.json']
        for run in p['runs']:
            folder=OUT/'captures'/run['key'];rp=folder/'collection-receipt.json'
            if rp.exists():r=read_record(rp)
            elif folder.exists():raise ValueError('incomplete_capture_retained:'+str(folder))
            else:
                print('CAPTURE_RUN',run['key'],flush=True)
                r=await collect(run['plan_path'],folder,mode='calibration')
            paths.append(rp)
            rows=r['views'];ok=r['status']=='complete_pending_review' and len(rows)==8 and all(x['status']=='captured' for x in rows)
            for row in rows:
                for name in ('rgb_path','depth_path'):
                    if name in row:paths.append(Path(row[name]))
            results.append(dict(run_key=run['key'],receipt=str(rp),captured=sum(x['status']=='captured' for x in rows),
                status='captured_review_pending' if ok else 'blocked',error=r.get('error')))
            print('RUN_RESULT',run['key'],results[-1]['captured'],results[-1]['status'],flush=True)
            if not ok:break
        return prior.frozen(dest,dict(status='64_captured_review_required' if len(results)==8 and all(r['status']=='captured_review_pending' for r in results) else 'capture_blocked',
            runs=results,training_started=False,training_ready=False,inputs={str(p):prior.file_sha256(p) for p in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--capture',action='store_true');a=ap.parse_args()
    if a.capture:asyncio.run(run())
    else:print(freeze()['status'],'NO_CAPTURE_NO_TRAINING')
