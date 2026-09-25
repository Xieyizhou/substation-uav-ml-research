"""Bounded new-pose original-appearance data proposal; projection is not review."""
import argparse
from collections import Counter
import copy
import itertools
import json
import math
from pathlib import Path
import shutil

from src.ml.artifacts import file_sha256, object_sha256
from src.vision.canonical.plan import read_record, write_record, pose_close
from src.vision.canonical.expansion import CLASSES, _center, _pose
from src.vision.canonical.gates import validate_view_pose, validate_preflight
from scripts.vision.design_full_scene_poses import screen
from scripts.vision.prepare_reactor_source_isolated import BASE
from scripts.vision.reviewed_negative_order_control import verify

OUT=BASE/'clear-context-data-increment-v1'


def checked(path):
    r=read_record(path);verify(r);return r


def candidates(source, config, seen):
    eligible=[];rejected=Counter()
    for obj in source['objects']:
        if obj['category'] not in CLASSES:continue
        center=_center(obj['bounds'])
        for bearing,distance,height in itertools.product(range(7,360,5),(3.5,5.,7.5,10.),(2.,3.,5.,7.,9.)):
            a=math.radians(bearing);camera=[center[0]+distance*math.cos(a),center[1]+distance*math.sin(a),height]
            position,q=_pose(camera,center,0)
            row=dict(map_id=source['map_id'],object_id=obj['name'],category=obj['category'],bearing=bearing,
                distance=distance,height=height,offset=0,camera_position=camera,position=position,orientation=q,
                family=f'clear-context-v1:{obj["name"]}:{bearing}',condition='original')
            row['view_id']=object_sha256(row)
            try:
                validate_view_pose(row,config)
                if any(pose_close(dict(position=position,orientation=q),s) for s in seen):raise ValueError('existing_pose')
                boxes=screen(row,source['objects'])
                planned=boxes[row['object_id']]
                if min(planned[2]-planned[0],planned[3]-planned[1])/3<48:raise ValueError('planned_target_design_scale')
                if any(min(b[2]-b[0],b[3]-b[1])/3<20 for b in boxes.values()):raise ValueError('incidental_target_design_scale')
                if len(boxes)>5:raise ValueError('too_many_projected_targets_for_small_pilot')
            except ValueError as error:
                rejected[str(error)]+=1;continue
            eligible.append(dict(view=row,projected_targets=boxes))
    return eligible,dict(rejected)


def select(rows, count=3):
    selected=[]
    for category in CLASSES:
        options=sorted((r for r in rows if r['view']['category']==category),key=lambda r:(len(r['projected_targets']),r['view']['height'],r['view']['distance'],r['view']['object_id'],r['view']['bearing']))
        chosen=[]
        while len(chosen)<count:
            options.sort(key=lambda r:(sum(x['view']['object_id']==r['view']['object_id'] for x in chosen),len(r['projected_targets']),r['view']['height'],r['view']['distance'],r['view']['object_id'],r['view']['bearing']))
            found=None
            for row in options:
                v=row['view']
                if any(v['object_id']==p['view']['object_id'] and min(abs(v['bearing']-p['view']['bearing']),360-abs(v['bearing']-p['view']['bearing']))<45 for p in chosen):continue
                found=row;break
            if found is None:raise ValueError('Insufficient distinct legal candidates: '+category)
            chosen.append(found);options.remove(found)
        selected.extend(chosen)
    return selected


def freeze():
    dest=OUT/'proposal.json'
    if dest.exists():return checked(dest)
    source=checked(BASE/'plan.json');config=json.loads((BASE/'obstacles.json').read_text())
    deps=[BASE/'plan.json',BASE/'obstacles.json',Path(__file__)]
    seen=[]
    for phase in ('calibration','pilot'):
        path=BASE/phase/'collection-receipt.json';r=read_record(path)
        if r['plan_identity']!=source['identity']:raise ValueError('Source receipt identity mismatch')
        seen.extend(v['actual_pose'] for v in r['views'] if v['status']=='captured');deps.append(path)
    eligible,rejects=candidates(source,config,seen);selected=select(eligible)
    OUT.mkdir(parents=True,exist_ok=True);folder=OUT/'plan';folder.mkdir(exist_ok=False)
    for name,h in source['files'].items():
        path=BASE/name
        if file_sha256(path)!=h:raise ValueError('Changed source file '+name)
        shutil.copy2(path,folder/name);deps.extend([path,folder/name])
    plan={k:copy.deepcopy(v) for k,v in source.items() if k!='identity'}
    views=[dict(r['view'],review_id=f'C{i+1:02}') for i,r in enumerate(selected)]
    plan.update(calibration_views=views,pilot_views=[],all_selected_views=views,selected_view_ids=[r['view_id'] for r in views],
        selection_rule='3/class; distinct target instances where available, >=45deg bearing separation within instance, deterministic full-scene geometric screen.',
        diagnostic_purpose='original_appearance_clear_context_new_view_training_candidates',
        source_isolation=dict(layout='existing_extreme_development_layout',new_layout=False,shared_asset_families=['canonical primitive equipment geometry'],independent_scene_claim=False),
        diagnostic_only=True,diagnostic_require_expected_presence=True,training_admitted=False,promotable=False,automatic_training=False)
    plan=write_record(folder/'plan.json',plan);gate,_,_=validate_preflight(plan,folder,views);deps.append(folder/'plan.json')
    return write_record(dest,dict(status='twelve_new_pose_candidates_frozen_not_captured',plan_path=str((folder/'plan.json').resolve()),
        selected=selected,eligible_count=len(eligible),rejection_counts=rejects,gate=gate,
        existing_actual_poses_excluded=len(seen),new_independent_asset_count=0,new_layout_count=0,
        policy='Original world, materials, lights and sensor settings unchanged. AABB/center-ray screens are acquisition prioritization only, never labels or visible-content certification. Full-frame and every-label review required. No training before approval; unresolved frame held whole.',
        bounds='12 frames, original condition, four classes x three poses. Each capture at most three technical attempts; semantic failure not retried into acceptance. Training design frozen only after complete review and exposure feasibility checks.',
        training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in deps}))


if __name__=='__main__':
    p=freeze();print(p['status'],p['eligible_count'],[(r['view']['category'],r['view']['object_id'],r['view']['bearing'],len(r['projected_targets'])) for r in p['selected']],flush=True)
