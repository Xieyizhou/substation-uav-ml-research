"""Conservative full-scene pose prefilter; projections never become truth labels."""
import argparse,copy,itertools,math,shutil
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scripts.vision.establish_material_view_candidates import OUT as PARENT,HISTORY,prior
from scripts.vision.prepare_visual_augmentation_batch import SOURCE
from src.vision.canonical.expansion import CLASSES,_pose
from src.vision.canonical.plan import rotate,object_sha256,write_record,pose_close
from src.vision.canonical.gates import validate_view_pose,validate_preflight
from src.vision.canonical.occlusion import segment_intersects_bounds

OUT=PARENT/'full-scene-pose-design-v1'
WIDTH,HEIGHT=1920,1080
HFOV=1.466
MARGIN=16


def projected_bounds(bounds,view):
    q=view['orientation'];inverse=[-q[0],-q[1],-q[2],q[3]]
    corners=np.array([rotate(inverse,[c[i]-view['camera_position'][i] for i in range(3)]) for c in itertools.product(*[(bounds[i],bounds[i+1]) for i in (0,2,4)])])
    x,y,z=corners.T;tx=math.tan(HFOV/2);ty=tx*HEIGHT/WIDTH
    planes=np.array([x-.1,120-x,x*tx-y,x*tx+y,x*ty-z,x*ty+z])
    if np.any(np.all(planes<0,axis=1)):return None
    if np.any(x<=.1):raise ValueError('target_crosses_near_plane')
    u=WIDTH/2-y/x*WIDTH/(2*tx);v=HEIGHT/2-z/x*HEIGHT/(2*ty)
    result=[float(u.min()),float(v.min()),float(u.max()),float(v.max())]
    if not (MARGIN<=result[0] and MARGIN<=result[1] and result[2]<=WIDTH-MARGIN and result[3]<=HEIGHT-MARGIN):raise ValueError('target_touches_frame_margin')
    return result


def screen(view,objects):
    visible={}
    for obj in objects:
        if obj['category'] not in CLASSES:continue
        box=projected_bounds(obj['bounds'],view)
        if box is None:continue
        center=[sum(obj['bounds'][i:i+2])/2 for i in (0,2,4)]
        if any(o['name']!=obj['name'] and segment_intersects_bounds(view['camera_position'],center,o['bounds']) for o in objects):raise ValueError('visible_target_center_occluded')
        visible[obj['name']]=box
    if view['object_id'] not in visible:raise ValueError('planned_target_not_projected')
    return visible


def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():p=prior.read(dest);prior.verify(p);return p
    source=prior.read(SOURCE);config=prior.read(SOURCE.parent/'obstacles.json')
    paths=[SOURCE,SOURCE.parent/'obstacles.json',HISTORY/'coverage-census.json',PARENT/'source-inventory.json',PARENT/'fresh-source-probe-v1/reviewed-completion.json',Path(__file__)]
    census=prior.read(paths[2]);inventory=prior.read(paths[3]);holds=prior.read(paths[4])
    for r in (census,inventory,holds):prior.verify(r)
    seen=[m['actual_pose'] for m in census['members'] if m.get('actual_pose')]
    seen += [r['actual_pose'] for r in inventory['records'] if r.get('actual_pose')]
    records=[];eligible=[]
    for obj in source['objects']:
        if obj['category'] not in CLASSES:continue
        center=[sum(obj['bounds'][i:i+2])/2 for i in (0,2,4)]
        for bearing,distance,height in itertools.product(range(0,360,15),(5.,7.5,10.,12.5,15.),(3.,5.,7.,9.)):
            angle=math.radians(bearing);camera=[center[0]+distance*math.cos(angle),center[1]+distance*math.sin(angle),height]
            position,q=_pose(camera,center,0)
            v=dict(map_id='complex',object_id=obj['name'],category=obj['category'],bearing=bearing,distance=distance,height=height,
                offset=0,camera_position=camera,position=position,orientation=q,family=f'full-scene-v1:{obj["name"]}:{bearing}')
            v['view_id']=object_sha256(v);reason=None;projected=None
            try:
                validate_view_pose(v,config)
                if any(pose_close({'position':position,'orientation':q},s) for s in seen):raise ValueError('previous_pose_excluded')
                projected=screen(v,source['objects'])
            except ValueError as e:reason=str(e)
            records.append(dict(view=v,rejection=reason,projected_targets=projected))
            if reason is None:eligible.append(v)
    selected=[];missing=[]
    for category in ('capacitor_bank','switchgear','reactor','transformer'):
        rows=sorted([r for r in eligible if r['category']==category],key=lambda r:(r['distance'],r['height'],r['object_id'],r['bearing'],r['view_id']))
        if rows:selected.append(dict(rows[0],probe_id='G'+str(len(selected)+1).zfill(2)))
        else:missing.append(category)
    OUT.mkdir(exist_ok=True);plan_dir=OUT/'plan';plan_dir.mkdir(exist_ok=True)
    for name,h in source['files'].items():
        src=SOURCE.parent/name
        if prior.file_sha256(src)!=h:raise ValueError('Ancestor hash changed: '+name)
        shutil.copy2(src,plan_dir/name);paths.append(src)
    tree=ET.parse(plan_dir/'world.sdf');boxes=tree.findall('.//sensor[@type="boundingbox_camera"]/camera/box_type')
    if len(boxes)!=1:raise ValueError('Camera configuration mismatch')
    boxes[0].text='full_2d';tree.write(plan_dir/'world.sdf',encoding='utf-8',xml_declaration=True)
    plan={k:copy.deepcopy(v) for k,v in source.items() if k!='identity'}
    plan.update(annotation_mode='full_2d',label_mode='visual-instance',hierarchy_mode='top-level-equipment',calibration_views=selected,pilot_views=[],diagnostic_only=True,diagnostic_require_expected_presence=True,
        training_admitted=False,promotable=False,automatic_training=False)
    plan['files']['world.sdf']=prior.file_sha256(plan_dir/'world.sdf');write_record(plan_dir/'plan.json',plan)
    validate_preflight(plan,plan_dir,selected);paths += list(plan_dir.iterdir())
    return prior.frozen(dest,dict(status='bounded_pose_plan_frozen' if not missing else 'missing_classes_no_expansion',
        candidates=records,selected=selected,missing_classes=missing,rejection_counts=dict(Counter(r['rejection'] for r in records if r['rejection'])),
        geometry_passed=len(eligible),plan_path=str(plan_dir/'plan.json'),
        policy='Four fixed class representatives before real capture; full target AABB inside 16px margin and all in-frustum target center rays unblocked. Margin is a conservative experimental prefilter, not a training quality threshold. Projection is not a mask or label. Heights 3/5/7/9m are offline camera experiments, not UAV flight certification.',
        limitations=['Shared layout/assets','Center ray does not certify complete unocclusion','Historical unresolved poses and old admission-chain limitations remain'],
        training_ready=False,training_started=False,inputs={str(p):prior.file_sha256(p) for p in paths}))

if __name__=='__main__':
    p=freeze();print('GEOMETRY_PASS',p['geometry_passed'],'SELECTED',[(x['probe_id'],x['category'],x['distance'],x['height']) for x in p['selected']],'MISSING',p['missing_classes'])
