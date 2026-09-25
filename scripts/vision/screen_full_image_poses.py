"""Conservative full-image geometric screening; not instance truth or admission."""
import math
import itertools
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from scripts.vision.replay_appearance_recovery import OUT as PRIOR,read,save,file_sha256,verify_tree,ROOT
from scripts.vision.prepare_visual_augmentation_batch import SOURCE,CONFIG
from scripts.vision.prepare_appearance_recovery_candidates import near
from src.vision.canonical.plan import read_record,rotate
from src.vision.canonical.expansion import target_candidates,CLASSES
from src.vision.canonical.gates import validate_view_pose
from src.vision.canonical.occlusion import view_blockers

OUT=PRIOR/'full-image-pose-screen-v1'

def screen(view,objects,hfov=1.466,width=1920,height=1080):
    q=view['orientation'];inverse=[-q[0],-q[1],-q[2],q[3]];origin=view['camera_position'];fx=width/(2*math.tan(hfov/2));risks=[];in_frame=[]
    for obj in objects:
        if obj['category'] not in CLASSES:continue
        b=obj['bounds'];corners=[rotate(inverse,[p[i]-origin[i] for i in range(3)]) for p in itertools.product(*[(b[i],b[i+1]) for i in (0,2,4)])]
        if max(p[0] for p in corners)<=.1:continue
        if min(p[0] for p in corners)<=.1:risks.append((obj['name'],'near_plane_crossing'));continue
        xs=[width/2-fx*p[1]/p[0] for p in corners];ys=[height/2-fx*p[2]/p[0] for p in corners]
        if max(xs)<=0 or min(xs)>=width or max(ys)<=0 or min(ys)>=height:continue
        in_frame.append(obj['name'])
        if min(xs)<0 or max(xs)>width or min(ys)<0 or max(ys)>height:risks.append((obj['name'],'projected_bounds_truncated'))
        blocked=view_blockers({**view,'object_id':obj['name']},objects)
        if blocked:risks.append((obj['name'],'center_ray_blocked'))
    if view['object_id'] not in in_frame:risks.append((view['object_id'],'planned_projection_missing'))
    return dict(risks=risks,projected_instances=in_frame)

def main():
    path=OUT/'screen.json'
    if path.exists():verify_tree(path);print('VERIFIED_EXISTING');return
    verify_tree(PRIOR/'handoff.json');source=read_record(SOURCE);config=read(CONFIG);known=[];inputs={}
    for root in (ROOT/'data/research/canonical_views_v1',ROOT/'data/research/ml_training_recovery_v1'):
        for p in sorted(root.rglob('plan.json')):
            if any(token in str(p).lower() for token in ('protected','sealed','unseen','new-scene','new_scene')):continue
            record=read(p);rows=record.get('calibration_views',[])+record.get('pilot_views',[])
            useful=[r for r in rows if r.get('map_id')=='complex' and 'position' in r and 'orientation' in r]
            if useful:known.extend(useful);inputs[str(p)]=file_sha256(p)
    camera=ET.parse(SOURCE.parent/'world.sdf').find('.//sensor[@type="boundingbox_camera"]/camera');fov=float(camera.findtext('horizontal_fov'))
    rows=[];seen=set();counts=Counter()
    for round_id in (402,403,404):
        for v in target_candidates('complex',source['objects'],config['gazebo_world_origin_m'][:2],(config['width'],config['height']),round_id):
            if v['view_id'] in seen:continue
            seen.add(v['view_id'])
            try:validate_view_pose(v,config)
            except ValueError:counts['illegal']+=1;continue
            if any(near(v,x) for x in known):counts['near_prior_pose']+=1;continue
            result=screen(v,source['objects'],fov)
            rows.append(dict(view=v,round=round_id,**result))
    selected=[]
    for category in ('capacitor_bank','switchgear','reactor','transformer'):
        options=sorted((r for r in rows if r['view']['category']==category and not r['risks']),key=lambda r:(abs(r['view']['distance']-10),r['view']['object_id'],r['view']['bearing'],r['view']['view_id']))
        chosen=[]
        for r in options:
            if any(near(r['view'],x['view']) for x in selected+chosen):continue
            chosen.append(r)
            if len(chosen)==2:break
        selected.extend(chosen);counts['selected:'+category]=len(chosen)
    for p in (PRIOR/'handoff.json',SOURCE,CONFIG,SOURCE.parent/'world.sdf',Path(__file__),ROOT/'tests/test_full_image_pose_screen.py'):inputs[str(p)]=file_sha256(p)
    OUT.mkdir(parents=True,exist_ok=True)
    save(path,dict(status='geometric_candidates_ready_for_original_only_pilot' if len(selected)==8 else 'bounded_screen_coverage_gap',rows=rows,selected=selected,counts=dict(counts),rounds=[402,403,404],scope='At most two novel poses per class; raw originals first. No appearance expansion before all full-image labels are visually reviewed. AABB frustum and center-ray checks are conservative prefilters, not masks or proof of visibility. A zero-candidate result does not prove no physically feasible view exists.',inputs=inputs))
    print('SCREEN_RESULT',len(rows),dict(counts),'SELECTED',len(selected),flush=True)

if __name__=='__main__':main()
