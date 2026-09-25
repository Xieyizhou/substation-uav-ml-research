"""Bounded geometry-only search for smaller known-asset training views."""
import itertools, math
from collections import Counter
import xml.etree.ElementTree as ET
from pathlib import Path
from scripts.vision.source_isolated_material_capture import SOURCE,relocate,prior,COUNTS
from scripts.vision.neutral_gray_capture import OUT as REF
from scripts.vision.design_full_scene_poses import screen
from src.vision.canonical.expansion import _pose
from src.vision.canonical.plan import read_record,object_sha256
from src.vision.canonical.gates import validate_view_pose

OUT=REF/'small-scale-layout-feasibility-v1'


def candidates(objects,cfg,layout):
    eligible=[];rejected=Counter()
    for obj in objects:
        if obj['category'] not in COUNTS:continue
        center=[sum(obj['bounds'][i:i+2])/2 for i in (0,2,4)]
        for bearing,distance,height in itertools.product(range(0,360,15),(10.,14.,18.,22.,26.),(2.5,4.,6.)):
            a=math.radians(bearing);optical=[center[0]+distance*math.cos(a),center[1]+distance*math.sin(a),height]
            position,q=_pose(optical,center,0)
            v=dict(map_id=layout,object_id=obj['name'],category=obj['category'],bearing=bearing,distance=distance,height=height,
                   offset=0,camera_position=optical,position=position,orientation=q,family=layout+':'+obj['name'])
            v['view_id']=object_sha256(v)
            try:
                validate_view_pose(v,cfg);boxes=screen(v,objects)
                box=boxes[obj['name']];short=min(box[2]-box[0],box[3]-box[1])/3
                if not 32<=short<64:raise ValueError('outside_target_scale_bin')
            except ValueError as exc:rejected[str(exc)]+=1;continue
            eligible.append(dict(view=v,projected_targets=boxes,planned_short_side_640=short))
    return eligible,dict(rejected)


def main():
    dest=OUT/'census.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    source=read_record(SOURCE);cfg=prior.read(SOURCE.parent/'obstacles.json');tree=ET.parse(SOURCE.parent/'world.sdf').getroot()
    deps=[SOURCE,REF/'pool-fit-diagnosis-v1/summary.json',REF/'view-coverage-feasibility-v1/census.json',Path(__file__).resolve()]
    for name,digest in source['files'].items():
        path=SOURCE.parent/name
        if prior.file_sha256(path)!=digest:raise ValueError('Source changed')
        deps.append(path)
    attempts=[];selected=[]
    for n in range(1,21):
        layout=f'small-material-layout-v1-{n:02}'
        try:
            objects,config,world,changes=relocate(source,cfg,tree,layout)
            rows,rejections=candidates(objects,config,layout)
        except ValueError as exc:
            attempts.append(dict(layout=layout,error=str(exc)));continue
        counts=Counter(x['view']['category'] for x in rows)
        attempts.append(dict(layout=layout,counts=dict(counts),rejections=rejections,candidates=rows,translations=changes))
        print(layout,dict(counts),flush=True)
        if all(counts[c] for c in COUNTS):
            views=[sorted((x for x in rows if x['view']['category']==c),key=lambda x:(x['view']['height'],x['view']['distance'],x['view']['object_id'],x['view']['bearing']))[0] for c in COUNTS]
            selected.append(dict(layout=layout,views=views,translations=changes))
            if len(selected)==2:break
    OUT.mkdir(exist_ok=True)
    return prior.frozen(dest,dict(status='two_small_scale_layouts_geometrically_feasible' if len(selected)==2 else 'bounded_small_scale_layout_search_incomplete',
        attempts=attempts,selected=selected,capture_started=False,training_started=False,
        interpretation='32-64 px is a descriptive acquisition bin, not a label approval threshold. Center-ray/AABB screening is conservative geometry only. Actual full-instance masks, RGB alignment and per-box review remain mandatory.',
        inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__=='__main__':print(main()['status'])
