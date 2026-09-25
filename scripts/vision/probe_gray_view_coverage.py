"""Read-only geometric feasibility census; projection never approves a label."""
import itertools,math
from collections import Counter
from pathlib import Path
from scripts.vision.neutral_gray_capture import OUT,freeze,prior
from scripts.vision.design_full_scene_poses import screen
from src.vision.canonical.expansion import _pose
from src.vision.canonical.gates import validate_view_pose
from src.vision.canonical.plan import object_sha256


def main():
    p=freeze();records=[]
    for u in p['units']:
        if u['illumination_condition']!='normal':continue
        source=prior.read(u['plan_path']);cfg=prior.read(Path(u['plan_path']).parent/'obstacles.json')
        original=u['view'];obj=next(x for x in source['objects'] if x['name']==original['object_id'])
        center=[sum(obj['bounds'][i:i+2])/2 for i in (0,2,4)]
        e=prior.read(OUT/'evidence'/u['unit_id']/'evidence.json');expected={x['object_id'] for x in e['events']}
        failures=Counter();eligible=[]
        for delta,distance,height in itertools.product(range(0,360,15),(6.,8.,10.,12.,14.,16.,20.,24.),(2.,3.,4.,5.)):
            if distance<=original['distance']:continue
            if (height-center[2])/distance >= (original['height']-center[2])/original['distance']:continue
            bearing=(original['bearing']+delta)%360;angle=math.radians(bearing)
            optical=[center[0]+distance*math.cos(angle),center[1]+distance*math.sin(angle),height]
            position,q=_pose(optical,center,0)
            view=dict(original,camera_position=optical,position=position,orientation=q,distance=distance,height=height,bearing=bearing)
            view.pop('view_id',None);view['view_id']=object_sha256(view)
            try:
                validate_view_pose(view,cfg)
                visible=screen(view,source['objects'])
                if set(visible)!=expected:raise ValueError('full_instance_set_changed')
            except ValueError as exc:failures[str(exc)]+=1;continue
            box=visible[original['object_id']];short=min(box[2]-box[0],box[3]-box[1])/3
            eligible.append(dict(view=view,projected_targets=visible,planned_short_side_640=short))
        records.append(dict(unit_id=u['unit_id'],pair_id=u['source_member']['pair_id'],expected=sorted(expected),eligible=eligible,rejections=dict(failures)))
        print(u['unit_id'],original['category'],'eligible',len(eligible),'min_short',min((x['planned_short_side_640'] for x in eligible),default=None),'reasons',dict(failures),flush=True)
    return records


if __name__=='__main__':main()
