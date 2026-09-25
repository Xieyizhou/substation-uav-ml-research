"""Bounded local pose supplement, explicitly grouped with its parent viewpoint."""
import asyncio
import math
from pathlib import Path
from scripts.vision.supplement_full_image_poses import OUT as PRIOR,read,save,file_sha256,verify_tree,SOURCE,CONFIG,screen,near
from src.vision.canonical.expansion import _pose
from src.vision.canonical.gates import validate_view_pose
from src.ml.artifacts import object_sha256
import scripts.vision.capture_full_image_pose_pilot as capture

OUT=PRIOR/'local-supplement-v1'

def prepare():
    path=OUT/'screen.json'
    if path.exists():verify_tree(path);return
    verify_tree(PRIOR/'screen.json');prior=read(PRIOR/'screen.json');selected=prior['selected'];base=next(x['view'] for x in selected if x['view']['category']=='switchgear')
    source=read(SOURCE);cfg=read(CONFIG);obj=next(o for o in source['objects'] if o['name']==base['object_id']);b=obj['bounds'];center=[(b[i]+b[i+1])/2 for i in (0,2,4)]
    known=[x['view'] for x in selected]
    for p in prior['inputs']:
        if Path(p).name=='plan.json':
            plan=read(p);known.extend(r for r in plan.get('calibration_views',[])+plan.get('pilot_views',[]) if r.get('map_id')=='complex' and 'position' in r and 'orientation' in r)
    candidates=[]
    for dx,dy in ((.75,0),(-.75,0),(0,.75),(0,-.75),(1.5,0),(-1.5,0),(0,1.5),(0,-1.5)):
        for dz in (0,.5,-.5):
            cam=[a+b for a,b in zip(base['camera_position'],(dx,dy,dz))];pos,q=_pose(cam,center,base['offset'])
            v={**base,'camera_position':cam,'position':pos,'orientation':q,'height':cam[2],'distance':math.hypot(cam[0]-center[0],cam[1]-center[1]),'bearing':math.degrees(math.atan2(cam[1]-center[1],cam[0]-center[0]))%360,'derived_from_view_id':base['view_id'],'shared_source_group':base['family']}
            v.pop('view_id');v['view_id']=object_sha256(v)
            reason=None
            try:validate_view_pose(v,cfg)
            except ValueError as e:reason=str(e)
            if not reason and any(near(v,x) for x in known):reason='near_existing_pose'
            result=screen(v,source['objects']) if not reason else {'risks':[['pose',reason]],'projected_instances':[]}
            candidates.append(dict(view=v,offset=[dx,dy,dz],**result))
    passed=[r for r in candidates if not r['risks']]
    if not passed:raise ValueError('Bounded local supplement exhausted; no capture')
    selected=selected+[passed[0]]
    if len(selected)!=8:raise ValueError('Unexpected full pilot size')
    OUT.mkdir(parents=True,exist_ok=True)
    save(path,dict(status='geometric_candidates_ready_for_original_only_pilot',selected=selected,local_candidates=candidates,selection='first passing of 24 fixed offsets; no model scores used',lineage_caveat='The two switchgear poses share equipment, parent viewpoint and family; they are not independent scene or asset evidence.',inputs={str(p):file_sha256(p) for p in (PRIOR/'screen.json',Path(__file__))}))
    print('LOCAL_SUPPLEMENT_FROZEN',len(passed),'OF 24; parent family retained',flush=True)

if __name__=='__main__':
    prepare();capture.SCREEN=OUT;capture.OUT=OUT/'original-pilot-v1';asyncio.run(capture.main())
