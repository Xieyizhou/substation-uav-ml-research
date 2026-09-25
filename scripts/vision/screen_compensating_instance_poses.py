"""Independent bounded geometric screening; projected boxes are never labels."""
import itertools,math
from collections import Counter
from pathlib import Path
from scripts.vision.design_full_scene_poses import prior,SOURCE,screen,validate_view_pose,_pose,object_sha256
from scripts.vision.merge_material_pose_candidates import OUT as CANDIDATES

OUT=CANDIDATES/'compensating-pose-screen-v1'
HEIGHTS=(13.,16.,20.,24.)
DISTANCES=(8.,12.,16.,20.)
BEARINGS=tuple(range(0,360,15))


def run():
    source=prior.read(SOURCE);config=prior.read(SOURCE.parent/'obstacles.json')
    proof=CANDIDATES/'budget-conflict-proof.json';prior.verify(prior.read(proof))
    objects=source['objects'];lookup={o['name']:o['category'] for o in objects}
    out=[];reasons=Counter();patterns=Counter();total=0
    for target in objects:
        if target['category'] not in ('reactor','capacitor_bank','switchgear','transformer'):continue
        center=[sum(target['bounds'][i:i+2])/2 for i in (0,2,4)]
        for h,d,b in itertools.product(HEIGHTS,DISTANCES,BEARINGS):
            total+=1;a=math.radians(b);camera=[center[0]+d*math.cos(a),center[1]+d*math.sin(a),h]
            pos,q=_pose(camera,center,0)
            view=dict(map_id='complex',object_id=target['name'],category=target['category'],height=h,distance=d,bearing=b,
                camera_position=camera,position=pos,orientation=q,offset=0,family='compensating-high-view-v1')
            view['view_id']=object_sha256(view)
            try:
                validate_view_pose(view,config);projected=screen(view,objects)
            except ValueError as ex:reasons[str(ex)]+=1;continue
            counts=Counter(lookup[x] for x in projected);patterns[str(tuple(counts.get(c,0) for c in ('transformer','switchgear','capacitor_bank','reactor')))]+=1
            if counts['capacitor_bank']+counts['reactor']>1:
                out.append(dict(view=view,projected_targets=projected,projected_counts=dict(counts)))
    out.sort(key=lambda x:(x['view']['height'],x['view']['distance'],x['view']['object_id'],x['view']['bearing'],x['view']['view_id']))
    OUT.mkdir(exist_ok=True)
    paths=[SOURCE,SOURCE.parent/'obstacles.json',proof,Path(__file__),prior.ROOT/'scripts/vision/design_full_scene_poses.py']
    prior.frozen(OUT/'result.json',dict(status='geometric_compensation_candidates_found' if out else 'no_compensating_pose_in_bounded_grid',
        grid=dict(heights=HEIGHTS,distances=DISTANCES,bearings=BEARINGS),total_candidates=total,
        rejection_counts=dict(reasons),passed_projected_patterns=dict(patterns),compensating_candidates=out,
        selected_for_capture=None,training_ready=False,training_started=False,capture_started=False,
        limitations=['Higher offline camera views, not UAV flight safety certification',
        'AABB projections and center-ray checks are planning evidence only, not complete masks or labels',
        'New data would require separately frozen source checks, real collection and explicit content review',
        'Existing 36 candidates and frozen budgets unchanged'],
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('SCREENED',total,'COMPENSATING',len(out),'PATTERNS',dict(patterns),flush=True)
    if out:print('FIRST',out[0]['view'],out[0]['projected_counts'],flush=True)


if __name__=='__main__':run()
