"""Freeze a source-isolated extreme-layout reactor coverage capture."""
import copy
import math
import shutil
from pathlib import Path
from src.ml.artifacts import file_sha256, object_sha256
from src.vision.canonical.plan import materialize, read_record, write_record
from src.vision.canonical.expansion import _center, _pose
from src.vision.canonical.occlusion import view_blockers

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/research/ml_training_recovery_v1/reactor-visibility-expansion-v4'
WORLD=BASE/'extreme-world'

def _legal(camera):
    return -15.75 < camera[0] < 15.75 and -15.75 < camera[1] < 15.75 and .25 < camera[2] < 5.75

def _candidates(plan):
    objs=plan['objects']; reactors=[o for o in objs if o.get('category')=='reactor']
    rows=[]
    for target in reactors:
        center=_center(target['bounds'])
        for bearing in range(0,360,5):
            for distance in (6,8,10,12,14,16,18,20):
                for height in (1.5,2.5,4.0):
                    a=math.radians(bearing); camera=[center[0]+distance*math.cos(a),center[1]+distance*math.sin(a),height]
                    if not _legal(camera):continue
                    for offset in (-30,-15,0,15,30):
                        position,orientation=_pose(camera,center,offset)
                        row=dict(map_id='extreme',object_id=target['name'],category='reactor',bearing=bearing,
                            distance=distance,height=height,offset=offset,camera_position=camera,position=position,orientation=orientation,
                            family=f"extreme:{target['name']}:{bearing}")
                        row['view_id']=object_sha256(row)
                        blocked=view_blockers(row,objs)
                        if offset==0 and not blocked and distance<=8: condition='clear_near'
                        elif offset==0 and not blocked and distance>=16: condition='clear_far'
                        elif offset==0 and blocked: condition='occluded'
                        elif offset in (-30,30) and not blocked: condition='edge_candidate'
                        else:continue
                        row['condition']=condition; rows.append(row)
    return rows

def _select(rows,condition,count):
    values=sorted((r for r in rows if r['condition']==condition),key=lambda r:(r['object_id'],r['distance'],r['height'],abs(r['offset']),r['bearing'],r['view_id']))
    # Round-robin target identities to prevent one reactor dominating a condition.
    by={name:[r for r in values if r['object_id']==name] for name in sorted({r['object_id'] for r in values})}
    out=[]
    while len(out)<count:
        progressed=False
        for name in sorted(by):
            if by[name]:out.append(by[name].pop(0));progressed=True
            if len(out)==count:break
        if not progressed:raise ValueError(f'Not enough {condition} candidates')
    return out

def run():
    path=BASE/'plan.json'
    if path.exists():return read_record(path)
    BASE.mkdir(parents=True,exist_ok=False)
    p=materialize('extreme',WORLD,label_mode='visual-instance',hierarchy_mode='top-level-equipment',instance_collision_strategy='linear-probe')
    if not p['instance_collision_resolutions']:
        raise ValueError('Expected explicit collision accounting')
    # The canonical collector resolves plan files relative to plan.json.  Keep
    # every named artifact beside the plan; do not encode an alternate path in
    # the historical plan schema.
    for name in ('source_world.sdf','obstacles.json','sensor_source.sdf','label_mapping.py','world.sdf'):
        shutil.copy2(WORLD/name,BASE/name)
    p['files']={name:file_sha256(BASE/name) for name in ('source_world.sdf','obstacles.json','sensor_source.sdf','label_mapping.py','world.sdf')}
    candidates=_candidates(p)
    selected=[]
    for condition in ('clear_near','occluded','edge_candidate','clear_far'):
        selected.extend(_select(candidates,condition,3))
    calibration=[]
    for condition in ('clear_near','occluded','edge_candidate','clear_far'):
        calibration.append(next(x for x in selected if x['condition']==condition))
    selected_ids={x['view_id'] for x in selected};pilot=[x for x in selected]
    record={k:copy.deepcopy(v) for k,v in p.items() if k!='identity'}
    record.update(schema_version=1,annotation_mode='full_2d',source_isolation=dict(layout='substation_extreme_v1',new_layout=True,
        shared_asset_families=['canonical primitive equipment geometry'],independent_scene_claim='layout-level only'),
        calibration_views=calibration,pilot_views=pilot,all_selected_views=selected,
        candidate_counts={c:sum(x['condition']==c for x in candidates) for c in ('clear_near','occluded','edge_candidate','clear_far')},
        selection_rule='3 per condition, deterministic round-robin reactor identity; calibration covers each condition once.',
        diagnostic_only=True,diagnostic_purpose='reactor_occlusion_truncation_transfer',
        training_admitted=False,automatic_training=False,promotable=False,
        pose_tolerance_m=.05,attitude_tolerance_deg=1,max_skew_ms=33.334,stable_frames=3,
        condition_interpretation='edge_candidate requires post-capture review; camera offset alone is not proof of truncation.',
        selected_view_ids=sorted(selected_ids))
    # materialize() already supplied hashes and the canonical unique instance map.
    deps=[BASE/name for name in ('world.sdf','source_world.sdf','obstacles.json','sensor_source.sdf','label_mapping.py')]+[Path(__file__).resolve()]
    record['inputs']={str(x):file_sha256(x) for x in deps}
    return write_record(path,record)

if __name__=='__main__':
    p=run();print('FROZEN',p['identity'],len(p['calibration_views']),len(p['pilot_views']),p['candidate_counts'])
