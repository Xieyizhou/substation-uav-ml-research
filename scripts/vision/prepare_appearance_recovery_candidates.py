"""Freeze geometric candidates only; no capture, visibility certification or admission."""
import math
from pathlib import Path
from collections import Counter
from scripts.vision.run_lower_rate_control import OUT as PRIOR,read,save,file_sha256,verify_tree,ROOT,checked_runtime
from scripts.vision.prepare_visual_augmentation_batch import SOURCE,CONFIG
from src.vision.canonical.plan import read_record
from src.vision.canonical.expansion import target_candidates
from src.vision.canonical.gates import validate_view_pose

OUT=PRIOR/'appearance-recovery-pilot-v1'
ROUND=401

def near(a,b):
    if math.dist(a['position'],b['position'])>=.5:return False
    qa,qb=a['orientation'],b['orientation'];na=math.sqrt(sum(x*x for x in qa));nb=math.sqrt(sum(x*x for x in qb))
    if not na or not nb:raise ValueError('Invalid pose orientation')
    dot=min(1.,abs(sum(x*y for x,y in zip(qa,qb))/(na*nb)))
    return math.degrees(2*math.acos(dot))<5

def main():
    target=OUT/'candidate-matrix.json'
    if target.exists():verify_tree(target);print('VERIFIED_EXISTING');return
    verify_tree(PRIOR/'completion.json');protocol=read(PRIOR/'protocol.json')
    for seed in (7,17,27):checked_runtime(f'J-300-{seed}',protocol)
    source=read_record(SOURCE);config=read(CONFIG);known=[];inputs={}
    # Only canonical/development plan metadata; do not traverse protected or sealed test directories.
    for root in (ROOT/'data/research/canonical_views_v1',ROOT/'data/research/ml_training_recovery_v1'):
        for path in sorted(root.rglob('plan.json')):
            if any(token in str(path).lower() for token in ('protected','sealed','unseen','new-scene','new_scene')):continue
            record=read(path)
            rows=record.get('calibration_views',[])+record.get('pilot_views',[])
            usable=[r for r in rows if r.get('map_id')=='complex' and 'position' in r and 'orientation' in r]
            if usable:known.extend(usable);inputs[str(path)]=file_sha256(path)
    selected=[];rejected=Counter();counts={}
    candidates=target_candidates('complex',source['objects'],config['gazebo_world_origin_m'][:2],(config['width'],config['height']),ROUND)
    for category in ('capacitor_bank','switchgear','reactor','transformer'):
        pool=[]
        for row in candidates:
            if row['category']!=category:continue
            if not 7<=row['distance']<=13:rejected['outside_prespecified_distance']+=1;continue
            try:validate_view_pose(row,config)
            except ValueError:rejected['illegal_pose']+=1;continue
            if any(near(row,x) for x in known):rejected['near_prior_pose']+=1;continue
            pool.append(row)
        pool.sort(key=lambda r:(abs(r['distance']-10),r['object_id'],r['bearing'],r['height'],r['view_id']))
        counts[category]=len(pool);chosen=[]
        for row in pool:
            if any(near(row,x) for x in selected+chosen):continue
            chosen.append(row)
            if len(chosen)==1:break
        if len(chosen)!=1:raise ValueError(f'Need one novel legal pose for {category}; found {len(chosen)}')
        selected.extend(chosen)
    for p in (PRIOR/'completion.json',PRIOR/'protocol.json',SOURCE,CONFIG,Path(__file__),ROOT/'tests/test_appearance_recovery_candidates.py'):inputs[str(p)]=file_sha256(p)
    materials={'steel':'0.22 0.26 0.30 1','ochre':'0.44 0.38 0.28 1','sage':'0.30 0.42 0.36 1'}
    lights={'original':None,'warm_dim':{'ambient':'0.25 0.22 0.18 1','sun_diffuse':'0.55 0.48 0.38 1'}}
    OUT.mkdir(parents=True,exist_ok=True)
    save(target,dict(status='candidates_frozen_not_capture_ready',round=ROUND,views=selected,known_pose_rows=len(known),candidate_counts=counts,rejected=dict(rejected),materials=materials,lights=lights,frame_count=28,pose_groups=4,pilot=dict(frame_count=12,pose_ids=[r['view_id'] for r in selected],variants=['original','steel_original','steel_warm_dim']),full_variants=['original']+[f'{m}_{l}' for m in materials for l in lights],preflight_gap='Initial 8-pose design required two different equipment instances per class; only one capacitor instance had eligible novel poses. Reduced to one pose per class rather than relaxing legal/novelty bounds. Initial selection failed before any world generation or capture.',world_mutation_rule='All four target classes reuse the same selected palette per variant on body/front_panel/reactor only; geometry, bases, attachments, background and sensor parameters unchanged. Light variants only change scene ambient and sun diffuse.',novelty='Reject carrier poses within 0.5 m AND 5 degrees of registered development plans. This is a deduplication heuristic, not independent layout/asset evidence or a safety clearance.',lineage='All variants of each pose remain together. Same complex layout and equipment assets; no new-scene/generalization claim.',next_required=['materialize worlds with structural allowlist check and actual full_2d verification','verify saved poses, sensor and unique instance mappings','capture twelve pilot frames with <=3 technical attempts each','full-image and target AI review; uncertain content remains held','instance replay if identity or visible content uncertain','pixel/lineage dedup and protected-membership exclusion without reading protected labels','only after pilot gate passes capture remaining 16'],inputs=inputs))
    print('CANDIDATES_FROZEN',len(selected),counts,OUT,flush=True)

if __name__=='__main__':main()
