"""Two-condition expansion after explicit original review; never starts training."""
import asyncio
import copy
import fcntl
import xml.etree.ElementTree as ET
from pathlib import Path
from scripts.vision.build_full_image_pose_review import OUT as ORIGINAL,read,save,file_sha256,verify_tree
from scripts.vision.capture_appearance_recovery_pilot import transform as pilot_transform,assert_world as pilot_assert
from src.vision.canonical.plan import read_record,write_record
from src.vision.canonical.gates import validate_preflight
from src.vision.canonical.recovery import resumed_views
from src.vision.canonical.collect import collect
from src.ml.artifacts import object_sha256

PRIOR=ORIGINAL/'appearance-expansion-v1'
OUT=ORIGINAL/'appearance-remaining-v1'
MATRIX={'materials':{'ochre':'0.44 0.38 0.28 1','sage':'0.30 0.42 0.36 1'},'lights':{'warm_dim':{'ambient':'0.25 0.22 0.18 1','sun_diffuse':'0.55 0.48 0.38 1'}}}

VARIANTS=('ochre_original','ochre_warm_dim','sage_original','sage_warm_dim')

def translate(variant,matrix):
    if variant not in VARIANTS:raise ValueError('Unknown remaining variant')
    material,light=variant.split('_',1)
    return 'steel_'+light,{'materials':{'steel':matrix['materials'][material]},'lights':matrix['lights']}

def transform(root,variant,matrix,objects):
    name,adapted=translate(variant,matrix)
    return pilot_transform(root,name,adapted,objects)

def assert_world(source,candidate,variant,matrix,objects):
    name,adapted=translate(variant,matrix)
    return pilot_assert(source,candidate,name,adapted,objects)

def prepare():
    target=OUT/'protocol.json'
    if target.exists():verify_tree(target);return read(target)
    verify_tree(PRIOR/'pilot-handoff.json')
    if read(PRIOR/'pilot-handoff.json')['status']!='24_frame_pilot_complete_not_full_matrix_or_training':raise ValueError('Pilot gate not complete')
    review=ORIGINAL/'review/decisions.json';audit=ORIGINAL/'review/intake-audit-v2.json'
    verify_tree(review);verify_tree(audit)
    r=read(review);a=read(audit)
    if r['frame_count']!=8 or any(o['review_status']!='visible_content_observed' for f in r['frames'] for o in f['objects']):raise ValueError('Original review not complete')
    if a['gaps'] or any(f['exact_matches'] or f['near_matches'] for f in a['frames']):raise ValueError('Reference audit held')
    sourcepath=ORIGINAL/'plan/plan.json';source=read_record(sourcepath)
    rp=ORIGINAL/'capture/collection-receipt.json';receipt=read_record(rp)
    gate,config,mapping=validate_preflight(source,sourcepath.parent,source['calibration_views'])
    restored,_=resumed_views(rp,source,'calibration',source['calibration_views'],config=config,check_version=gate['check_version'],instance_mapping=mapping)
    if len(restored)!=8:raise ValueError('Original capture invalid')
    # Explicit linkage overrides the per-image identities; neither maps nor assets are new.
    lineage=[dict(frame_id=f['frame_id'],original_view_id=f['view_id'],source_group=f['shared_source_group'],
      parent_view_id=f['derived_from_view_id'],map_id='complex',data_role='new_training_candidate',
      scene_independence=False,asset_ids=sorted(o['name'] for o in source['objects']),
      visible_equipment_ids=[o['object_id'] for o in f['objects']]) for f in r['frames']]
    OUT.mkdir(exist_ok=False);runs=[]
    for variant in VARIANTS:
        folder=OUT/variant/'plan';folder.mkdir(parents=True)
        for name,digest in source['files'].items():
            p=sourcepath.parent/name
            if file_sha256(p)!=digest:raise ValueError('Changed source file')
            (folder/name).write_bytes(p.read_bytes())
        tree=ET.parse(folder/'world.sdf');transform(tree.getroot(),variant,MATRIX,source['objects'])
        ET.indent(tree,space='  ');tree.write(folder/'world.sdf',encoding='utf-8',xml_declaration=True)
        assert_world(sourcepath.parent/'world.sdf',folder/'world.sdf',variant,MATRIX,source['objects'])
        p=copy.deepcopy(source);p.pop('identity');views=[]
        byid={f['view_id']:f for f in r['frames']}
        for v in source['calibration_views']:
            f=byid[v['view_id']];v=copy.deepcopy(v)
            v.update(original_view_id=f['view_id'],variant=variant,pair_id=v['pose_id'],
              shared_source_group=f['shared_source_group'],derivation_group=f['shared_source_group'],
              view_id=object_sha256({'original':f['view_id'],'variant':variant,'experiment':'full-image-appearance-v1'}))
            views.append(v)
        p['calibration_views']=views;p['files']['world.sdf']=file_sha256(folder/'world.sdf')
        p=write_record(folder/'plan.json',p);validate_preflight(p,folder,views)
        runs.append(dict(variant=variant,plan_path=str(folder/'plan.json')))
    paths=[PRIOR/'pilot-handoff.json',review,audit,sourcepath,rp,Path(__file__),Path(__file__).with_name('capture_appearance_recovery_pilot.py')]+[Path(x['plan_path']) for x in runs]
    return save(target,dict(status='frozen_32_frame_remaining',runs=runs,matrix=MATRIX,lineage=lineage,
      original_frames=8,new_frames=32,source_groups=7,map_count=1,
      role_policy='All variants and nearby parent views remain together as development training candidates. Shared complex layout/assets with viewed development references is disclosed, not independent-scene isolation. No sealed testing or training admission.',
      review_policy='Every new frame and full-image box reviewed separately; corresponding original boxes must differ by at most 1 pixel. No automatic approval or further palette expansion before review.',
      inputs={str(p):file_sha256(p) for p in paths}))

async def main():
    p=prepare();runs=[]
    with (OUT/'capture.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for run in p['runs']:
            pp=Path(run['plan_path']);plan=read_record(pp);out=pp.parent.parent/'capture';rp=out/'collection-receipt.json'
            gate,config,mapping=validate_preflight(plan,pp.parent,plan['calibration_views'])
            if out.exists():
                if not rp.exists():raise ValueError('Incomplete prior attempt retained')
                restored,_=resumed_views(rp,plan,'calibration',plan['calibration_views'],config=config,check_version=gate['check_version'],instance_mapping=mapping)
                if len(restored)!=8:raise ValueError('Incomplete attempt; no reset of retry budget')
                r=read_record(rp)
            else:r=await collect(pp,out,mode='calibration')
            n=sum(v['status']=='captured' for v in r['views'])
            runs.append(dict(**run,receipt_path=str(rp),captured=n,status=r['status']))
            save(OUT/'progress.json',dict(status='complete_pending_AI_review' if len(runs)==4 and all(x['captured']==8 for x in runs) else 'incomplete',runs=runs,
              inputs={str(x):file_sha256(x) for x in [OUT/'protocol.json']+[Path(x['receipt_path']) for x in runs]}))
            print('CAPTURED',run['variant'],n,flush=True)
            if n!=8 or r['status']!='complete_pending_review':raise ValueError('Capture held')

if __name__=='__main__':asyncio.run(main())
