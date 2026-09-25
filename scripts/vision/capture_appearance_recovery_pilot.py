"""Materialize tightly checked worlds and capture the frozen twelve-frame pilot."""
import argparse
import asyncio
import copy
import fcntl
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.vision.prepare_appearance_recovery_candidates import OUT,SOURCE,read,save,file_sha256,verify_tree,ROOT
from src.ml.artifacts import object_sha256
from src.vision.canonical.plan import read_record,write_record
from src.vision.canonical.gates import annotation_mode_from_world,validate_preflight
from src.vision.canonical.recovery import resumed_views
from src.vision.canonical.collect import collect

def payload(node):
    return (node.tag,tuple(sorted(node.attrib.items())),(node.text or '').strip(),tuple(payload(x) for x in node))

def transform(root,variant,matrix,objects):
    sensors=[s for s in root.iter('sensor') if s.get('type')=='boundingbox_camera']
    if len(sensors)!=1 or sensors[0].find('camera/box_type') is None:raise ValueError('Bounding-box sensor missing/duplicate')
    sensors[0].find('camera/box_type').text='full_2d'
    world=root.find('world');changed=0
    if variant!='original':
        color=matrix['materials']['steel'];names={o['name'] for o in objects if o['category'] in ('transformer','switchgear','reactor','capacitor_bank')}
        for model in world.iter('model'):
            if model.get('name') not in names:continue
            for visual in model.iter('visual'):
                if visual.get('name') not in ('body','front_panel','reactor'):continue
                for tag in ('ambient','diffuse'):
                    node=visual.find('material/'+tag)
                    if node is None:raise ValueError('Expected material field missing')
                    node.text=color;changed+=1
        if not changed:raise ValueError('No target materials changed')
    if variant=='steel_warm_dim':
        values=matrix['lights']['warm_dim']
        for path,value in [('scene/ambient',values['ambient']),('light[@name="sun"]/diffuse',values['sun_diffuse'])]:
            node=world.find(path)
            if node is None:raise ValueError('Required light missing')
            node.text=value
    if variant not in ('original','steel_original','steel_warm_dim'):raise ValueError('Unknown pilot variant')

def assert_world(source,candidate,variant,matrix,objects):
    expected=ET.parse(source).getroot();transform(expected,variant,matrix,objects)
    if payload(expected)!=payload(ET.parse(candidate).getroot()):raise ValueError('World fields differ from exact allowed transformation')
    if annotation_mode_from_world(candidate)!='full_2d':raise ValueError('World mode not persisted')

def prepare():
    path=OUT/'pilot-protocol.json'
    if path.exists():verify_tree(path);return read(path)
    verify_tree(OUT/'handoff.json');m=read(OUT/'candidate-matrix.json');source=read_record(SOURCE);runs=[]
    for variant in m['pilot']['variants']:
        folder=OUT/'pilot'/variant/'plan';folder.mkdir(parents=True,exist_ok=False)
        for name,digest in source['files'].items():
            src=SOURCE.parent/name
            if file_sha256(src)!=digest:raise ValueError('Source changed')
            (folder/name).write_bytes(src.read_bytes())
        tree=ET.parse(folder/'world.sdf');transform(tree.getroot(),variant,m,source['objects'])
        ET.indent(tree,space='  ');tree.write(folder/'world.sdf',encoding='utf-8',xml_declaration=True)
        assert_world(SOURCE.parent/'world.sdf',folder/'world.sdf',variant,m,source['objects'])
        p=copy.deepcopy(source);p.pop('identity');views=[]
        for row in m['views']:
            v=copy.deepcopy(row);v.update(source_view_id=row['view_id'],pose_id=row['view_id'],variant=variant,pair_id=object_sha256({'experiment':'appearance-recovery-pilot-v1','pose':row['view_id']}),derivation_group='appearance-recovery:'+row['view_id'],data_role='new_training_candidate',training_admitted=False,promotable=False)
            v['view_id']=object_sha256({'pose':row['view_id'],'variant':variant,'experiment':'appearance-recovery-pilot-v1'});views.append(v)
        p.update(annotation_mode='full_2d',label_mode='visual-instance',hierarchy_mode='top-level-equipment',calibration_views=views,pilot_views=[],diagnostic_only=True,diagnostic_require_expected_presence=True,automatic_training=False,training_admitted=False,promotable=False,variant=variant)
        p['files']['world.sdf']=file_sha256(folder/'world.sdf');p=write_record(folder/'plan.json',p)
        gate,_,_=validate_preflight(p,folder,views)
        runs.append(dict(variant=variant,plan_path=str(folder/'plan.json'),plan_identity=p['identity'],gate=gate))
    inputs={str(x):file_sha256(x) for x in [OUT/'candidate-matrix.json',OUT/'handoff.json',Path(__file__),ROOT/'tests/test_appearance_recovery_world.py']+[Path(r['plan_path']) for r in runs]}
    return save(path,dict(status='frozen_before_capture',runs=runs,frames=12,max_technical_attempts_per_frame=3,review_status='not_started',inputs=inputs))

async def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args();p=prepare()
    if args.prepare_only:print('WORLD_PREFLIGHT_PASSED',OUT,flush=True);return
    with (OUT/'capture.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);runs=[]
        for spec in p['runs']:
            planpath=Path(spec['plan_path']);plan=read_record(planpath);folder=planpath.parent
            for name,digest in plan['files'].items():
                if file_sha256(folder/name)!=digest:raise ValueError('Stale generated world input')
            gate,config,mapping=validate_preflight(plan,folder,plan['calibration_views'])
            out=folder.parent/'capture';receipt=out/'collection-receipt.json'
            if out.exists():
                if not receipt.exists():raise ValueError('Incomplete prior capture; preserve and investigate before any restart')
                r=read_record(receipt)
                recovered,_=resumed_views(receipt,plan,'calibration',plan['calibration_views'],config=config,check_version=gate['check_version'],instance_mapping=mapping)
                if len(recovered)!=4:raise ValueError('Prior capture incomplete; no silent attempt-budget reset')
            else:r=await collect(planpath,out,mode='calibration')
            count=sum(x['status']=='captured' for x in r['views']);runs.append(dict(variant=spec['variant'],receipt_path=str(receipt),captured=count,status=r['status']))
            save(OUT/'capture-progress.json',dict(status='complete_pending_AI_review' if len(runs)==3 and all(x['captured']==4 for x in runs) else 'in_progress_or_incomplete',runs=runs,inputs={str(receipt):file_sha256(receipt),str(OUT/'pilot-protocol.json'):file_sha256(OUT/'pilot-protocol.json')}))
            print('CAPTURED',spec['variant'],count,r['status'],flush=True)
            if count!=4 or r['status']!='complete_pending_review':raise ValueError('Pilot incomplete; stop before remaining variants')

if __name__=='__main__':asyncio.run(main())
