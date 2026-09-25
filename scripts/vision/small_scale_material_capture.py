"""Bounded known-asset small-scale four-condition capture, no automatic approval."""
import argparse,asyncio,copy,fcntl,json,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.vision.probe_small_material_layouts import OUT as CENSUS,REF,prior
from scripts.vision.source_isolated_material_capture import SOURCE,relocate
from scripts.vision.build_material_view_world_drafts import material_nodes,check_only_materials
from scripts.vision.cool_light_capture import LIGHT,lighting_only
from src.vision.canonical.plan import read_record,write_record
from src.vision.canonical.gates import validate_preflight
from src.vision.canonical.collect import collect

OUT=REF/'small-scale-material-control-v1'
CLASSES=('capacitor_bank','reactor','switchgear','transformer')


def select(census):
    chosen=[]
    for category in CLASSES:
        for attempt in census['attempts']:
            rows=[x for x in attempt.get('candidates',[]) if x['view']['category']==category]
            if not rows:continue
            row=sorted(rows,key=lambda x:(x['view']['height'],x['view']['distance'],x['view']['object_id'],x['view']['bearing']))[0]
            chosen.append(dict(layout=attempt['layout'],candidate=row,translations=attempt['translations']));break
        else:raise ValueError('Missing small-scale class '+category)
    return chosen


def freeze():
    dest=OUT/'capture-protocol.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    cp=CENSUS/'census.json';c=prior.read(cp);prior.verify(c);chosen=select(c)
    ancestor=read_record(SOURCE);config=prior.read(SOURCE.parent/'obstacles.json');tree=ET.parse(SOURCE.parent/'world.sdf').getroot()
    deps=[cp,REF/'evaluation/error-review-v1/completion.json',REF/'evaluation/analysis-v1.json',REF/'pool-fit-diagnosis-v1/summary.json',Path(__file__).resolve(),SOURCE]
    units=[];OUT.mkdir(exist_ok=True)
    for n,choice in enumerate(chosen,1):
        objects,cfg,base,changes=relocate(ancestor,config,tree,choice['layout'])
        if changes!=choice['translations']:raise ValueError('Layout not reproducible')
        boxes=base.findall('.//sensor[@type="boundingbox_camera"]/camera/box_type')
        if len(boxes)!=1:raise ValueError('Box mode ambiguity')
        boxes[0].text='full_2d'
        v=copy.deepcopy(choice['candidate']['view']);v['pair_id']=f'small-material-v1:{n:02}';v['data_role']='new_training_candidate'
        names={o['name'] for o in objects if o['category'] in CLASSES}
        for light,variant in (('normal','original'),('normal','gray035'),('cool','original'),('cool','gray035')):
            uid=f'S{len(units)+1:02}';folder=OUT/'plans'/uid;folder.mkdir(parents=True)
            for name,digest in ancestor['files'].items():
                path=SOURCE.parent/name
                if prior.file_sha256(path)!=digest:raise ValueError('Ancestor drift')
                shutil.copy2(path,folder/name);deps.append(path)
            material=copy.deepcopy(base)
            if variant=='gray035':
                for node in material_nodes(material,names).values():node.text='0.35 0.35 0.35 1'
            material_changes=check_only_materials(base,material,names)
            world=copy.deepcopy(material)
            if light=='cool':
                for path,value in LIGHT.items():
                    nodes=world.findall(path)
                    if len(nodes)!=1:raise ValueError('Light ambiguity')
                    nodes[0].text=value
                lighting_only(material,world)
            ET.ElementTree(world).write(folder/'world.sdf',encoding='utf-8',xml_declaration=True)
            (folder/'obstacles.json').write_text(json.dumps(cfg,indent=2))
            p={k:copy.deepcopy(x) for k,x in ancestor.items() if k!='identity'}
            p.update(objects=objects,map_id=choice['layout'],calibration_views=[v],pilot_views=[],annotation_mode='full_2d',label_mode='visual-instance',hierarchy_mode='top-level-equipment',
                diagnostic_only=True,diagnostic_require_expected_presence=True,data_role='bounded_development_training_candidate',training_admitted=False,promotable=False,automatic_training=False,canonical_scene_geometry_modified=True,variant=variant)
            p['files']={name:prior.file_sha256(folder/name) for name in ancestor['files']}
            write_record(folder/'plan.json',p);validate_preflight(p,folder,[v]);deps.extend(folder.iterdir())
            units.append(dict(unit_id=uid,plan_path=str(folder/'plan.json'),view=v,pair_id=v['pair_id'],variant=variant,illumination=light,layout=choice['layout'],
                projected_targets=choice['candidate']['projected_targets'],material_changes=material_changes))
    return prior.frozen(dest,dict(status='16_small_scale_candidates_frozen_quality_pending',units=units,pilot_units=[u['unit_id'] for u in units if u['illumination']=='normal'],
        selection='First qualifying layout in the frozen 20-layout census per class; then fixed height/distance/object/bearing sort. No prediction used for selection. Four poses in three training-layout rearrangements.',
        training_proposal=dict(families=['SR1100','SM1100'],seeds=[7,17,27],initialization='independent_v2.11',preserved_ICG1000_draws=6000,additional_draws=600,steps=1100,
            reference='New small-pose original-material images under normal/cool light.',treatment='Same poses, lighting positions and full labels; replace half the added original-material exposures with paired gray035 images.',
            fixed='CPU640 batch/nbs6 AdamW LR.0005 no warmup; preserve old brightness and use frozen first-600 source brightness factors in the new tail for both families; endpoint last.pt only.',
            restriction='Actual full labels must be identical within each four-condition pose. Across both new families, complete class/lineage budgets and all old/negative positions must match. Training only after review and real preflight.'),
        limitations=['Only four new pose groups and shared asset geometry, not broad source diversity or blind generalization.',
                    'Separate matched-budget reference is required: compare added material coverage at small scales, not pure scale causality against shorter IC1000/ICG1000.',
                    'Geometry is a prefilter, not visibility certification. All captured targets and unmapped structures require review. No protected labels or sealed test scenes.'],
        inputs={str(d):prior.file_sha256(d) for d in deps}))


async def run(pilot=False):
    p=freeze()
    if not pilot:
        r=prior.read(OUT/'pilot-quality.json');prior.verify(r)
        if r['status']!='pilot_reviewed_expand_remaining_eight':raise ValueError('Pilot not approved')
    with (OUT/'capture.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for u in p['units']:
            if pilot and u['unit_id'] not in p['pilot_units']:continue
            folder=OUT/'captures'/u['unit_id'];cp=folder/'collection-receipt.json'
            if cp.exists():r=read_record(cp)
            elif folder.exists():raise ValueError('Incomplete capture retained '+u['unit_id'])
            else:
                print('CAPTURING',u['unit_id'],u['view']['category'],flush=True)
                r=await collect(u['plan_path'],folder,mode='calibration')
            if r['status']!='complete_pending_review' or len(r['views'])!=1 or r['views'][0]['status']!='captured':raise ValueError('Capture blocked '+u['unit_id'])
            print('CAPTURED',u['unit_id'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--capture',action='store_true');ap.add_argument('--pilot',action='store_true');a=ap.parse_args()
    if a.capture:asyncio.run(run(a.pilot))
    else:print(freeze()['status'])
