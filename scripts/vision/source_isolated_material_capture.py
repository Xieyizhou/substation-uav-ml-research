"""New-layout material candidates. Default freeze only; never train or approve."""
import argparse
import asyncio
import copy
import fcntl
import itertools
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.vision.prepare_visual_augmentation_batch import SOURCE
from scripts.vision.establish_material_view_candidates import prior
from scripts.vision.design_full_scene_poses import screen
from scripts.vision.build_material_view_world_drafts import material_nodes, check_only_materials, signature
from src.vision.canonical.plan import read_record, write_record, object_sha256
from src.vision.canonical.expansion import _pose
from src.vision.canonical.gates import validate_preflight, validate_view_pose
from src.vision.canonical.collect import collect
from src.vision.canonical.recovery import resumed_views

OUT = prior.ROOT/'data/research/ml_training_recovery_v1/source-isolated-material-retention-v1'
PALETTES = {'original': None, 'warm': '0.52 0.43 0.31 1',
            'cool': '0.27 0.34 0.43 1', 'neutral': '0.42 0.42 0.42 1'}
COUNTS = {'reactor': 3, 'capacitor_bank': 2, 'switchgear': 2, 'transformer': 1}


def overlap(a, b, clearance=.5):
    return all(a[i] <= b[i+1]+clearance and b[i] <= a[i+1]+clearance for i in (0, 2))


def relocate(source, config, tree, layout):
    """Fixed hash order packs existing assets at genuinely new XY centers."""
    objects = copy.deepcopy(source['objects']); config = copy.deepcopy(config)
    tree = copy.deepcopy(tree)
    occupied = [o['bounds'] for o in objects if o['category'] not in COUNTS]
    targets = sorted((o for o in objects if o['category'] in COUNTS),
                     key=lambda o: (-(o['bounds'][1]-o['bounds'][0])*(o['bounds'][3]-o['bounds'][2]), o['name']))
    changes = []
    for obj in targets:
        b = obj['bounds']; w, h = b[1]-b[0], b[3]-b[2]
        options = sorted(itertools.product(range(-11, 12, 2), repeat=2),
                         key=lambda xy: object_sha256(["layout-pack-v1", layout, obj['name'], xy]))
        selected = None
        for x, y in options:
            nb = [x-w/2, x+w/2, y-h/2, y+h/2, b[4], b[5]]
            if math.dist([x,y], [(b[0]+b[1])/2,(b[2]+b[3])/2]) < 2: continue
            if any(overlap(nb, other) for other in occupied): continue
            selected = nb; break
        if selected is None: raise ValueError('layout_packing_infeasible:'+obj['name'])
        dx, dy = selected[0]-b[0], selected[2]-b[2]
        model = tree.find('./world/model[@name="'+obj['name']+'"]')
        if model is None: raise ValueError('missing_top_level_asset')
        pose = model.find('pose')
        values = list(map(float, pose.text.split()))
        if pose.attrib: raise ValueError('relative_pose_not_supported')
        asset = copy.deepcopy(model); asset.find('pose').text = 'POSE_EXCLUDED'
        values[0] += dx; values[1] += dy; pose.text = ' '.join(map(str, values))
        cfg = next(o for o in config['obstacles'] if o['name']==obj['name'])
        for key in ('x', 'x_min', 'x_max'):
            if key in cfg: cfg[key] += dx
        for key in ('y', 'y_min', 'y_max'):
            if key in cfg: cfg[key] += dy
        changes.append(dict(object_id=obj['name'],translation=[dx,dy,0],
                            asset_signature=object_sha256(signature(asset)),old_bounds=b,new_bounds=selected))
        obj['bounds'] = selected; occupied.append(selected)
    # Only equipment translations and the explicit semantic-mode correction may differ.
    a, b = copy.deepcopy(tree), copy.deepcopy(ET.parse(SOURCE.parent/'world.sdf').getroot())
    for change in changes:
        for root in (a,b): root.find('./world/model[@name="'+change['object_id']+'"]/pose').text='ALLOWED_POSE'
    if signature(a)!=signature(b): raise ValueError('non_pose_layout_change')
    return objects, config, tree, changes


def choose(objects, config, layout):
    accepted = []; rejects = {}
    for obj in objects:
        if obj['category'] not in COUNTS: continue
        b=obj['bounds']; center=[(b[i]+b[i+1])/2 for i in (0,2,4)]
        for bearing, distance, height in itertools.product(range(0,360,15),(3.,4.,5.,7.5,10.,12.5),(3.,5.,7.,9.,11.)):
            a=math.radians(bearing); cam=[center[0]+distance*math.cos(a),center[1]+distance*math.sin(a),height]
            pos,q=_pose(cam,center,0)
            v=dict(map_id=layout,object_id=obj['name'],category=obj['category'],bearing=bearing,distance=distance,
                   height=height,offset=0,camera_position=cam,position=pos,orientation=q,family=layout+':'+obj['name'])
            v['view_id']=object_sha256(v)
            try: validate_view_pose(v,config); screen(v,objects)
            except ValueError as ex:
                reason=str(ex); rejects[reason]=rejects.get(reason,0)+1; continue
            accepted.append(v)
    selected=[]
    for category, count in COUNTS.items():
        options=sorted((r for r in accepted if r['category']==category),key=lambda r:(r['distance'],r['height'],r['object_id'],r['bearing']))
        chosen=[]
        for row in options:
            if any(row['object_id']==p['object_id'] and row['bearing']==p['bearing'] for p in chosen): continue
            chosen.append(row)
            if len(chosen)==count: break
        if len(chosen)!=count: raise ValueError('insufficient_legal_full_scene_poses:'+layout+':'+category)
        for i,row in enumerate(chosen):
            row['pair_id']=layout+':'+category+':'+str(i+1); row['pilot']=layout=='layout-A' and i==0
            selected.append(row)
    return selected, rejects


def freeze():
    dest=OUT/'protocol.json'
    if dest.exists(): p=prior.read(dest); prior.verify(p); return p
    source=read_record(SOURCE); config=json.loads((SOURCE.parent/'obstacles.json').read_text())
    for name,h in source['files'].items():
        if prior.file_sha256(SOURCE.parent/name)!=h: raise ValueError('ancestor_changed:'+name)
    OUT.mkdir(parents=True,exist_ok=True)
    tree=ET.parse(SOURCE.parent/'world.sdf').getroot(); units=[]; layouts=[]; paths=[Path(__file__),SOURCE]
    prepared=[]
    for layout in ('layout-A','layout-B'):
        failures=[]
        for attempt in range(1,21):
            try:
                objects,cfg,world,changes=relocate(source,config,tree,layout+':packing:'+str(attempt))
                views,rejects=choose(objects,cfg,layout)
                prepared.append((layout,objects,cfg,world,changes,views,rejects,failures));break
            except ValueError as ex:
                failures.append(dict(attempt=attempt,reason=str(ex)))
                print('LAYOUT_PREFILTER',layout,attempt,str(ex),flush=True)
        else: raise ValueError('bounded_layout_search_exhausted:'+layout)
    for layout,objects,cfg,world,changes,views,rejects,failures in prepared:
        boxes=world.findall('.//sensor[@type="boundingbox_camera"]/camera/box_type')
        if len(boxes)!=1: raise ValueError('camera_not_unique')
        boxes[0].text='full_2d'
        layouts.append(dict(layout_id=layout,changes=changes,selected=views,rejections=rejects,geometry_rejected_layouts=failures,
                            asset_independent=False,ancestor_world_sha256=source['files']['world.sdf']))
        names={o['name'] for o in objects if o['category'] in COUNTS}
        for variant,color in PALETTES.items():
            folder=OUT/'plans'/layout/variant; folder.mkdir(parents=True,exist_ok=False)
            modified=copy.deepcopy(world)
            if color:
                nodes=material_nodes(modified,names)
                for n in nodes.values(): n.text=color
            differences=check_only_materials(world,modified,names)
            for name in source['files']:
                if name in ('world.sdf','obstacles.json'): continue
                (folder/name).write_bytes((SOURCE.parent/name).read_bytes())
            ET.ElementTree(modified).write(folder/'world.sdf',encoding='utf-8',xml_declaration=True)
            (folder/'obstacles.json').write_text(json.dumps(cfg,indent=2))
            plan={k:copy.deepcopy(v) for k,v in source.items() if k!='identity'}
            plan.update(map_id=layout,objects=objects,calibration_views=views,pilot_views=[],annotation_mode='full_2d',
                        label_mode='visual-instance',hierarchy_mode='top-level-equipment',diagnostic_only=True,
                        diagnostic_require_expected_presence=True,training_admitted=False,promotable=False,
                        canonical_scene_geometry_modified=True,automatic_training=False,variant=variant)
            plan['files']={name:prior.file_sha256(folder/name) for name in source['files']}
            record=write_record(folder/'plan.json',plan)
            gate,_,_=validate_preflight(record,folder,views)
            paths.extend(folder.iterdir())
            for view in views:
                units.append(dict(key=view['pair_id']+':'+variant,pair_id=view['pair_id'],variant=variant,
                                  plan_path=str(folder/'plan.json'),view_id=view['view_id'],pilot=view['pilot'],
                                  material_changes=differences,world_sha256=gate['world_sha256']))
    return prior.frozen(dest,dict(status='64_candidates_frozen_16_pilot_authorized',units=units,layouts=layouts,
        data_role='new_training_candidate',palettes=PALETTES,independent_pose_groups=16,independent_asset_families_added=0,
        source_scope='Two deterministic rearrangements of known training-layout assets, shared ancestor; not sealed/new-asset testing.',
        training_ready=False,training_started=False,expansion_requires_explicit_pilot_review=True,
        inputs={str(p):prior.file_sha256(p) for p in paths}))


def verify_capture(unit, receipt_path):
    plan=read_record(unit['plan_path']); base=Path(unit['plan_path']).parent
    views=[v for v in plan['calibration_views'] if v['view_id']==unit['view_id']]
    gate,cfg,mapping=validate_preflight(plan,base,views)
    receipt=read_record(receipt_path)
    if receipt['world_sha256']!=gate['world_sha256']: raise ValueError('capture_world_changed')
    rows,_=resumed_views(receipt_path,plan,'calibration',views,config=cfg,check_version=gate['check_version'],instance_mapping=mapping)
    if receipt['status']!='complete_pending_review' or len(rows)!=1: raise ValueError('capture_not_complete')
    return rows[0]


async def run():
    p=freeze(); dest=OUT/'pilot-completion.json'
    if dest.exists(): prior.verify(prior.read(dest)); return
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=[]; paths=[OUT/'protocol.json']; failed=False
        for unit in (r for r in p['units'] if r['pilot']):
            folder=OUT/'captures'/unit['key']; rp=folder/'collection-receipt.json'
            if not rp.exists():
                if folder.exists(): raise ValueError('incomplete_capture_preserved:'+str(folder))
                print('CAPTURING',unit['key'],flush=True)
                await collect(unit['plan_path'],folder,mode='calibration',view_id=unit['view_id'])
            paths.append(rp)
            try:
                frame=verify_capture(unit,rp)
                paths.extend([Path(frame['rgb_path']),Path(frame['depth_path'])])
                status='captured_review_pending';reason=None
            except ValueError as ex: status='blocked';reason=str(ex);failed=True
            rows.append(dict(unit,receipt=str(rp),status=status,reason=reason))
            print(status,reason,flush=True)
            if failed: break
        return prior.frozen(dest,dict(status='pilot_blocked' if failed else '16_captured_review_required_no_expansion',units=rows,
            training_ready=False,training_started=False,inputs={str(p):prior.file_sha256(p) for p in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--capture-pilot',action='store_true');args=ap.parse_args()
    if args.capture_pilot: asyncio.run(run())
    else: print(freeze()['status'],'NO_CAPTURE_NO_TRAINING')
