"""Identity-bound canonical camera views; no equipment geometry changes."""
import copy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from src.ml.artifacts import object_sha256, file_sha256
from src.vision.collection.simulator_labels import CLASS_LABELS, instance_simulator_label

ROOT = Path(__file__).resolve().parents[3]
MAPS = {'simple':'config/substation_obstacles.json', 'medium':'config/maps/substation_medium.json', 'complex':'config/maps/substation_complex.json', 'extreme':'config/maps/substation_extreme.json'}
CARRIER = 'canonical_camera'

def write_record(path, record):
    record = dict(record)
    record['identity'] = object_sha256(record)
    with Path(path).open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True); f.write('\n')
    return record

def read_record(path):
    record = json.loads(Path(path).read_text())
    identity = record.pop('identity')
    if identity != object_sha256(record):
        raise ValueError('Artifact identity mismatch')
    return {**record, 'identity':identity}

def quaternion(roll, pitch, yaw):
    cr,sr=math.cos(roll/2),math.sin(roll/2); cp,sp=math.cos(pitch/2),math.sin(pitch/2); cy,sy=math.cos(yaw/2),math.sin(yaw/2)
    return [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]

def rotate(q,v):
    x,y,z,w=q; a,b,c=v
    return [(1-2*(y*y+z*z))*a+2*(x*y-z*w)*b+2*(x*z+y*w)*c,2*(x*y+z*w)*a+(1-2*(x*x+z*z))*b+2*(y*z-x*w)*c,2*(x*z-y*w)*a+2*(y*z+x*w)*b+(1-2*(x*x+y*y))*c]

def pose_close(actual, requested):
    if not all(math.isfinite(v) for v in actual['position']+actual['orientation']): return False
    distance=math.dist(actual['position'],requested['position'])
    a=actual['orientation'];b=requested['orientation'];norm=math.sqrt(sum(v*v for v in a)*sum(v*v for v in b))
    if norm == 0:return False
    angle=2*math.acos(min(1,abs(sum(x*y for x,y in zip(a,b))/norm)))
    return distance <= .05+1e-9 and angle <= math.radians(1)+1e-9

def bounds(o, origin):
    x0=o.get('x_min',o.get('x'));x1=o.get('x_max',o.get('x'));y0=o.get('y_min',o.get('y'));y1=o.get('y_max',o.get('y'))
    return [x0+origin[0],x1+1+origin[0],y0+origin[1],y1+1+origin[1],o.get('z_min_m',0),o['z_max_m']]

def visible(camera,target,obstacles,target_name):
    # Conservative segment test through configured solid occupancy.
    steps=max(2,math.ceil(math.dist(camera,target)/.1))
    for i in range(steps):
        p=[a+(b-a)*i/steps for a,b in zip(camera,target)]
        for o in obstacles:
            if o['name']==target_name:continue
            x0,x1,y0,y1,z0,z1=o['bounds']
            if x0-.1<=p[0]<=x1+.1 and y0-.1<=p[1]<=y1+.1 and z0-.1<=p[2]<=z1+.1:return False
    return True

def materialize(map_id, output, label_mode='source', hierarchy_mode='source', instance_collision_strategy='reject'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    paths={'source_world.sdf':ROOT/f'simulation/worlds/substation_{map_id}.sdf','obstacles.json':ROOT/MAPS[map_id],'sensor_source.sdf':ROOT/'simulation/models/x500_research/model.sdf','label_mapping.py':ROOT/'src/vision/collection/simulator_labels.py'}
    hashes={}
    for name,path in paths.items():
        payload=path.read_bytes();(output/name).write_bytes(payload);hashes[name]=file_sha256(output/name)
    config=json.loads((output/'obstacles.json').read_text());origin=config['gazebo_world_origin_m']
    objects=[{'name':o['name'],'category':o.get('visual_category'),'bounds':bounds(o,origin)} for o in config['obstacles']]
    tree=ET.parse(output/'source_world.sdf');world=tree.getroot().find('world');world_name=f'canonical_{map_id}'
    world.set('name',world_name)
    if hierarchy_mode not in ('source','top-level-equipment'):
        raise ValueError('Unsupported hierarchy mode')
    hierarchy_changes=[]
    if hierarchy_mode=='top-level-equipment':
        from .hierarchy import flatten_equipment
        hierarchy_changes=flatten_equipment(world,[o['name'] for o in objects])
    # Preserve scene content; only configure simulator systems and the camera carrier.
    systems=[('physics','Physics'),('user-commands','UserCommands'),('scene-broadcaster','SceneBroadcaster'),('sensors','Sensors')]
    for lib,cls in systems:
        plugin=next((p for p in world.findall('plugin') if p.get('name')==f'gz::sim::systems::{cls}'),None)
        if plugin is None:plugin=ET.SubElement(world,'plugin',filename=f'gz-sim-{lib}-system',name=f'gz::sim::systems::{cls}')
        if cls=='Sensors' and plugin.find('render_engine') is None:ET.SubElement(plugin,'render_engine').text='ogre2'
    mapping=[];used=set();collision_resolutions=[]
    models={m.get('name'):m for m in world.iter('model')}
    for o in objects:
        m=models.get(o['name'])
        if m is None:raise ValueError(f'Canonical object missing: {o["name"]}')
        old=[p.findtext('label') for p in m.iter('plugin') if p.get('name')=='gz::sim::systems::Label']
        if label_mode=='visual-instance' and o['category'] in CLASS_LABELS:
            label=instance_simulator_label(o['category'],o['name'])
            if label in used:
                if instance_collision_strategy != 'linear-probe':
                    raise ValueError('Instance label collision; no silent reassignment')
                original=label; class_base=CLASS_LABELS[o['category']]*50
                for offset in range(1,50):
                    candidate=class_base+((label-class_base-1+offset)%49)+1
                    if candidate not in used:label=candidate;break
                else:raise ValueError('Instance label class range exhausted')
                collision_resolutions.append({'object_id':o['name'],'hashed_label':original,'resolved_label':label,'strategy':'linear-probe-v1'})
            used.add(label)
            for parent in m.iter():
                for plugin in list(parent.findall('plugin')):
                    if plugin.get('name')=='gz::sim::systems::Label':parent.remove(plugin)
            for v in m.findall('./link/visual'):
                plugin=ET.SubElement(v,'plugin',filename='gz-sim-label-system',name='gz::sim::systems::Label');ET.SubElement(plugin,'label').text=str(label)
        mapping.append({**o,'source_labels':old,'runtime_labels':[p.findtext('label') for p in m.iter('plugin') if p.get('name')=='gz::sim::systems::Label']})
    sensor=ET.parse(output/'sensor_source.sdf').find('.//link[@name="research_camera_link"]')
    if sensor is None:raise ValueError('Missing camera link')
    extrinsic=[float(x) for x in sensor.findtext('pose').split()]
    if len(extrinsic)!=6 or any(extrinsic[3:]):raise ValueError('Unsupported rotated camera-link extrinsic')
    carrier=ET.SubElement(world,'model',name=CARRIER);ET.SubElement(carrier,'static').text='true';ET.SubElement(carrier,'pose').text='0 0 100 0 0 0'
    link=copy.deepcopy(sensor);link.find('pose').attrib.clear()
    for v in list(link.findall('visual')):link.remove(v)
    carrier.append(link)
    ET.indent(tree,space='  ');tree.write(output/'world.sdf',encoding='utf-8',xml_declaration=True)
    targets=[o for o in objects if o['category'] in CLASS_LABELS or o['category']=='cabinet']
    targets += [{'name':'empty_ground','category':'empty_ground','bounds':[origin[0]+1,origin[0]+3,origin[1]+1,origin[1]+3,0,0]}]
    candidates=[]
    for o in targets:
        b=o['bounds'];target=[(b[0]+b[1])/2,(b[2]+b[3])/2,(b[4]+b[5])/2]
        for bearing in range(0,360,45):
            for distance in (6,10,16):
                for height in (1.5,2.5,4):
                    theta=math.radians(bearing);camera=[target[0]+distance*math.cos(theta),target[1]+distance*math.sin(theta),height]
                    if not (origin[0]+.25<camera[0]<origin[0]+config['width']-.25 and origin[1]+.25<camera[1]<origin[1]+config['height']-.25):continue
                    if not visible(camera,target,objects,o['name']):continue
                    if any(a['bounds'][0]-.25<=camera[0]<=a['bounds'][1]+.25 and a['bounds'][2]-.25<=camera[1]<=a['bounds'][3]+.25 and a['bounds'][4]-.25<=height<=a['bounds'][5]+.25 for a in objects):continue
                    for offset in (0,-15,15):
                        yaw=math.atan2(target[1]-camera[1],target[0]-camera[0])+math.radians(offset);pitch=math.atan2(camera[2]-target[2],distance);q=quaternion(0,pitch,yaw);delta=rotate(q,extrinsic[:3]);position=[a-b for a,b in zip(camera,delta)]
                        row={'map_id':map_id,'object_id':o['name'],'category':o['category'],'bearing':bearing,'distance':distance,'height':height,'offset':offset,'position':position,'orientation':q,'camera_position':camera,'family':f'{map_id}:{o["name"]}:{bearing}'}
                        row['view_id']=object_sha256(row);candidates.append(row)
    calibration=[]
    categories=sorted({o['category'] for o in targets})
    for category in categories:
        options=sorted([v for v in candidates if v['category']==category and v['offset']==0],key=lambda v:(v['distance'],v['height'],v['object_id'],v['bearing']))
        chosen=[]
        for v in options:
            if v['bearing'] not in {c['bearing'] for c in chosen}:chosen.append(v)
            if len(chosen)==3:break
        if len(chosen)<3:raise ValueError(f'Not enough calibration views: {category}')
        calibration.extend(chosen)
    selected=list(calibration);seen={v['view_id'] for v in selected}
    remaining=sorted(candidates,key=lambda v:v['view_id'])
    for v in remaining:
        if len(selected)>=40:break
        if v['view_id'] not in seen:selected.append(v);seen.add(v['view_id'])
    record={'schema_version':1,'map_id':map_id,'world_name':world_name,'split':'development','label_mode':label_mode,'files':{**hashes,'world.sdf':file_sha256(output/'world.sdf')},'objects':mapping,'calibration_views':calibration,'pilot_views':selected,'candidate_count':len(candidates),'pose_tolerance_m':.05,'attitude_tolerance_deg':1,'max_skew_ms':33.334,'stable_frames':3,'view_timeout_s':10,'automatic_training':False,'canonical_scene_geometry_modified':False}
    record['hierarchy_mode']=hierarchy_mode
    record['hierarchy_changes']=hierarchy_changes
    record['instance_collision_strategy']=instance_collision_strategy
    record['instance_collision_resolutions']=collision_resolutions
    return write_record(output/'plan.json',record)
