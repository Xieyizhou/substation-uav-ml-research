"""Frozen, non-admitting instance visibility diagnostics. Historical inputs are read-only."""
import argparse
import asyncio
import copy
import hashlib
import json
import os
import signal
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from scripts.vision.record_held_visibility_followup import OUT as FOLLOWUP, SOURCE, read, save, file_sha256, verify_tree
from scripts.vision.audit_edge_label_sources import BASE, ROOT
from scripts.vision.run_stratified_negative_control import OUT as TRAIN
from src.vision.canonical.gates import instance_mapping, annotation_mode_from_world

OUT = SOURCE / 'instance-visibility-diagnosis-v1'
PILOT = ['T049', 'T054', 'T077', 'T095']

def pixel_hash(path):
    im=Image.open(path).convert('RGB')
    return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()

def bind(inputs, path):
    path=Path(path);inputs[str(path)]=file_sha256(path);return path

def tree_signature(node):
    return (node.tag,tuple(sorted(node.attrib.items())),(node.text or '').strip(),tuple(tree_signature(c) for c in node))

def add_sensor(tree):
    result=copy.deepcopy(tree)
    link=result.find(".//model[@name='canonical_camera']/link[@name='research_camera_link']")
    if link is None:raise ValueError('Missing camera link')
    rgb=link.find("sensor[@name='research_rgb']")
    sensor=copy.deepcopy(rgb);sensor.set('name','diagnostic_instances');sensor.set('type','segmentation_camera')
    sensor.find('topic').text='diagnostic/instances'
    ET.SubElement(sensor.find('camera'),'segmentation_type').text='panoptic'
    link.append(sensor)
    validate_world(tree,result)
    return result

def validate_world(original, derived):
    clean=copy.deepcopy(derived)
    found=clean.findall(".//sensor[@name='diagnostic_instances']")
    if len(found)!=1:raise ValueError('Missing/duplicate diagnostic sensor')
    sensor=found[0]
    if sensor.get('type')!='segmentation_camera' or sensor.findtext('camera/segmentation_type')!='panoptic':
        raise ValueError('Not an instance sensor')
    link=clean.find(".//model[@name='canonical_camera']/link[@name='research_camera_link']")
    expected=copy.deepcopy(link.find("sensor[@name='research_rgb']"))
    expected.set('name','diagnostic_instances');expected.set('type','segmentation_camera')
    expected.find('topic').text='diagnostic/instances';ET.SubElement(expected.find('camera'),'segmentation_type').text='panoptic'
    if tree_signature(expected)!=tree_signature(sensor):raise ValueError('Diagnostic sensor not RGB aligned')
    link.remove(sensor)
    if tree_signature(original.getroot())!=tree_signature(clean.getroot()):raise ValueError('Non-allowed world change')

def raw_box(raw):
    a=raw['box']['minCorner'];b=raw['box']['maxCorner']
    return [a.get('x',0),a.get('y',0),b.get('x',0),b.get('y',0)]

def prepare():
    protocol=OUT/'protocol.json'
    if protocol.exists():verify_tree(protocol);return read(protocol)
    verify_tree(FOLLOWUP/'review.json')
    m=read(FOLLOWUP/'manifest.json');pool=read(TRAIN/'protocol.json')['pool_rows']
    inputs={};[bind(inputs,p) for p in (FOLLOWUP/'review.json',FOLLOWUP/'manifest.json',TRAIN/'protocol.json',Path(__file__))]
    frames={}
    for r in m['items']:frames.setdefault(r['member_id'],[]).append(r)
    index={}
    # Only the already-reviewed positive development sources are searched.
    for manifest in (BASE/'visual-augmentation-240-v1/positive-review-v1/manifest.json',BASE/'visual-bridge-supplement-v2/frozen-positive-ledger.json'):
        bind(inputs,manifest)
        for r in read(manifest)['frames']:
            if 'positive-review' in str(manifest) and r.get('subset')!='regular_positive':continue
            p=Path(r['image_path']);index.setdefault(pixel_hash(p),[]).append(p)
    results=[]
    for mid,items in frames.items():
        row=items[0];entry=dict(member_id=mid,review_ids=[r['review_id'] for r in items],image_path=row['image_path'],label_path=row['label_path'],lineage_id=row['lineage_id'])
        try:
            for k in ('image','label'):
                if file_sha256(bind(inputs,row[k+'_path']))!=row[k+'_sha256']:raise ValueError('Stale training '+k)
            if mid.startswith('base:'):
                view=mid.split(':',1)[1]
                paths=list((BASE/'stratified-expansion-v1').glob(f'*/capture/{view}/rgb.ppm'))
            else:paths=index.get(pixel_hash(row['image_path']),[])
            paths=list(set(paths))
            if len(paths)!=1:raise ValueError(f'Source resolution is not unique: {len(paths)}')
            rgb=paths[0];capture=rgb.parent.parent;run=capture.parent;view=rgb.parent.name
            rp=bind(inputs,capture/f'{view}.json');pp=bind(inputs,run/'plan/plan.json');wp=bind(inputs,run/'plan/world.sdf')
            cp=bind(inputs,capture/'collection-receipt.json');rec=read(rp);plan=read(pp)
            if file_sha256(bind(inputs,rgb))!=rec['image_sha256'] or pixel_hash(rgb)!=pixel_hash(row['image_path']):raise ValueError('Source pixels changed')
            if file_sha256(wp)!=plan['files']['world.sdf']:raise ValueError('World identity mismatch')
            if file_sha256(bind(inputs,rec['depth_path']))!=rec['depth_sha256']:raise ValueError('Depth identity mismatch')
            mapping=instance_mapping(plan);events=[]
            for r in items:
                candidates=[(max(abs(a-b) for a,b in zip(r['bbox_xyxy'],raw_box(b))),b) for b in rec['raw_truth']['annotatedBox']]
                matches=[b for delta,b in candidates if delta<1e-4]
                if len(matches)!=1:raise ValueError('Nonunique raw box match')
                label=matches[0]['label']
                if label not in mapping or mapping[label]['category']!='switchgear':raise ValueError('Instance category mismatch')
                events.append(dict(review_id=r['review_id'],runtime_label=label,object_id=mapping[label]['object_id'],bbox_xyxy=r['bbox_xyxy']))
            entry.update(status='source_verified',source_receipt=str(rp),source_world=str(wp),source_plan=str(pp),source_image=str(rgb),source_depth=rec['depth_path'],
                world_name=plan['world_name'],actual_pose=rec['actual_pose'],annotation_mode=annotation_mode_from_world(wp),instance_mapping={str(k):v for k,v in mapping.items()},events=events)
        except (ValueError,KeyError,FileNotFoundError) as e:entry.update(status='source_blocked',reason=str(e))
        results.append(entry)
    affected={r['member_id'] for r in m['items']};selected=[r for r in pool if r['member_id'] in affected]
    keys={r.get('derivation_group') or r['lineage_id'] for r in selected}
    closure=[r['member_id'] for r in pool if (r.get('derivation_group') or r['lineage_id']) in keys]
    return save(protocol,dict(status='frozen',frames=results,pilot=PILOT,inputs=inputs,
        scope='Existing development positives only; no training, relabeling or sealed-scene access.',
        policy=dict(max_attempts=3,pose_tolerance_m=.05,attitude_tolerance_deg=1,max_skew_ms=33.334,stable_frames=3,original_rgb_requires_exact_pixels=True),
        lineage_closure=dict(pool_members=len(pool),affected_members=len(affected),closure_members=sorted(closure),
            limitation='Uses registered derivation_group else lineage_id; member-only fallback does not establish scene independence.'),
        training_admitted=False,promotable=False,unseen_scene_status='sealed_not_evaluated'))

async def command(*args,env=None,timeout=5):
    p=await asyncio.create_subprocess_exec(*args,env=env,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    try:
        out,err=await asyncio.wait_for(p.communicate(),timeout)
    except BaseException:
        if p.returncode is None:p.kill()
        await p.communicate();raise
    if p.returncode:raise RuntimeError(err.decode(errors='replace')[-2000:])
    return out.decode()

async def stop_group(p):
    if p is None:return
    try:os.killpg(p.pid,signal.SIGINT)
    except ProcessLookupError:pass
    try:await asyncio.wait_for(p.wait(),5)
    except asyncio.TimeoutError:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        await p.wait()

async def probe(frame,attempt):
    folder=OUT/'replay'/frame['review_ids'][0]/f'attempt-{attempt:02}'
    if folder.exists():raise ValueError('Attempt directory already exists')
    folder.mkdir(parents=True)
    derived=add_sensor(ET.parse(frame['source_world']));wp=folder/'world.sdf';derived.write(wp,encoding='utf-8',xml_declaration=True)
    env=dict(os.environ,GZ_PARTITION='visibility-'+uuid.uuid4().hex,GZ_IP='127.0.0.1')
    p=None;status='technical_failure';reason='';snapshot=[]
    try:
        with (folder/'simulator.log').open('x') as log:
            p=await asyncio.create_subprocess_exec('gz','sim','-s','-r',str(wp),env=env,stdout=log,stderr=asyncio.subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+45
            while time.monotonic()<deadline:
                if p.returncode is not None:raise RuntimeError('Simulator exited')
                snapshot=(await command('gz','topic','-l',env=env)).splitlines()
                if any('/diagnostic/instances' in t for t in snapshot):break
                await asyncio.sleep(1)
            else:raise TimeoutError('Segmentation topic not discovered in 45 seconds')
            status='topics_available';reason='Runtime mask decoding and alignment verification required'
    except (Exception,asyncio.CancelledError) as e:reason=f'{type(e).__name__}: {e}'
    finally:await stop_group(p)
    return save(folder/'receipt.json',dict(status=status,reason=reason,topics=snapshot,partition=env['GZ_PARTITION'],process_cleanup_complete=p is None or p.returncode is not None,
        inputs={str(x):file_sha256(x) for x in (OUT/'protocol.json',wp,folder/'simulator.log')},training_admitted=False,promotable=False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','probe']);args=parser.parse_args()
    p=prepare();print('SOURCES',[(f['review_ids'],f['status'],f.get('reason')) for f in p['frames']],flush=True)
    if args.action=='probe':
        frame=next(f for f in p['frames'] if 'T049' in f['review_ids'])
        print(asyncio.run(probe(frame,1)),flush=True)

if __name__=='__main__':main()
