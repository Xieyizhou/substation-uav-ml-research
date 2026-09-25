"""Evidence generation and alignment checks only. No automatic review decisions."""
import math
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from scripts.vision.source_isolated_material_capture import OUT, freeze, verify_capture, prior, read_record
from src.vision.canonical.plan import pose_close, rotate
from src.vision.canonical.gates import validate_point
import json
import xml.etree.ElementTree as ET


def frame_evidence(unit):
    rp=OUT/'captures'/unit['key']/'collection-receipt.json'
    row=verify_capture(unit,rp); receipt=read_record(rp)
    plan=read_record(unit['plan_path']); folder=Path(unit['plan_path']).parent
    view=next(v for v in plan['calibration_views'] if v['view_id']==unit['view_id'])
    if not pose_close(row['actual_pose'],view):raise ValueError('actual_pose_out_of_tolerance')
    config=json.loads((folder/'obstacles.json').read_text())
    extrinsic=list(map(float,ET.parse(folder/'world.sdf').findtext('.//link[@name="research_camera_link"]/pose').split()))[:3]
    delta=rotate(row['actual_pose']['orientation'],extrinsic)
    optical=[a+b for a,b in zip(row['actual_pose']['position'],delta)]
    validate_point(optical,config,role='actual_optical_recomputed')
    stamps=[row[k] for k in ('rgb_timestamp','depth_timestamp','truth_timestamp')]+[row['actual_pose']['timestamp']]
    if max(abs(stamps[0]-t) for t in stamps)>0.033334+1e-9:raise ValueError('sync_violation')
    dest=OUT/'evidence'/unit['key'];dest.mkdir(parents=True,exist_ok=True)
    image=Image.open(row['rgb_path']).convert('RGB');annotated=image.copy();d=ImageDraw.Draw(annotated)
    events=[];paths=[rp,Path(row['rgb_path']),Path(row['depth_path'])]
    mapping=receipt['collection_checks']['instance_mapping'];seen=set()
    for n,truth in enumerate(row['truth']['objects']):
        label=re.search(r'instance-(\d+)-',truth['annotation_id'])[1];label=str(int(label))
        identity=mapping[label]
        if label in seen or identity['category']!=truth['class_name']:raise ValueError('nonunique_or_wrong_class_instance')
        seen.add(label);box=truth['bbox_xyxy'];name=identity['object_id']
        d.rectangle(box,outline='red',width=4);d.text((box[0],max(0,box[1]-20)),str(n)+':'+name,fill='red',stroke_width=1)
        crop=dest/f'box-{n:02}.png'
        if not crop.exists():image.crop((max(0,math.floor(box[0])-15),max(0,math.floor(box[1])-15),min(1920,math.ceil(box[2])+15),min(1080,math.ceil(box[3])+15))).save(crop)
        paths.append(crop)
        events.append(dict(event_id=unit['key']+':'+name,object_id=name,runtime_label=label,truth=truth,crop_path=str(crop),
                           crop_sha256=prior.file_sha256(crop),planned=name==row['expected_object_id']))
    full=dest/'full.png'
    if not full.exists():annotated.save(full)
    paths.append(full)
    card=Image.new('RGB',(1440,860),'white');cd=ImageDraw.Draw(card);cd.text((10,8),unit['key'],fill='black')
    card.paste(annotated.resize((960,540)),(0,40))
    for n,e in enumerate(events):
        if n>=6:break
        crop=Image.open(e['crop_path']);crop.thumbnail((470,250))
        x,y=(960,40+n*270) if n<3 else ((n-3)*480,590)
        card.paste(crop,(x,y+18));cd.text((x,y),str(n)+':'+e['object_id'],fill='black')
    cardpath=dest/'card.png'
    if not cardpath.exists():card.save(cardpath)
    paths.append(cardpath)
    ep=dest/'evidence.json'
    if ep.exists():record=prior.read(ep);prior.verify(record);return record
    return prior.frozen(ep,dict(unit=unit,events=events,card=str(cardpath),full_image=str(full),
        actual_pose_checked=True,actual_optical_center=optical,max_rgb_skew_ms=1000*max(abs(stamps[0]-t) for t in stamps),
        unboxed_instance_review='pending',pixel_visibility_certified=False,
        inputs={str(p):prior.file_sha256(p) for p in paths+[Path(__file__)]}))


def main():
    p=freeze();rows=[];pairs={}
    for unit in (x for x in p['units'] if x['pilot']):
        rp=OUT/'captures'/unit['key']/'collection-receipt.json'
        if not rp.exists():continue
        e=frame_evidence(unit);rows.append(e);pairs.setdefault(unit['pair_id'],{})[unit['variant']]=e
        print(unit['key'],len(e['events']),e['card'])
    checks=[]
    for pair, variants in pairs.items():
        if set(variants)!={'original','warm','cool','neutral'}:continue
        ref={e['object_id']:e['truth'] for e in variants['original']['events']}
        for variant,record in variants.items():
            cur={e['object_id']:e['truth'] for e in record['events']}
            if cur.keys()!=ref.keys():raise ValueError('pair_instance_membership_changed:'+pair)
            drift=max((abs(a-b) for key in ref for a,b in zip(ref[key]['bbox_xyxy'],cur[key]['bbox_xyxy'])),default=0)
            checks.append(dict(pair_id=pair,variant=variant,max_box_difference_px=drift,passed=drift<=1))
    if len(rows)==16:
        dest=OUT/'evidence/completion.json'
        if not dest.exists():prior.frozen(dest,dict(status='evidence_complete_review_pending',frames=16,pair_checks=checks,
            inputs={str(OUT/'evidence'/r['unit']['key']/'evidence.json'):prior.file_sha256(OUT/'evidence'/r['unit']['key']/'evidence.json') for r in rows}))


if __name__=='__main__':main()
