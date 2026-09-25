"""Full label identity and pairing evidence, with no automatic review approval."""
import math
import re
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image,ImageDraw
from scripts.vision.closed_exterior_material_capture import OUT,freeze,prior,read_record
from src.vision.canonical.gates import validate_preflight,validate_point
from src.vision.canonical.recovery import resumed_views
from src.vision.canonical.plan import pose_close,rotate


def verified_run(run):
    plan=read_record(run['plan_path']);base=Path(run['plan_path']).parent
    gate,cfg,mapping=validate_preflight(plan,base,plan['calibration_views'])
    cp=OUT/'captures'/run['key']/'collection-receipt.json';receipt=read_record(cp)
    if receipt['status']!='complete_pending_review' or receipt['world_sha256']!=gate['world_sha256']:raise ValueError('invalid_capture_run')
    rows,_=resumed_views(cp,plan,'calibration',plan['calibration_views'],config=cfg,check_version=gate['check_version'],instance_mapping=mapping)
    if len(rows)!=8:raise ValueError('missing_capture_views')
    ext=list(map(float,ET.parse(base/'world.sdf').findtext('.//link[@name="research_camera_link"]/pose').split()))[:3]
    for row in rows:
        v=next(v for v in plan['calibration_views'] if v['view_id']==row['view_id'])
        if not pose_close(row['actual_pose'],v):raise ValueError('pose_drift')
        delta=rotate(row['actual_pose']['orientation'],ext)
        optical=[a+b for a,b in zip(row['actual_pose']['position'],delta)]
        validate_point(optical,cfg,role='actual_optical_center')
        ts=[row[k] for k in ('rgb_timestamp','depth_timestamp','truth_timestamp')]+[row['actual_pose']['timestamp']]
        if max(abs(t-ts[0]) for t in ts)>.033334+1e-9:raise ValueError('sync_drift')
    return {r['view_id']:r for r in rows},gate['instance_mapping'],cp


def build():
    p=freeze();pairs={};runs={};paths=[OUT/'protocol.json',Path(__file__)]
    for run in p['runs']:
        if not (OUT/'captures'/run['key']/'collection-receipt.json').exists():continue
        runs[run['key']]=verified_run(run)
    for u in p['units']:
        if u['run_key'] not in runs:continue
        rows,mapping,cp=runs[u['run_key']];row=rows[u['view_id']]
        folder=OUT/'evidence'/u['key'];folder.mkdir(parents=True,exist_ok=True)
        ep=folder/'evidence.json'
        if ep.exists():e=prior.read(ep);prior.verify(e)
        else:
            image=Image.open(row['rgb_path']).convert('RGB');drawn=image.copy();d=ImageDraw.Draw(drawn);events=[];deps=[cp,Path(row['rgb_path'])]
            seen=set()
            for n,t in enumerate(row['truth']['objects']):
                label=str(int(re.search(r'instance-(\d+)-',t['annotation_id'])[1]));identity=mapping[label]
                if label in seen or identity['category']!=t['class_name']:raise ValueError('instance_collision_or_class_conflict')
                seen.add(label);b=t['bbox_xyxy'];d.rectangle(b,outline='red',width=4)
                d.text((b[0],max(0,b[1]-18)),f'{n} '+identity['object_id'],fill='red',stroke_width=1)
                crop=folder/f'box-{n:02}.png';image.crop((max(0,math.floor(b[0])-12),max(0,math.floor(b[1])-12),min(1920,math.ceil(b[2])+12),min(1080,math.ceil(b[3])+12))).save(crop);deps.append(crop)
                events.append(dict(event_id=u['key']+':'+identity['object_id'],object_id=identity['object_id'],runtime_label=label,
                    truth=t,crop_path=str(crop),crop_sha256=prior.file_sha256(crop),planned=identity['object_id']==u['object_id']))
            full=folder/'full.png';drawn.save(full);deps.append(full)
            e=prior.frozen(ep,dict(unit=u,events=events,image_path=row['rgb_path'],image_sha256=row['image_sha256'],
                full_truth=row['truth'],full_image=str(full),actual_pose=row['actual_pose'],instance_mapping=mapping,
                source_receipt=str(cp),review_status='pending',training_ready=False,
                inputs={str(path):prior.file_sha256(path) for path in deps+[Path(__file__)]}))
        pairs.setdefault(u['pair_id'],{})[u['variant']]=e;paths.append(ep)
    checks=[]
    for pair,vs in pairs.items():
        if set(vs)!=set(('original','warm','cool','neutral')):continue
        ref={e['object_id']:e['truth'] for e in vs['original']['events']};drifts=[]
        card=Image.new('RGB',(1920,1480),'white');d=ImageDraw.Draw(card)
        for i,variant in enumerate(('original','warm','cool','neutral')):
            e=vs[variant];cur={x['object_id']:x['truth'] for x in e['events']}
            if ref.keys()!=cur.keys():raise ValueError('paired_instance_membership_changed:'+pair)
            drift=max(abs(a-b) for k in ref for a,b in zip(ref[k]['bbox_xyxy'],cur[k]['bbox_xyxy']));drifts.append(drift)
            if drift>1:raise ValueError('paired_boxes_drift:'+pair)
            full=Image.open(e['full_image']).resize((960,540));x,y=(i%2)*960,(i//2)*570
            d.text((x+8,y+5),pair+' '+variant,fill='black');card.paste(full,(x,y+24))
            target=next(x for x in e['events'] if x['planned']);crop=Image.open(target['crop_path']);crop.thumbnail((475,300));card.paste(crop,(i*480,1170));d.text((i*480,1145),variant+' target',fill='black')
        cardpath=OUT/'evidence'/(pair+'.png')
        if not cardpath.exists():card.save(cardpath)
        paths.append(cardpath);checks.append(dict(pair_id=pair,card=str(cardpath),max_box_delta=max(drifts),instance_ids=sorted(ref)))
        print('PAIR',pair,'labels',len(ref),'drift',max(drifts),str(cardpath),flush=True)
    if len(checks)==16:
        dest=OUT/'evidence/completion.json'
        if not dest.exists():prior.frozen(dest,dict(status='64_evidence_ready_review_pending',pairs=checks,frames=64,
            label_events=sum(len(v['events']) for vs in pairs.values() for v in vs.values()),
            inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__=='__main__':build()
