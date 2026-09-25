"""Capture revalidation, full-box evidence and exact low-light frame replay."""
import argparse,asyncio,math,re,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image,ImageDraw
from scripts.vision.cool_light_capture import OUT,freeze,prior,lighting_only
from scripts.vision.verify_designed_full_scene_poses import replay_frame,DESIGN
from scripts.vision.export_material_candidate_batch import label_text
from src.vision.canonical.plan import read_record,pose_close,rotate
from src.vision.canonical.gates import validate_preflight,validate_point
from src.vision.canonical.recovery import resumed_views

def verify_capture(u):
    pp=Path(u['plan_path']);p=read_record(pp);gate,cfg,mapping=validate_preflight(p,pp.parent,p['calibration_views'])
    lighting_only(ET.parse(Path(u['source_plan']).parent/'world.sdf').getroot(),ET.parse(pp.parent/'world.sdf').getroot())
    cp=OUT/'captures'/u['unit_id']/'collection-receipt.json';c=read_record(cp)
    if c['status']!='complete_pending_review' or c['world_sha256']!=gate['world_sha256']:raise ValueError('Capture identity failure')
    rows,_=resumed_views(cp,p,'calibration',p['calibration_views'],config=cfg,check_version=gate['check_version'],instance_mapping=mapping)
    if len(rows)!=1 or len({x['object_id'] for x in mapping.values()})!=len(mapping):raise ValueError('Missing frame or mapping collision')
    row=rows[0]
    if not pose_close(row['actual_pose'],u['view']):raise ValueError('Actual pose failure')
    ext=list(map(float,ET.parse(pp.parent/'world.sdf').findtext('.//link[@name="research_camera_link"]/pose').split()))[:3]
    optical=[a+b for a,b in zip(row['actual_pose']['position'],rotate(row['actual_pose']['orientation'],ext))]
    validate_point(optical,cfg,role='actual_optical_center');validate_point(row['actual_pose']['position'],cfg,role='actual_carrier')
    stamps=[row[k] for k in ('rgb_timestamp','depth_timestamp','truth_timestamp')]+[row['actual_pose']['timestamp']]
    if max(abs(t-stamps[0]) for t in stamps)>.033334+1e-9:raise ValueError('Synchronization failure')
    return row,{str(k):v for k,v in mapping.items()},cp

def evidence(u):
    row,mapping,cp=verify_capture(u);folder=OUT/'evidence'/u['unit_id'];folder.mkdir(parents=True,exist_ok=True);dest=folder/'evidence.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    source=prior.read(u['source_member']['evidence_path']);prior.verify(source)
    reference={x['object_id']:x['truth'] for x in source['events']};events=[];seen=set()
    im=Image.open(row['rgb_path']).convert('RGB');full=im.copy();d=ImageDraw.Draw(full);deps=[cp,Path(row['rgb_path']),Path(row['depth_path']),Path(u['plan_path']),Path(u['source_member']['evidence_path']),Path(__file__).resolve()]
    for n,t in enumerate(row['truth']['objects']):
        match=re.search(r'instance-(\d+)-',t['annotation_id'])
        if not match:raise ValueError('Unparsed instance')
        label=str(int(match[1]));identity=mapping[label];name=identity['object_id']
        if name in seen or identity['category']!=t['class_name'] or name not in reference:raise ValueError('Identity or full supervision conflict')
        seen.add(name);b=t['bbox_xyxy'];delta=max(abs(a-z) for a,z in zip(b,reference[name]['bbox_xyxy']))
        if delta>1:raise ValueError('Pair alignment exceeds one pixel')
        d.rectangle(b,outline='red',width=4);d.text((b[0],max(0,b[1]-20)),f'{n} '+name,fill='red',stroke_width=1)
        crop=folder/f'box-{n:02}.png';im.crop((max(0,math.floor(b[0])-12),max(0,math.floor(b[1])-12),min(im.width,math.ceil(b[2])+12),min(im.height,math.ceil(b[3])+12))).save(crop);deps.append(crop)
        events.append(dict(event_id=u['unit_id']+':'+name,object_id=name,runtime_label=label,truth=t,max_pair_box_delta=delta,crop_path=str(crop),crop_sha256=prior.file_sha256(crop)))
    if seen!=set(reference):raise ValueError('Missing corresponding instance')
    exact=label_text(row['truth'],*im.size)==Path(u['source_member']['label_path']).read_text()
    if not exact:raise ValueError('Full paired exported labels differ; investigate before training')
    fp=folder/'full.png';full.save(fp);deps.append(fp)
    card=Image.new('RGB',(1440,600+240*math.ceil(len(events)/3)),'white');cd=ImageDraw.Draw(card)
    cd.text((5,5),u['unit_id']+' '+u['source_member']['pair_id']+' '+u['source_member']['variant']+' lower_light',fill='black');card.paste(full.resize((960,540)),(0,35))
    refim=Image.open(u['source_member']['image_path']).convert('RGB');refim.thumbnail((470,300));card.paste(refim,(965,35));cd.text((965,15),'original lighting reference',fill='black')
    for n,e in enumerate(events):
        crop=Image.open(e['crop_path']);crop.thumbnail((470,210));x=(n%3)*480;y=600+(n//3)*240;card.paste(crop,(x,y+20));cd.text((x,y),e['object_id'],fill='black')
    cpimage=folder/'card.png';card.save(cpimage);deps.append(cpimage)
    return prior.frozen(dest,dict(status='captured_full_label_evidence_pending_review',unit_id=u['unit_id'],source_member_id=u['source_member']['member_id'],pair_id=u['source_member']['pair_id'],variant=u['source_member']['variant'],
        events=events,card_path=str(cpimage),image_path=row['rgb_path'],image_sha256=row['image_sha256'],full_truth=row['truth'],actual_pose=row['actual_pose'],instance_mapping=mapping,
        full_paired_labels_exact=exact,pixel_visibility_certified=False,inputs={str(p):prior.file_sha256(p) for p in deps}))

async def replay_unit(u):
    e=evidence(u);row,mapping,cp=verify_capture(u);unit=OUT/'replays'/u['unit_id'];unit.mkdir(parents=True,exist_ok=True);sp=unit/'source.json'
    if not sp.exists():prior.frozen(sp,dict(row,inputs={str(cp):prior.file_sha256(cp)}))
    helper=unit/'gz_visibility_capture_cleanup_fixed'
    if not helper.exists():shutil.copy2(DESIGN.parent/'native-source-replay-v2/gz_visibility_capture_cleanup_fixed',helper)
    f=dict(member_id=u['unit_id'],lineage_id=u['source_member']['pair_id'],class_name=u['source_member']['planned_category'],review_ids=[u['unit_id']],source_image=row['rgb_path'],source_receipt=str(sp),source_plan=u['plan_path'],source_world=str(Path(u['plan_path']).parent/'world.sdf'),actual_pose=row['actual_pose'],world_name=read_record(u['plan_path'])['world_name'],instance_mapping=mapping,
        events=[dict(review_id=x['event_id'],runtime_label=int(x['runtime_label']),object_id=x['object_id'],bbox_xyxy=x['truth']['bbox_xyxy']) for x in e['events']])
    pp=unit/'protocol.json'
    if not pp.exists():prior.frozen(pp,dict(frame=f,inputs={str(x):prior.file_sha256(x) for x in (sp,helper,OUT/'evidence'/u['unit_id']/'evidence.json',Path(__file__).resolve())}))
    prior.verify(prior.read(pp));rp,r=await replay_frame(f,unit)
    if not r or r['status']!='original_pixel_evidence_certified' or any(x['missing_targets'] for x in r['full_mask_coverage']):raise ValueError('Low-light replay blocked '+u['unit_id']+' '+str(r.get('reason') if r else 'missing'))
    dest=unit/'completion.json'
    if not dest.exists():prior.frozen(dest,dict(status='low_light_capture_exact_replay_verified',receipt_path=str(rp),inputs={str(x):prior.file_sha256(x) for x in (rp,pp)}))
    else:prior.verify(prior.read(dest))
    print('REPLAY_VERIFIED',u['unit_id'],len(e['events']),'labels',flush=True)

async def run(pilot=False):
    p=freeze()
    for u in p['units']:
        if pilot and u['unit_id'] not in p['pilot_units']:continue
        await replay_unit(u)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');ap.add_argument('--pilot',action='store_true');a=ap.parse_args()
    if a.replay:asyncio.run(run(a.pilot))
    else:
        p=freeze()
        for u in p['units']:
            if (OUT/'captures'/u['unit_id']/'collection-receipt.json').exists():print(evidence(u)['unit_id'])
