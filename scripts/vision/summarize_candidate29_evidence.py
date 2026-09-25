"""Independent raw checks and diagnostic overlays; no generated approvals."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scripts.vision.audit_candidate29_completeness import OUT,prior,capture

def analyze_raw(frame,folder):
    base=capture.base;mapping=frame['instance_mapping'];source=Image.open(frame['source_image']).convert('RGB')
    old=base.box_map(prior.read(frame['source_receipt'])['raw_truth'],mapping)
    clock=prior.read(folder/'clock-preflight.json');prior.verify(clock);fence=base.message_timestamp(clock['samples'][-1]['message'])
    indices=capture.selected_indices(folder);records=[];paths=[folder/'clock-preflight.json']
    config=prior.read(Path(frame['source_plan']).parent/'obstacles.json')
    camera=base.ET.parse(frame['source_world']).find(".//model[@name='canonical_camera']/link[@name='research_camera_link']")
    offset=list(map(float,camera.findtext('pose').split()))[:3]
    for n in indices:
        meta={k:prior.read(folder/f'frame-{n}-{k}.json') for k in ('rgb','depth','mask','pose','boxes','visible-boxes')}
        paths += [folder/f'frame-{n}-{k}.json' for k in meta]+[folder/f'frame-{n}-{k}.bin' for k in ('rgb','mask','depth')]
        stamps=[base.message_timestamp(x) for x in meta.values()];pose=base.pose_record(meta['pose'])
        skew=max(abs(t-stamps[0]) for t in stamps)*1000
        if min(stamps)<=fence or skew>33.334 or not pose or not base.pose_close(pose,frame['actual_pose']):raise ValueError('Pose/time gate failed')
        base.validate_point(pose['position'],config,role='actual carrier');base.validate_point((np.array(pose['position'])+base.rotate(pose['orientation'],offset)).tolist(),config,role='actual optical')
        rgb=(folder/f'frame-{n}-rgb.bin').read_bytes()
        if meta['rgb']['width']!=1920 or meta['rgb']['height']!=1080 or source.tobytes()!=rgb:raise ValueError('Original RGB mismatch')
        mask=np.frombuffer((folder/f'frame-{n}-mask.bin').read_bytes(),dtype='u1').reshape(1080,1920,3)
        labels=base.check_mask(meta['mask'],mask,mapping);boxes=base.box_map(meta['boxes'],mapping);base.parse_visible(meta['visible-boxes'],mapping)
        targets=[]
        for k in sorted((set(old)|set(boxes)|{str(x) for x in labels if x not in (0,255)}),key=int):
            ys,xs=np.where(mask[:,:,2]==int(k))
            targets.append(dict(label=k,instance=mapping[k],historical_box=old.get(k),replay_box=boxes.get(k),visible_pixels=len(xs),
                visible_box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None))
        records.append(dict(index=n,rgb_exact=True,pose_pass=True,max_rgb_skew_ms=skew,targets=targets,
            added=sorted(set(boxes)-set(old)),lost=sorted(set(old)-set(boxes)),unboxed=sorted({str(x) for x in labels if x not in (0,255)}-set(boxes)),
            deltas={k:max(abs(a-b) for a,b in zip(old[k],boxes[k])) for k in set(old)&set(boxes)}))
    if any(r['targets']!=records[0]['targets'] for r in records[1:]):raise ValueError('Unstable target output')
    return records,paths

def main():
    pp=OUT/'protocol.json';rp=OUT/'replay-index.json';p,r=prior.read(pp),prior.read(rp)
    prior.verify(p);prior.verify(r);folder=OUT/'evidence';folder.mkdir(exist_ok=True);events=[];paths=[pp,rp,Path(__file__).resolve()]
    frames={f['pair_id']:f for f in p['frames']}
    for item in r['results']:
        f=frames[item['pair_id']];receipt=Path(item['receipt']);prior.verify(prior.read(receipt));paths.append(receipt)
        try:records,rawpaths=analyze_raw(f,receipt.parent)
        except (ValueError,KeyError,FileNotFoundError) as exc:
            events.append(dict(pair_id=f['pair_id'],member_id=f['member_id'],status='named_evidence_gap',reason=str(exc)));continue
        paths+=rawpaths;rec=records[0];im=Image.open(f['source_image']).convert('RGB');mask=np.frombuffer((receipt.parent/f"frame-{rec['index']}-mask.bin").read_bytes(),dtype='u1').reshape(1080,1920,3)[:,:,2]
        full=im.copy();draw=ImageDraw.Draw(full)
        for t in rec['targets']:
            b=t['historical_box'] or t['visible_box']
            if b:draw.rectangle(b,outline='red' if t['historical_box'] else 'magenta',width=3);draw.text(b[:2],t['label'],fill='red')
        dest=folder/(f['pair_id']+'-full.png');full.save(dest);paths.append(dest)
        for t in rec['targets']:
            b=t['visible_box'] if t['historical_box'] is None else t['historical_box']
            if b:
                a=np.array(im);sel=mask==int(t['label']);a[sel]=(a[sel]*.6+np.array([255,0,255])*.4).astype('u1')
                crops=[x.crop(b) for x in (im,Image.fromarray(a))]
                page=Image.new('RGB',(2*crops[0].width,crops[0].height+30),'white');d=ImageDraw.Draw(page);d.text((3,3),f['pair_id']+' '+t['label']+' '+t['instance']['object_id'],fill='black')
                for j,crop in enumerate(crops):page.paste(crop,(j*crop.width,30))
                cp=folder/(f['pair_id']+'-'+t['label']+'.png');page.save(cp);paths.append(cp);t['crop']=str(cp)
        events.append(dict(pair_id=f['pair_id'],member_id=f['member_id'],status='aligned_raw_evidence',full=str(dest),records=records,
            membership_conflict=bool(rec['added'] or rec['lost'] or rec['unboxed'] or any(d>1 for d in rec['deltas'].values()))))
    prior.frozen(folder/'manifest.json',dict(status='raw_evidence_not_quality_approval',events=events,training_ready=False,
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print([(x['pair_id'],x.get('membership_conflict'),x.get('reason')) for x in events]);print('targets',sum(len(x['records'][0]['targets']) for x in events if 'records' in x))

if __name__=='__main__':main()
