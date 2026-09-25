"""Full-frame contact sheets and raw-boundary diagnostic; no admission decisions."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scripts.vision.audit_candidate29_completeness import OUT,prior,capture

def raw_boxes(message):
    result={}
    for b in message.get('annotatedBox',[]):
        k=str(b['label']);a,z=b['box']['minCorner'],b['box']['maxCorner']
        if k in result:raise ValueError('Duplicate raw label')
        result[k]=[a.get('x',0),a.get('y',0),z.get('x',0),z.get('y',0)]
    return result

def main():
    pp=OUT/'protocol.json';ep=OUT/'evidence/manifest.json';p,e=prior.read(pp),prior.read(ep);prior.verify(p);prior.verify(e)
    folder=OUT/'review-pages';folder.mkdir(exist_ok=True);paths=[pp,ep,Path(__file__).resolve()];events={x['pair_id']:x for x in e['events']}
    for start in range(0,29,4):
        page=Image.new('RGB',(1920,1120),'white');d=ImageDraw.Draw(page)
        for j,f in enumerate(p['frames'][start:start+4]):
            ev=events[f['pair_id']];ip=Path(ev.get('full',f['source_image']));paths.append(ip)
            im=Image.open(ip).convert('RGB');im.thumbnail((960,540));x=(j%2)*960;y=(j//2)*560
            page.paste(im,(x,y+20));d.text((x+5,y+3),f['pair_id']+' '+ev['status']+' membership_conflict='+str(ev.get('membership_conflict','unknown')),fill='black')
        dest=folder/f'page-{start//4+1:02}.png';page.save(dest);paths.append(dest)
    f=next(f for f in p['frames'] if f['pair_id']=='A16');source=Image.open(f['source_image']).convert('RGB');attempt=OUT/'replay/A16-original/attempt-01'
    old=raw_boxes(prior.read(f['source_receipt'])['raw_truth']);records=[]
    for n in capture.selected_indices(attempt):
        meta={k:prior.read(attempt/f'frame-{n}-{k}.json') for k in ('rgb','mask','pose','boxes','visible-boxes','depth')}
        paths += [attempt/f'frame-{n}-{k}.json' for k in meta]+[attempt/f'frame-{n}-{k}.bin' for k in ('rgb','mask','depth')]
        b=raw_boxes(meta['boxes']);mask=np.frombuffer((attempt/f'frame-{n}-mask.bin').read_bytes(),dtype='u1').reshape(1080,1920,3)
        labels=capture.base.check_mask(meta['mask'],mask,f['instance_mapping']);stamps=[capture.base.message_timestamp(meta[k]) for k in ('rgb','depth','mask','pose','boxes','visible-boxes')]
        records.append(dict(index=n,raw_box_equality=old==b,rgb_exact=source.tobytes()==(attempt/f'frame-{n}-rgb.bin').read_bytes(),
            actual_pose_pass=capture.base.pose_close(capture.base.pose_record(meta['pose']),f['actual_pose']),max_rgb_skew_ms=max(abs(t-stamps[0]) for t in stamps)*1000,
            raw_boxes=b,boundary_excess={k:dict(right=max(0,v[2]-1920),bottom=max(0,v[3]-1080)) for k,v in b.items()},
            mask_labels=sorted(int(x) for x in labels),unboxed_visible_labels=sorted({str(x) for x in labels if x not in (0,255)}-set(b)),
            visible_pixels={k:int((mask[:,:,2]==int(k)).sum()) for k in b}))
    boundary=folder/'A16-boundary.json';prior.frozen(boundary,dict(status='raw_boundary_numeric_anomaly_not_gate_pass',records=records,
        interpretation='Old and replay full box coordinates are equal, but bottom boundary exceeds height by 0.000030517578125 pixel. Consistent with floating-point representation; strict original gate remains failed. No clipping or relabeling performed.',
        inputs={str(x):prior.file_sha256(x) for x in paths}));paths.append(boundary)
    prior.frozen(folder/'manifest.json',dict(status='awaiting_explicit_full_frame_screen',inputs={str(x):prior.file_sha256(x) for x in paths}))
    print(records)

if __name__=='__main__':main()
