"""Read-only analysis of blocked capture; cannot admit data or start training."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scripts.vision.run_physical_lighting_capture_v2 import OUT, prior, base, selected_indices


def main():
    pp=OUT/'protocol.json'; cp=OUT/'capture-receipt.json'
    p=prior.read(pp); prior.verify(p); c=prior.read(cp); prior.verify(c)
    f=next(x for x in p['frames'] if x['pair_id']=='L05' and x['variant']=='original')
    rp=Path(next(x['receipt'] for x in c['results'] if x['pair_id']=='L05'))
    r=prior.read(rp); prior.verify(r)
    if r['status']!='semantic_blocked': raise ValueError('Expected semantic blocker')
    folder=rp.parent; dest=OUT/'blocked-evidence'; dest.mkdir(exist_ok=True)
    source=Image.open(f['source_image']).convert('RGB')
    old=base.box_map(prior.read(f['source_receipt'])['raw_truth'],f['instance_mapping'])
    paths=[pp,cp,rp,Path(f['source_image']),Path(f['source_receipt']),Path(f['source_plan']),Path(f['source_world']),Path(__file__).resolve()]
    rows=[]
    for n in selected_indices(folder):
        raw={k:folder/f'frame-{n}-{k}.bin' for k in ('rgb','mask')}
        meta={k:folder/f'frame-{n}-{k}.json' for k in ('rgb','depth','mask','pose','boxes','visible-boxes')}
        paths.extend([*raw.values(),*meta.values()])
        rgb=np.frombuffer(raw['rgb'].read_bytes(),dtype='u1').reshape(1080,1920,3)
        mask=np.frombuffer(raw['mask'].read_bytes(),dtype='u1').reshape(1080,1920,3)
        base.check_mask(prior.read(meta['mask']),mask,f['instance_mapping'])
        boxes=base.box_map(prior.read(meta['boxes']),f['instance_mapping'])
        stamps=[base.message_timestamp(prior.read(x)) for x in meta.values()]
        pose=base.pose_record(prior.read(meta['pose']))
        added=sorted(set(boxes)-set(old)); targets=[]
        for label in added:
            ys,xs=np.where(mask[:,:,2]==int(label))
            targets.append(dict(label=label,instance=f['instance_mapping'][label],visible_pixels=len(xs),
                visible_box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None,full_box=boxes[label]))
        rows.append(dict(index=n,rgb_exact=source.tobytes()==rgb.tobytes(),pose_pass=bool(pose and base.pose_close(pose,f['actual_pose'])),
            max_rgb_skew_ms=max(abs(t-stamps[0]) for t in stamps)*1000,added=targets,lost=sorted(set(old)-set(boxes)),
            old_box_deltas={k:max(abs(a-b) for a,b in zip(old[k],boxes[k])) for k in set(old)&set(boxes)}))
        if n==selected_indices(folder)[0]:
            canvas=Image.new('RGB',(1600,1000),'white'); d=ImageDraw.Draw(canvas)
            overview=source.copy(); od=ImageDraw.Draw(overview)
            for b in old.values(): od.rectangle(b,outline='lime',width=4)
            for t in targets: od.rectangle(t['full_box'],outline='red',width=4)
            overview.thumbnail((1200,675)); canvas.paste(overview,(0,30))
            d.text((10,10),'L05 original RGB: green historical / red added full box',fill='black')
            for t in targets:
                b=t['visible_box']; cropbox=(max(0,b[0]-45),max(0,b[1]-45),min(1920,b[2]+45),min(1080,b[3]+45))
                pixels=np.array(source); sel=mask[:,:,2]==int(t['label']); pixels[sel]=(pixels[sel]*.5+np.array([255,0,255])*.5).astype('u1')
                for j,im in enumerate((source,Image.fromarray(pixels))):
                    crop=im.crop(cropbox); crop.thumbnail((390,650)); canvas.paste(crop,(j*400,730 if crop.height<260 else 330))
                # Separate full-resolution ROI avoids overlapping overview panels.
                source.crop(cropbox).save(dest/'L05-added-roi.png')
                Image.fromarray(pixels).crop(cropbox).save(dest/'L05-added-mask-roi.png')
                d.text((1210,40),str(t),fill='black')
            canvas.save(dest/'L05-overview.png')
    paths.extend(sorted(dest.glob('*.png')))
    prior.frozen(dest/'evidence.json',dict(status='semantic_blocker_evidence_not_quality_approval',member_id=f['member_id'],
        records=rows,training_ready=False,training_started=False,inputs={str(x):prior.file_sha256(x) for x in paths}))
    print(dest); print(rows)


if __name__=='__main__': main()
