"""Record a new semantic risk after the one allowed solve, without resolving again."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scripts.vision.replay_l05_increase_queue import OUT as REPLAY,DESIGN,prior,capture

def main():
    pp=REPLAY/'protocol.json';cp=REPLAY/'receipt.json';p,c=prior.read(pp),prior.read(cp)
    for x in (p,c):prior.verify(x)
    result=c['results'][-1]
    if result['pair_id']!='I17' or result['status']!='semantic_blocked':raise ValueError('Unexpected blocker')
    rp=Path(result['receipt']);prior.verify(prior.read(rp));folder=rp.parent
    f=next(f for f in p['frames'] if f['pair_id']=='I17');mapping=f['instance_mapping'];source=Image.open(f['source_image']).convert('RGB')
    old=capture.base.box_map(prior.read(f['source_receipt'])['raw_truth'],mapping)
    dest=DESIGN/'new-risk';dest.mkdir(exist_ok=True)
    paths=[pp,cp,rp,Path(f['source_image']),Path(f['source_receipt']),Path(f['source_plan']),Path(f['source_world']),Path(__file__).resolve()];records=[]
    for n in capture.selected_indices(folder):
        metas={k:folder/f'frame-{n}-{k}.json' for k in ('rgb','depth','mask','pose','boxes','visible-boxes')}
        rb=folder/f'frame-{n}-rgb.bin';mb=folder/f'frame-{n}-mask.bin';paths += [*metas.values(),rb,mb]
        mask=np.frombuffer(mb.read_bytes(),dtype='u1').reshape(1080,1920,3);capture.base.check_mask(prior.read(metas['mask']),mask,mapping)
        boxes=capture.base.box_map(prior.read(metas['boxes']),mapping);stamps=[capture.base.message_timestamp(prior.read(x)) for x in metas.values()]
        targets=[]
        for k in sorted(set(boxes)-set(old)):
            ys,xs=np.where(mask[:,:,2]==int(k));targets.append(dict(label=k,instance=mapping[k],visible_pixels=len(xs),full_box=boxes[k],visible_box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None))
        records.append(dict(index=n,rgb_exact=source.tobytes()==rb.read_bytes(),pose_pass=capture.base.pose_close(capture.base.pose_record(prior.read(metas['pose'])),f['actual_pose']),max_rgb_skew_ms=max(abs(t-stamps[0]) for t in stamps)*1000,
            added=targets,lost=sorted(set(old)-set(boxes)),old_deltas={k:max(abs(a-b) for a,b in zip(old[k],boxes[k])) for k in set(old)&set(boxes)}))
        if n==capture.selected_indices(folder)[0]:
            full=source.copy();draw=ImageDraw.Draw(full)
            for b in old.values():draw.rectangle(b,outline='lime',width=4)
            for t in targets:draw.rectangle(t['visible_box'],outline='magenta',width=4)
            full.save(dest/'I17-full-visible-overlay.png')
            for t in targets:
                b=t['visible_box'];crop=(max(0,b[0]-40),max(0,b[1]-40),min(1920,b[2]+40),min(1080,b[3]+40))
                a=np.array(source);sel=mask[:,:,2]==int(t['label']);a[sel]=(a[sel]*.6+np.array([255,0,255])*.4).astype('u1')
                source.crop(crop).save(dest/'I17-roi.png');Image.fromarray(a).crop(crop).save(dest/'I17-mask-roi.png')
    counts=prior.read(DESIGN/'counts.json');prior.verify(counts);paths.append(DESIGN/'counts.json')
    impact={seed:next(x for x in r['changes'] if x['member_id']==f['member_id']) for seed,r in counts['results'].items()}
    paths += sorted(dest.glob('*.png'))
    prior.frozen(dest/'evidence.json',dict(status='new_quality_risk_after_single_solve',member_id=f['member_id'],records=records,impact=impact,
        technical_pass_count=sum(x['status']=='capture_technical_checks_passed' for x in c['results']),not_run_members=[x['member_id'] for x in p['frames'] if x['member_id'] not in {r['member_id'] for r in c['results']}],
        training_ready=False,training_started=False,inputs={str(x):prior.file_sha256(x) for x in paths}))
    print(records);print(impact)

if __name__=='__main__':main()
