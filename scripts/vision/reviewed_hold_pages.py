"""Hash-bound full-image/instance/crop pages; no review decisions."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scripts.vision.reanalyze_reviewed_hold import OUT as ANALYSIS,prior
from scripts.vision.reviewed_hold_control import OUT


def main():
    dest=OUT/'review-evidence.json'
    if dest.exists():prior.verify(prior.read(dest));return
    cp=ANALYSIS/'completion.json';prior.verify(prior.read(cp));protocol=prior.read(ANALYSIS/'protocol.json');frames={f['review_ids'][0]:f for f in protocol['frames']};paths=[cp,ANALYSIS/'protocol.json',Path(__file__)];events=[]
    for unit in prior.read(cp)['results']:
        rp=Path(unit['analysis']);r=prior.read(rp);prior.verify(r);f=frames[unit['review_id']]
        if r['status']!='existing_pose_technical_checks_passed':raise ValueError('Replay blocked')
        for row in r['records']:
            if row['lost_labels'] or row['added_labels'] or row['unboxed_visible_labels'] or max(row['historical_deltas'].values())>1:raise ValueError('Full image completeness conflict')
        row=r['records'][0];n=row['capture_index'];rgb=Image.open(rp.parent/f'frame-{n}-rgb.png').convert('RGB');mask=np.frombuffer((rp.parent/f'frame-{n}-mask.bin').read_bytes(),dtype='u1').reshape(1080,1920,3)
        labels=list(row['full_boxes']);canvas=Image.new('RGB',(1200,760+((len(labels)+2)//3)*280),'white');full=rgb.copy();draw=ImageDraw.Draw(full)
        for k,b in row['full_boxes'].items():draw.rectangle(b,outline='red',width=3);draw.text(tuple(b[:2]),k,fill='yellow')
        full.thumbnail((1200,675));canvas.paste(full,(0,30));ds=[]
        for i,k in enumerate(labels):
            b=row['full_boxes'][k];box=(max(0,int(b[0])-12),max(0,int(b[1])-12),min(1920,int(b[2])+13),min(1080,int(b[3])+13))
            crop=rgb.crop(box);crop.thumbnail((390,220));x=(i%3)*400;y=740+(i//3)*280;canvas.paste(crop,(x,y+30))
            ys,xs=np.where(mask[:,:,2]==int(k));pixels=len(xs)
            ImageDraw.Draw(canvas).text((x,y),k+' '+f['instance_mapping'][k]['object_id']+' px='+str(pixels),fill='black')
            ds.append(dict(runtime_label=k,object_id=f['instance_mapping'][k]['object_id'],bbox=b,visible_pixels=pixels))
        page=OUT/(unit['review_id']+'-page.png');ImageDraw.Draw(canvas).text((0,5),unit['review_id']+' full RGB + all existing label crops',fill='black');canvas.save(page)
        events.append(dict(review_id=unit['review_id'],member_id=f['member_id'],labels=ds,page=str(page),page_sha256=prior.file_sha256(page),analysis=str(rp)))
        paths += [rp,page]
    prior.frozen(dest,dict(status='seven_pages_ready_for_explicit_review',events=events,training_ready=False,training_started=False,inputs={str(p):prior.file_sha256(p) for p in paths}))


if __name__=='__main__':main()
