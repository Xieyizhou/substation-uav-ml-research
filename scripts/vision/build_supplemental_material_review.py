"""Evidence pages and own-label crops for sixteen newly rendered variants."""
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.capture_supplemental_material_triplets import OUT,prior

def main():
    cp=OUT/'completion.json';c=prior.read(cp);prior.verify(c)
    if c['status']!='sixteen_variants_captured_review_pending':raise ValueError('Incomplete variant matrix')
    protocol=prior.read(OUT/'protocol.json');planned={r['key']:r for r in protocol['units']}
    dest=OUT/'review';dest.mkdir(exist_ok=True);paths=[cp,OUT/'protocol.json',Path(__file__)];events=[]
    for unit in c['units']:
        row=planned[unit['variant']];f=row['frame'];rp=Path(unit['receipt']);r=prior.read(rp);prior.verify(r)
        truths=prior.read(f['source_receipt'])['truth']['objects'];ip=rp.parent/'first-stable-window/frame-1-rgb.png'
        im=Image.open(ip).convert('RGB');count=len(truths);page=Image.new('RGB',(1500,750+300*((count+2)//3)),'white');d=ImageDraw.Draw(page);over=im.copy();od=ImageDraw.Draw(over)
        for t in truths:od.rectangle(t['bbox_xyxy'],outline='red',width=3)
        over.thumbnail((1450,700));page.paste(over,(0,25));d.text((0,0),unit['variant'],fill='black')
        for n,t in enumerate(truths):
            eid=f'{row["probe_id"]}-{row["variant"]}-{n:02}';op=dest/(eid+'.png');crop=im.crop(tuple(t['bbox_xyxy']));crop.save(op);paths.append(op)
            crop.thumbnail((480,270));x=n%3*500;y=750+n//3*300;d.text((x,y),eid+' '+f['events'][n]['object_id'],fill='black');page.paste(crop,(x,y+20))
            events.append(dict(event_id=eid,probe_id=row['probe_id'],variant=row['variant'],object_id=f['events'][n]['object_id'],runtime_label=f['events'][n]['runtime_label'],truth=t,
                image_path=str(ip),image_sha256=prior.file_sha256(ip),crop_path=str(op),crop_sha256=prior.file_sha256(op)))
        pp=dest/f'{row["probe_id"]}-{row["variant"]}.png';page.save(pp);paths += [pp,ip,rp]
    prior.frozen(dest/'evidence.json',dict(status='44_explicit_label_reviews_required',events=events,training_ready=False,inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('PAGES16 LABELS',len(events))

if __name__=='__main__':main()
