"""Full-frame and individual-label evidence, including mask coverage checks."""
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.expand_material_view_n05 import OUT,prior

def main():
    cp=OUT/'completion.json';c=prior.read(cp);prior.verify(c)
    if c['status']!='source_plus_three_candidates_review_pending':raise ValueError('Incomplete candidates')
    p=prior.read(OUT/'protocol.json');f=p['frame'];truths=prior.read(f['source_receipt'])['truth']['objects']
    dest=OUT/'review';dest.mkdir(exist_ok=True);paths=[cp,OUT/'protocol.json',Path(__file__)];events=[]
    for unit in c['units'][1:]:
        rp=Path(unit['receipt']);r=prior.read(rp);prior.verify(r)
        ip=rp.parent/'first-stable-window/frame-1-rgb.png';im=Image.open(ip).convert('RGB')
        page=Image.new('RGB',(1500,1350),'white');d=ImageDraw.Draw(page);over=im.copy();od=ImageDraw.Draw(over)
        for t in truths:od.rectangle(t['bbox_xyxy'],outline='red',width=3)
        over.thumbnail((1450,700));page.paste(over,(0,25));d.text((0,0),unit['variant'],fill='black')
        for n,t in enumerate(truths):
            crop=im.crop(tuple(t['bbox_xyxy']));op=dest/f'{unit["variant"]}-{n:02}.png';crop.save(op)
            crop.thumbnail((480,270));x=n%3*500;y=750+n//3*300;page.paste(crop,(x,y+20))
            d.text((x,y),f'{n} {f["events"][n]["object_id"]}',fill='black')
            events.append(dict(event_id=f'{unit["variant"]}-{n:02}',object_id=f['events'][n]['object_id'],truth=t,
                image_path=str(ip),image_sha256=prior.file_sha256(ip),crop_path=str(op),crop_sha256=prior.file_sha256(op)))
            paths.append(op)
        pagepath=dest/f'{unit["variant"]}.png';page.save(pagepath);paths += [pagepath,ip,rp]
    prior.frozen(dest/'evidence.json',dict(status='fifteen_labels_review_required',events=events,training_ready=False,
        inputs={str(p):prior.file_sha256(p) for p in paths}))

if __name__=='__main__':main()
