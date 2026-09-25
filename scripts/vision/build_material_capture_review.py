"""Render own-box review evidence; no generated review decisions."""
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.capture_material_view_pilot import OUT,prior

def main():
    cp=OUT/'completion.json';p=prior.read(cp);prior.verify(p)
    if p['status']!='three_candidates_captured_review_required':raise ValueError('Incomplete triplet')
    frame=prior.read(OUT/'protocol.json')['frame'];source=prior.read(frame['source_receipt'])
    dest=OUT/'review';dest.mkdir(exist_ok=True);rows=[];paths=[cp,Path(__file__)]
    for unit in p['units']:
        rp=Path(unit['receipt']);r=prior.read(rp);prior.verify(r)
        ip=rp.parent/'first-stable-window/frame-1-rgb.png';im=Image.open(ip).convert('RGB')
        page=Image.new('RGB',(1500,1050),'white');draw=ImageDraw.Draw(page)
        over=im.copy();od=ImageDraw.Draw(over)
        for n,t in enumerate(source['truth']['objects']):od.rectangle(t['bbox_xyxy'],outline='red',width=3)
        over.thumbnail((1500,700));page.paste(over,(0,20));draw.text((0,0),unit['variant'],fill='black')
        for n,t in enumerate(source['truth']['objects']):
            crop=im.crop(tuple(t['bbox_xyxy']));op=dest/f'{unit["variant"]}-{n:02}.png';crop.save(op)
            crop.thumbnail((480,260));page.paste(crop,(n*500,770));draw.text((n*500,745),f'{n}: {frame["events"][n]["object_id"]}',fill='black')
            rows.append(dict(event_id=f'{unit["variant"]}-{n:02}',truth=t,object_id=frame['events'][n]['object_id'],
                image_path=str(ip),crop_path=str(op),image_sha256=prior.file_sha256(ip),crop_sha256=prior.file_sha256(op)))
            paths.append(op)
        op=dest/f'{unit["variant"]}.png';page.save(op);paths += [op,ip,rp]
    prior.frozen(dest/'evidence.json',dict(status='nine_labels_review_required',events=rows,training_ready=False,
        inputs={str(p):prior.file_sha256(p) for p in paths}))

if __name__=='__main__':main()
