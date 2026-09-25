"""Produce five full-frame paired review pages, never approvals."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scripts.vision.run_physical_lighting_capture_v2 import OUT,prior

def main():
    cp=OUT/'capture-receipt.json';c=prior.read(cp);prior.verify(c)
    if c['status']!='all_ten_captured_review_pending':raise ValueError('Capture blocked; no complete review set')
    p=prior.read(OUT/'protocol.json');prior.verify(p);folder=OUT/'review';folder.mkdir(exist_ok=True)
    paths=[cp,OUT/'protocol.json',Path(__file__).resolve()];events=[]
    for pid in sorted({f['pair_id'] for f in p['frames']}):
        arms={};mapping=next(f['instance_mapping'] for f in p['frames'] if f['pair_id']==pid)
        for variant in ('original','physical-lighting'):
            item=next(r for r in c['results'] if r['pair_id']==pid and r['variant']==variant);rp=Path(item['receipt']);r=prior.read(rp);prior.verify(r)
            row=r['records'][0];n=row['capture_index'];ip=rp.parent/f'frame-{n}-rgb.png';mp=rp.parent/f'frame-{n}-mask.bin';paths += [rp,ip,mp]
            arms[variant]=dict(receipt=str(rp),image=str(ip),image_sha256=prior.file_sha256(ip),mask=str(mp),mask_sha256=prior.file_sha256(mp),record=row)
        orig=Image.open(arms['original']['image']).convert('RGB');dark=Image.open(arms['physical-lighting']['image']).convert('RGB')
        if orig.tobytes()==dark.tobytes():raise ValueError('Lighting treatment produced no RGB change')
        boxes=arms['original']['record']['full_boxes'];labels=sorted(boxes)
        canvas=Image.new('RGB',(1600,520+300*((len(labels)+1)//2)),'white');draw=ImageDraw.Draw(canvas)
        for a,variant in enumerate(('original','physical-lighting')):
            im=Image.open(arms[variant]['image']).convert('RGB');d=ImageDraw.Draw(im)
            for lab,b in boxes.items():d.rectangle(b,outline='red',width=3);d.text(b[:2],lab,fill='red')
            im.thumbnail((800,450));canvas.paste(im,(a*800,30));draw.text((a*800,5),pid+' '+variant,fill='black')
        targets=[]
        for j,label in enumerate(labels):
            x=(j%2)*800;y=520+(j//2)*300;b=boxes[label]
            for a,variant in enumerate(('original','physical-lighting')):
                im=Image.open(arms[variant]['image']).convert('RGB');mask=np.frombuffer(Path(arms[variant]['mask']).read_bytes(),dtype='u1').reshape(1080,1920,3)[:,:,2]==int(label)
                pixels=np.asarray(im).copy();pixels[mask]=(pixels[mask]*.65+np.array([255,0,255])*.35).astype('u1')
                crop=im.crop(b);crop.thumbnail((190,250));overlay=Image.fromarray(pixels).crop(b);overlay.thumbnail((190,250))
                canvas.paste(crop,(x+a*400,y+35));canvas.paste(overlay,(x+a*400+195,y+35))
            draw.text((x,y),label+' '+mapping[label]['object_id'],fill='black')
            targets.append(dict(runtime_label=label,object=mapping[label],full_box=b,visible_pixels=arms['original']['record']['mask_runtime_pixels'][label]))
        dest=folder/f'{pid}.png';canvas.save(dest);paths.append(dest)
        events.append(dict(pair_id=pid,arms=arms,targets=targets,page=str(dest),page_sha256=prior.file_sha256(dest)))
    prior.frozen(folder/'evidence.json',dict(status='awaiting_explicit_AI_review',events=events,inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('PAIRED_REVIEW_PAGES',len(events))

if __name__=='__main__':main()
