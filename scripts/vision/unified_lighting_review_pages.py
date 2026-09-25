"""Four paired review pages with every target crop; no approval inference."""
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
from scripts.vision.capture_unified_hold_lighting import OUT as CAP,DESIGN,prior,capture

def main():
    cp=CAP/'receipt.json';c=prior.read(cp);prior.verify(c)
    if c['status']!='four_captured_review_pending':raise ValueError('Incomplete capture')
    folder=DESIGN/'light-review';folder.mkdir(exist_ok=True);paths=[cp,Path(__file__).resolve()];events=[]
    for result in c['results']:
        pid=result['pair_id'];rp=Path(result['receipt']);r=prior.read(rp);prior.verify(r)
        nrp=capture.OUT/'replay'/f'{pid}-original/attempt-01/receipt.json';nr=prior.read(nrp);prior.verify(nr)
        n=r['records'][0]['capture_index'];nn=nr['records'][0]['capture_index'];ip=rp.parent/f'frame-{n}-rgb.png';op=nrp.parent/f'frame-{nn}-rgb.png';mp=rp.parent/f'frame-{n}-mask.bin'
        paths += [rp,nrp,ip,op,mp];orig=Image.open(op).convert('RGB');dark=Image.open(ip).convert('RGB')
        if orig.tobytes()==dark.tobytes():raise ValueError('No treatment RGB change')
        page=Image.new('RGB',(1920,560),'white');d=ImageDraw.Draw(page)
        for j,im in enumerate((orig.copy(),dark.copy())):
            draw=ImageDraw.Draw(im)
            for k,b in r['records'][0]['full_boxes'].items():draw.rectangle(b,outline='red',width=3);draw.text(b[:2],k,fill='red')
            im.thumbnail((960,540));page.paste(im,(j*960,20));d.text((j*960+5,3),pid+(' original' if j==0 else ' physical light'),fill='black')
        fp=folder/f'{pid}-pair.png';page.save(fp);paths.append(fp);targets=[]
        mask=np.frombuffer(mp.read_bytes(),dtype='u1').reshape(1080,1920,3)[:,:,2]
        for k,b in r['records'][0]['full_boxes'].items():
            a=np.array(dark);sel=mask==int(k);a[sel]=(a[sel]*.6+np.array([255,0,255])*.4).astype('u1')
            crops=[im.crop(b) for im in (orig,dark,Image.fromarray(a))];w,h=crops[0].size
            canvas=Image.new('RGB',(3*w,h+20),'white');draw=ImageDraw.Draw(canvas);draw.text((3,3),pid+' '+k+' original / light / instance overlay',fill='black')
            for j,im in enumerate(crops):canvas.paste(im,(j*w,20))
            dest=folder/f'{pid}-{k}.png';canvas.save(dest);paths.append(dest);targets.append(dict(label=k,crop=str(dest)))
        events.append(dict(pair_id=pid,pair_page=str(fp),image=str(ip),original=str(op),receipt=str(rp),targets=targets))
    prior.frozen(folder/'evidence.json',dict(status='awaiting_explicit_review',events=events,training_ready=False,inputs={str(x):prior.file_sha256(x) for x in paths}))

if __name__=='__main__':main()
