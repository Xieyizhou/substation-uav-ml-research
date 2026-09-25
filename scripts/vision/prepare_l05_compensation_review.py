"""Prepare aligned full-frame evidence only; no automatic review decisions."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scripts.vision.run_physical_lighting_capture_v2 import OUT as CAPTURE,prior
from scripts.vision.trace_l05_risk_scope import OUT as SCOPE

OUT=SCOPE.parent/'l05-held-lighting-control-v1'

def main():
    cp=CAPTURE/'capture-receipt.json';pp=CAPTURE/'protocol.json'
    c,p=prior.read(cp),prior.read(pp)
    for x in (c,p,prior.read(SCOPE/'completion.json')):prior.verify(x)
    folder=OUT/'evidence';folder.mkdir(parents=True,exist_ok=True)
    events=[];paths=[cp,pp,SCOPE/'completion.json',Path(__file__).resolve()]
    for result in c['results']:
        if result['status']!='capture_technical_checks_passed':continue
        rp=Path(result['receipt']);r=prior.read(rp);prior.verify(r)
        frame=next(f for f in p['frames'] if f['pair_id']==result['pair_id'] and f['variant']=='original')
        rec=r['records'][0];n=rec['capture_index'];ip=rp.parent/f'frame-{n}-rgb.png';mp=rp.parent/f'frame-{n}-mask.bin'
        paths += [rp,ip,mp];im=Image.open(ip).convert('RGB');mask=np.frombuffer(mp.read_bytes(),dtype='u1').reshape(1080,1920,3)[:,:,2]
        pid=result['pair_id'];overview=im.copy();d=ImageDraw.Draw(overview)
        for lab,b in rec['full_boxes'].items():d.rectangle(b,outline='red',width=3);d.text(b[:2],lab,fill='red')
        full=folder/f'{pid}-full.png';overview.save(full);paths.append(full)
        targets=[]
        for lab,b in rec['full_boxes'].items():
            rgb=np.array(im);sel=mask==int(lab);rgb[sel]=(rgb[sel]*.6+np.array([255,0,255])*.4).astype('u1')
            crops=[x.crop(b) for x in (im,Image.fromarray(rgb))]
            canvas=Image.new('RGB',(crops[0].width*2,crops[0].height+30),'white');draw=ImageDraw.Draw(canvas)
            for j,x in enumerate(crops):canvas.paste(x,(j*x.width,30))
            draw.text((4,4),pid+' '+lab+' '+frame['instance_mapping'][lab]['object_id'],fill='black')
            dest=folder/f'{pid}-{lab}.png';canvas.save(dest);paths.append(dest)
            targets.append(dict(runtime_label=lab,instance=frame['instance_mapping'][lab],box=b,crop=str(dest),visible_pixels=int(sel.sum())))
        events.append(dict(pair_id=pid,member_id=frame['member_id'],full=str(full),targets=targets,receipt=str(rp)))
    prior.frozen(folder/'manifest.json',dict(status='awaiting_explicit_full_frame_AI_review',events=events,training_ready=False,
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print(folder)

if __name__=='__main__':main()
