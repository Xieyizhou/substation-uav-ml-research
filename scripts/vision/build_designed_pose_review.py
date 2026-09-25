"""New-camera full-frame and per-truth visual evidence; no decisions."""
from PIL import Image,ImageDraw
from pathlib import Path
from scripts.vision.verify_designed_full_scene_poses import OUT,prior

def main():
    cp=OUT/'completion.json';c=prior.read(cp);prior.verify(c);dest=OUT/'review';dest.mkdir(exist_ok=True);events=[];paths=[cp,Path(__file__)]
    for r in c['units']:
        sid=r['probe_id'];up=OUT/sid/'protocol.json'
        if not up.exists():continue
        unit=prior.read(up);prior.verify(unit);f=unit['frame'];source=prior.read(f['source_receipt']);im=Image.open(f['source_image']).convert('RGB')
        count=len(f['events']);canvas=Image.new('RGB',(1500,750+300*((count+2)//3)),'white');d=ImageDraw.Draw(canvas);over=im.copy();od=ImageDraw.Draw(over)
        for e in f['events']:od.rectangle(e['bbox_xyxy'],outline='red',width=3)
        over.thumbnail((1450,700));canvas.paste(over,(0,25));d.text((0,0),sid+' '+f['class_name'],fill='black')
        for n,e in enumerate(f['events']):
            crop=im.crop(tuple(e['bbox_xyxy']));op=dest/(e['review_id']+'.png');crop.save(op);paths.append(op);crop.thumbnail((480,270));x=n%3*500;y=750+n//3*300
            d.text((x,y),e['review_id']+' '+e['object_id'],fill='black');canvas.paste(crop,(x,y+20))
            events.append(dict(e,truth=source['truth']['objects'][n],probe_id=sid,image_sha256=prior.file_sha256(f['source_image']),crop_sha256=prior.file_sha256(op),crop_path=str(op)))
        op=dest/(sid+'.png');canvas.save(op);paths += [op,up,Path(f['source_image']),Path(f['source_receipt'])]
    prior.frozen(dest/'evidence.json',dict(status='explicit_review_required',events=events,training_ready=False,inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('EVENTS',len(events))

if __name__=='__main__':main()
