"""Generate all-label pilot evidence; never invent review decisions."""
from pathlib import Path
from PIL import Image, ImageDraw
from scripts.vision.run_condition_transfer_pilot import OUT, prior
from scripts.vision.freeze_condition_transfer_probe import OUT as DESIGN
from scripts.vision.audit_material_transfer_scope import resolve_truth


def main():
    dest=OUT/'review';ep=dest/'evidence.json'
    if ep.exists():
        e=prior.read(ep);prior.verify(e);return e
    cp=OUT/'completion.json';c=prior.read(cp);prior.verify(c)
    if c['status']!='pilot_captured_explicit_review_pending' or len(c['units'])!=20:raise ValueError('Incomplete pilot')
    pp=DESIGN/'protocol.json';p=prior.read(pp);prior.verify(p)
    sources={s['source_pose_id']:s for s in p['sources']}
    dest.mkdir(exist_ok=True);paths=[cp,pp,Path(__file__).resolve()];events=[];pages=[]
    for u in c['units']:
        source=sources[u['source_id']];f=source['source_frame']
        rp=Path(u['receipt']);r=prior.read(rp);prior.verify(r)
        if not r['process_cleanup_complete'] or r['stable_frames']<3:raise ValueError('Invalid replay')
        tp=Path(f['source_receipt']);truth=prior.read(tp)['truth'];resolved=resolve_truth(truth,f['instance_mapping'])
        ip=rp.parent/'first-stable-window/frame-1-rgb.png';im=Image.open(ip).convert('RGB')
        count=len(resolved);page=Image.new('RGB',(1500,750+300*((count+2)//3)),'white')
        over=im.copy();draw=ImageDraw.Draw(over)
        for n,t in enumerate(resolved):
            draw.rectangle(t['bbox_xyxy'],outline='red',width=3);draw.text(t['bbox_xyxy'][:2],str(n),fill='white')
        over.thumbnail((1450,700));page.paste(over,(0,25));pd=ImageDraw.Draw(page)
        pd.text((0,0),u['source_id']+' '+u['condition'],fill='black')
        ppng=dest/(u['source_id']+'-'+u['condition']+'.png');local=[]
        for n,t in enumerate(resolved):
            eid=f"{u['source_id']}-{u['condition']}-{n:02}"
            crop=im.crop(tuple(t['bbox_xyxy']));crop_path=dest/(eid+'.png');crop.save(crop_path)
            crop.thumbnail((480,270));x=n%3*500;y=750+n//3*300
            pd.text((x,y),str(n)+' '+t['object_id'],fill='black');page.paste(crop,(x,y+20))
            local.append(dict(event_id=eid,source_id=u['source_id'],condition=u['condition'],object_id=t['object_id'],
                truth=truth['objects'][n],image_path=str(ip),image_sha256=prior.file_sha256(ip),
                crop_path=str(crop_path),crop_sha256=prior.file_sha256(crop_path),page_path=str(ppng),
                source_receipt=str(tp),source_receipt_sha256=prior.file_sha256(tp),replay_receipt=str(rp),
                replay_receipt_sha256=prior.file_sha256(rp)))
            paths.append(crop_path)
        page.save(ppng)
        for event in local:event['page_sha256']=prior.file_sha256(ppng)
        events+=local;pages.append(dict(source_id=u['source_id'],condition=u['condition'],page_path=str(ppng),label_count=count))
        paths += [ip,ppng,rp,tp]
    return prior.frozen(ep,dict(status='explicit_all_label_pilot_review_pending',events=events,pages=pages,
        training_ready=False,training_started=False,inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':
    r=main();print('pages',len(r['pages']),'labels',len(r['events']))
