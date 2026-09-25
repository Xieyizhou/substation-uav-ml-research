"""Produce review evidence only; never synthesize review decisions."""
from PIL import Image, ImageDraw, ImageOps
from scripts.vision.replay_retained_bridge_pilot import OUT,read,save,file_sha256,verify_tree
from pathlib import Path

def main():
    dest=OUT/'review-manifest.json'
    if dest.exists():verify_tree(dest);return
    verify_tree(OUT/'progress.json')
    protocol=read(OUT/'protocol.json');progress=read(OUT/'progress.json')
    if len(progress['frames'])!=6:raise ValueError('Incomplete replay')
    results={r['member_id']:r for r in progress['frames']};items=[]
    inputs={str(p):file_sha256(p) for p in (OUT/'progress.json',Path(__file__))}
    for frame in protocol['frames']:
        rp=Path(results[frame['member_id']]['receipt_path']);record=read(rp)
        if record['status']!='original_pixel_evidence_certified':raise ValueError('Uncertified replay; original visibility remains unknown')
        targets={t['review_id']:t for t in record['records'][0]['targets']}
        page=OUT/(frame['review_ids'][0]+'-review.png')
        canvas=Image.new('RGB',(1600,1000),'white');draw=ImageDraw.Draw(canvas)
        for index,event in enumerate(frame['events']):
            crop=rp.parent/'first-stable-window'/f"1-{event['review_id']}-crop.png"
            x,y=(index%2)*800,(index//2)*500
            with Image.open(crop) as im:canvas.paste(ImageOps.contain(im,(780,455)),(x+10,y+40))
            draw.text((x+10,y+8),f"{event['review_id']} {event['object_id']} {frame['variant']} / magenta=instance",fill='black')
            items.append(dict(**{**event,**targets[event['review_id']]},
                member_id=frame['member_id'],variant=frame['variant'],lineage_id=frame['lineage_id'],
                source_image=frame['source_image'],crop_path=str(crop),page_path=str(page),
                source_sha256=file_sha256(frame['source_image']),crop_sha256=file_sha256(crop),
                receipt_path=str(rp),review_status='pending'))
        canvas.save(page);inputs[str(page)]=file_sha256(page)
    save(dest,dict(status='evidence_pending_explicit_review',items=items,inputs=inputs))
    print('EVIDENCE_READY',OUT)

if __name__=='__main__':main()
