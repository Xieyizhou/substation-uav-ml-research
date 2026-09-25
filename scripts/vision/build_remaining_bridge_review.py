"""Build per-lineage three-variant review pages without review decisions."""
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageOps
from scripts.vision.replay_retained_bridge_remaining import OUT, read,save,file_sha256,verify_tree

def main():
    dest=OUT/'review-manifest.json'
    if dest.exists():verify_tree(dest);return
    verify_tree(OUT/'progress.json');p=read(OUT/'protocol.json');progress=read(OUT/'progress.json')
    partial='--partial' in sys.argv
    if len(progress['frames'])!=33 and not partial:raise ValueError('Incomplete replay')
    results={r['member_id']:r for r in progress['frames']};items=[];pages=[];blocked=[]
    inputs={str(x):file_sha256(x) for x in (OUT/'progress.json',Path(__file__))}
    for gi,key in enumerate(sorted({f['lineage_id'] for f in p['frames']}),1):
        frames=[f for f in p['frames'] if f['lineage_id']==key];data=[]
        if partial and any(f['member_id'] not in results for f in frames):continue
        for f in frames:
            result=results[f['member_id']]
            if result['status']!='original_pixel_evidence_certified':
                blocked.append(dict(member_id=f['member_id'],reason=result['status']));continue
            rp=Path(result['receipt_path']);r=read(rp);data.append((f,rp,r['records'][0]['targets']))
        n=max((len(f['events']) for f,_,_ in data),default=0)
        for start in range(0,n,3):
            page=OUT/f'group-{gi:02}-part-{start//3+1}.png';canvas=Image.new('RGB',(2400,1500),'white');draw=ImageDraw.Draw(canvas)
            for col,(f,rp,targets) in enumerate(data):
                x=col*800
                draw.text((x+8,8),f"Group {gi} {f['variant']} {key[-12:]}",fill='black')
                rgb=rp.parent/'first-stable-window/frame-1-rgb.png'
                with Image.open(rgb) as im:canvas.paste(ImageOps.contain(im,(780,430)),(x+8,30))
                for j,event in enumerate(f['events'][start:start+3]):
                    target=next(t for t in targets if t['review_id']==event['review_id']);y=480+j*335
                    crop=rp.parent/'first-stable-window'/f"1-{event['review_id']}-crop.png"
                    draw.text((x+8,y),f"{event['review_id']} {event['object_id']} pixels={target['visible_pixel_count']}",fill='black')
                    with Image.open(crop) as im:canvas.paste(ImageOps.contain(im,(780,305)),(x+8,y+23))
                    items.append(dict(**{**event,**target},member_id=f['member_id'],lineage_id=key,variant=f['variant'],source_image=f['source_image'],source_sha256=file_sha256(f['source_image']),crop_path=str(crop),crop_sha256=file_sha256(crop),page_path=str(page),receipt_path=str(rp),review_status='pending'))
            canvas.save(page);inputs[str(page)]=file_sha256(page);pages.append(str(page))
    if not partial:save(dest,dict(status='evidence_pending_explicit_review',items=items,pages=pages,blocked_frames=blocked,inputs=inputs))
    print('REVIEW_PAGES',len(pages),'BOXES',len(items),'BLOCKED',len(blocked),flush=True)

if __name__=='__main__':main()
