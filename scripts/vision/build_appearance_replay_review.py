"""Render certified pilot masks for explicit content review, never auto-approve."""
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.replay_appearance_recovery import OUT,read,save,file_sha256,verify_tree

def main():
    target=OUT/'review-manifest.json'
    if target.exists():verify_tree(target);print('VERIFIED_EXISTING');return
    verify_tree(OUT/'progress.json');progress=read(OUT/'progress.json');p=read(OUT/'protocol.json')
    if progress['status']!='replay_complete_pending_review':raise ValueError('Replay is incomplete or held')
    inputs={};items=[]
    for f in p['frames']:
        rows=[r for r in progress['frames'] if r['member_id']==f['member_id']]
        if len(rows)!=1:raise ValueError('Missing/duplicate frame result')
        rp=Path(rows[0]['receipt_path']);r=read(rp);inputs[str(rp)]=file_sha256(rp)
        if r['status']!='original_pixel_evidence_certified' or not r['process_cleanup_complete']:raise ValueError('Not certified or cleanup incomplete')
        for t in r['records'][0]['targets']:
            e=next(e for e in f['events'] if e['review_id']==t['review_id']);crop=rp.parent/'first-stable-window'/f"1-{t['review_id']}-crop.png"
            items.append(dict(**t,object_id=e['object_id'],category=f['category'],variant=f['variant'],source_image=f['source_image'],member_id=f['member_id'],initial_reason=e['initial_reason'],crop_path=str(crop),crop_sha256=file_sha256(crop),review_status='pending',original_pixel_evidence_certified=True))
    items.sort(key=lambda x:x['review_id'])
    for start in range(0,len(items),6):
        page=OUT/f'review-page-{start//6+1:02}.png';im=Image.new('RGB',(1200,900),'white');d=ImageDraw.Draw(im)
        for n,item in enumerate(items[start:start+6]):
            crop=Image.open(item['crop_path']).convert('RGB');crop.thumbnail((580,245));x=n%2*600;y=n//2*300;im.paste(crop,(x,y+50))
            d.text((x+3,y+3),f"{item['review_id']} {item['object_id']} {item['variant']}\npixels={item['visible_pixel_count']} magenta=instance",fill='black');item['page_path']=str(page)
        im.save(page);inputs[str(page)]=file_sha256(page)
    for path in (OUT/'protocol.json',OUT/'progress.json',Path(__file__)):inputs[str(path)]=file_sha256(path)
    save(target,dict(status='evidence_only_pending_AI_review',items=items,inputs=inputs))
    print('EVIDENCE_READY',len(items),OUT,flush=True)

if __name__=='__main__':main()
