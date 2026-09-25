"""Evidence only for all replayed instances; never creates review approvals."""
from PIL import Image,ImageDraw
from scripts.vision.run_visibility_cleanup_validation import OUT,prepare,read,save,file_sha256,verify_tree,Path

def main():
    protocol=prepare();items=[];inputs={str(OUT/'protocol.json'):file_sha256(OUT/'protocol.json'),str(Path(__file__)):file_sha256(Path(__file__))}
    for frame in protocol['frames']:
        receipts=sorted((OUT/'replay'/frame['review_ids'][0]).glob('attempt-*/receipt.json'))
        valid=[p for p in receipts if read(p)['status']=='original_pixel_evidence_certified']
        if len(valid)!=1:raise ValueError('Require exactly one certified result: '+str(frame['review_ids']))
        rp=valid[0];record=read(rp);inputs[str(rp)]=file_sha256(rp)
        for target in record['records'][0]['targets']:
            crop=rp.parent/'first-stable-window'/f'1-{target["review_id"]}-crop.png'
            items.append(dict(**target,member_id=frame['member_id'],source_image=frame['source_image'],label_path=frame['label_path'],receipt_path=str(rp),crop_path=str(crop),crop_sha256=file_sha256(crop),original_frame_certified=True))
    items.sort(key=lambda r:r['review_id'])
    for start in range(0,len(items),6):
        canvas=Image.new('RGB',(1200,900),'white');draw=ImageDraw.Draw(canvas);page=OUT/f'review-page-{start//6+1:02}.png'
        for n,r in enumerate(items[start:start+6]):
            im=Image.open(r['crop_path']).convert('RGB');im.thumbnail((580,260));x=n%2*600;y=n//2*300
            canvas.paste(im,(x,y+35));draw.text((x+3,y+5),f'{r["review_id"]} visible pixels={r["visible_pixel_count"]}; magenta=instance',fill='black')
            r['page_path']=str(page)
        canvas.save(page);inputs[str(page)]=file_sha256(page)
    save(OUT/'review-manifest.json',dict(status='evidence_only',items=items,inputs=inputs))
    print('EVIDENCE',len(items))

if __name__=='__main__':main()
