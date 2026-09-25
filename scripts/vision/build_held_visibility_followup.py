"""Independent evidence for the frozen 27 unresolved condition events."""
from PIL import Image, ImageDraw
from scripts.vision.record_switchgear_condition_review import OUT as SOURCE, read, save, file_sha256, verify_tree
from pathlib import Path
OUT = SOURCE / 'held-visibility-followup-v1'

def main():
    verify_tree(SOURCE / 'completion.json')
    m = read(SOURCE / 'manifest.json')
    ids = read(SOURCE / 'review.json')['held_review_ids']
    rows = [r for r in m['items'] if r['review_id'] in ids]
    OUT.mkdir(exist_ok=True)
    inputs = {str(p):file_sha256(p) for p in (SOURCE/'completion.json',SOURCE/'manifest.json',SOURCE/'review.json',Path(__file__))}
    for start in range(0,len(rows),6):
        canvas=Image.new('RGB',(1200,900),'white');draw=ImageDraw.Draw(canvas)
        path=OUT/f'page-{start//6+1:02}.png'
        for n,r in enumerate(rows[start:start+6]):
            im=Image.open(r['image_path']).convert('RGB')
            x1,y1,x2,y2=r['bbox_xyxy'];pad=50
            crop=(max(0,int(x1)-pad),max(0,int(y1)-pad),min(im.width,int(x2)+pad+1),min(im.height,int(y2)+pad+1))
            tile=im.crop(crop);d=ImageDraw.Draw(tile);d.rectangle((x1-crop[0],y1-crop[1],x2-crop[0],y2-crop[1]),outline='red',width=2)
            tile.thumbnail((590,260));x=n%2*600;y=n//2*300
            canvas.paste(tile,(x,y+35));draw.text((x+3,y+5),r['review_id']+' target=red; contextual crop',fill='black')
            r['followup_evidence_path']=str(path)
        canvas.save(path);inputs[str(path)]=file_sha256(path)
    save(OUT/'manifest.json',dict(items=rows,inputs=inputs,status='evidence_only',training_admitted=False,promotable=False))

if __name__=='__main__':main()
