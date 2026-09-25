"""Generate complete-frame evidence for residual material fitting errors; no decisions."""
from collections import defaultdict
from PIL import Image, ImageDraw, ImageOps
from scripts.vision.diagnose_lr_material_fit import OUT, KEYS, checked
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from pathlib import Path

def main():
    dest=OUT/'residual-review-v1';dest.mkdir(exist_ok=True)
    grouped=defaultdict(list);deps={}
    for key in KEYS:
        path=OUT/f'{key}.json';r=checked(path);deps[str(path)]=file_sha256(path)
        for row in r['rows']:
            for miss in row['misses']:grouped[row['member_id']].append((key,row,miss))
    frames=[]
    for i,(member,events) in enumerate(sorted(grouped.items())):
        row=events[0][1];image=Image.open(row['image_path']).convert('RGB')
        if file_sha256(row['image_path'])!=row['image_sha256']:raise ValueError('Stale image')
        indices=sorted({e[2]['truth_index'] for e in events})
        canvas=Image.new('RGB',(1200,760+len(indices)*310),'white');draw=ImageDraw.Draw(canvas)
        draw.text((12,10),member,fill='black')
        full=image.copy();d=ImageDraw.Draw(full)
        for j,t in enumerate(row['truth']):
            d.rectangle(t['bbox_xyxy'],outline='red' if j in indices else 'lime',width=4)
            d.text(tuple(t['bbox_xyxy'][:2]),str(j)+' '+t['class_name'],fill='red')
        canvas.paste(ImageOps.contain(full,(1180,710)),(10,40))
        for n,j in enumerate(indices):
            t=row['truth'][j];x1,y1,x2,y2=t['bbox_xyxy'];crop=image.crop((max(0,int(x1)-12),max(0,int(y1)-12),min(image.width,int(x2)+12),min(image.height,int(y2)+12)))
            y=760+n*310;canvas.paste(ImageOps.contain(crop,(580,290)),(10,y))
            draw.text((610,y+10),f"GT {j} {t['class_name']}",fill='black')
            for z,(key,_,miss) in enumerate(e for e in events if e[2]['truth_index']==j):draw.text((610,y+40+z*25),key+' '+miss['reason'],fill='black')
        page=dest/f'frame-{i:02}.png';canvas.save(page)
        frames.append(dict(member_id=member,page=str(page),page_sha256=file_sha256(page),image=row['image_path'],image_sha256=row['image_sha256'],label_sha256=row['label_sha256'],truth=row['truth'],events=[dict(key=k,**m) for k,_,m in events],review_status='pending'))
        deps[str(page)]=file_sha256(page);deps[row['image_path']]=row['image_sha256']
    deps[str(Path(__file__).resolve())]=file_sha256(__file__)
    write_record(dest/'evidence.json',dict(frames=frames,training_admitted=False,promotable=False,inputs=deps))
    print(dest)

if __name__=='__main__':main()
