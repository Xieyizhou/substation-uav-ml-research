"""Render observed formal false-positive boxes, without assigning review decisions."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
from scripts.vision.freeze_full_image_training import OUT,read,save,file_sha256,verify_tree
from scripts.vision.exposure_protocol import BASE,verify

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cell',required=True);args=ap.parse_args()
    target=OUT/'negative-evidence'/args.cell/'manifest.json'
    if target.exists():verify_tree(target);print('VERIFIED_EXISTING',target);return
    ep=OUT/f'evaluation-{args.cell}.json';r=read(ep);verify(r)
    sp=BASE/'hard-negative-isolated-v2/semantic-review.json'
    sources={(x['view_id'],x['variant']):x for x in read(sp)['frames']}
    inputs={str(p):file_sha256(p) for p in (ep,sp,Path(__file__))};frames=[]
    target.parent.mkdir(parents=True,exist_ok=False)
    for row in r['negative_rows']:
        if not row['predictions']:continue
        src=Path(sources[row['view_id'],row['variant']]['image_path'])
        if file_sha256(src)!=row['image_sha256']:raise ValueError('Stale image')
        inputs[str(src)]=row['image_sha256']
        with Image.open(src) as im:image=im.convert('RGB')
        overlay=image.copy();draw=ImageDraw.Draw(overlay);boxes=[]
        for i,p in enumerate(row['predictions']):
            box=p['bbox_xyxy'];draw.rectangle(box,outline='red',width=4)
            draw.text((box[0],box[1]),f'{i} {p["class_name"]} {p["confidence"]:.4f}',fill='red')
            crop=target.parent/f'{row["view_id"]}-{i}.png'
            image.crop((max(0,int(box[0])),max(0,int(box[1])),min(image.width,int(box[2])+1),min(image.height,int(box[3])+1))).save(crop)
            inputs[str(crop)]=file_sha256(crop)
            boxes.append(dict(**p,index=i,crop_path=str(crop),review_status='pending'))
        op=target.parent/f'{row["view_id"]}-overlay.png';overlay.save(op);inputs[str(op)]=file_sha256(op)
        frames.append(dict(view_id=row['view_id'],variant=row['variant'],image_path=str(src),image_sha256=row['image_sha256'],overlay_path=str(op),boxes=boxes))
    save(target,dict(status='evidence_only_not_reviewed',cell=args.cell,frames=frames,inputs=inputs))
    print('EVIDENCE',target,flush=True)

if __name__=='__main__':main()
