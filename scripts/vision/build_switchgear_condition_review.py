"""Evidence for every existing switchgear training box; no inferred approval."""
import sys,math
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.run_stratified_negative_control import OUT as SOURCE,read,save,file_sha256,verify_tree
from scripts.vision.run_grayscale_control import OUT as GRAY
OUT=SOURCE.parent/'switchgear-condition-review-v1'

def main():
    source=SOURCE/'switchgear-condition-audit.json';verify_tree(source);verify_tree(GRAY/'report-receipt.json');census=read(source)
    items=[];inputs={str(q):file_sha256(q) for q in (source,GRAY/'report-receipt.json',Path(__file__))};OUT.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(census['rows'],1):items.append(dict(row,review_id=f'T{i:03}'))
    for offset in range(0,len(items),12):
        canvas=Image.new('RGB',(1200,1000),'white');draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(items[offset:offset+12]):
            for kind in ('image','label'):
                if file_sha256(row[kind+'_path'])!=row[kind+'_sha256']:raise ValueError('Stale source')
            im=Image.open(row['image_path']).convert('RGB');w,h=im.size;x,y,bw,bh=row['bbox_yolo'];box=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]
            margin=max(8,.12*max(bw*w,bh*h));region=[max(0,math.floor(box[0]-margin)),max(0,math.floor(box[1]-margin)),min(w,math.ceil(box[2]+margin)),min(h,math.ceil(box[3]+margin))]
            crop=im.crop(region);scale=min(390/crop.width,190/crop.height);crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))))
            cd=ImageDraw.Draw(crop);cd.rectangle([(box[0]-region[0])*scale,(box[1]-region[1])*scale,(box[2]-region[0])*scale,(box[3]-region[1])*scale],outline='red',width=1)
            px=j%3*400;py=j//3*250;canvas.paste(crop,(px,py+55));draw.text((px+3,py+3),f"{row['review_id']} {row['subset']}\n640 box: {row['model_640_box_size'][0]:.1f}x{row['model_640_box_size'][1]:.1f}; exposures={row['image_exposures']}",fill='black')
            row.update(bbox_xyxy=box,context_crop_xyxy=region,original_image_size=[w,h])
        page=OUT/f'page-{offset//12+1:02}.jpg';canvas.save(page);inputs[str(page)]=file_sha256(page)
        for row in items[offset:offset+12]:row.update(evidence_path=str(page),evidence_sha256=file_sha256(page))
    save(OUT/'manifest.json',dict(status='pending_AI_assisted_condition_review',items=items,count=len(items),
        scope='Additional visible-side, panel contrast and occlusion annotation only. Original training admission unchanged. Native source resolution governs certainty; enlargement adds no detail.',inputs=inputs))
    print('PAGES',math.ceil(len(items)/12),'ITEMS',len(items))

if __name__=='__main__':main()
