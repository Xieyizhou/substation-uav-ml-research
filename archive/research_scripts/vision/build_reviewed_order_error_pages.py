"""Generate full images and exact prediction crops; no review decisions."""
from pathlib import Path
from PIL import Image,ImageDraw
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision.evaluate_reviewed_negative_order import OUT,checked,ev

def run():
    report=checked(OUT/'summary.json');_,negative,_=ev._load_inputs()
    by={r['image_sha256']:r for r in negative};groups={}
    for i,e in enumerate(report['negative_fp_review_queue']):groups.setdefault(e['image_sha256'],[]).append(dict(e,event_id=f'fp-{i:02}'))
    root=OUT/'error-review-v1';root.mkdir(exist_ok=True);frames=[]
    for i,(h,events) in enumerate(sorted(groups.items())):
        source=by[h];im=Image.open(source['image_path']).convert('RGB');overlay=im.copy();d=ImageDraw.Draw(overlay)
        page=Image.new('RGB',(1920,1140+350*((len(events)+3)//4)),'white');pd=ImageDraw.Draw(page)
        for j,e in enumerate(events):
            b=e['prediction']['bbox_xyxy'];d.rectangle(b,outline='red',width=3);d.text((b[0],b[1]),e['event_id'],fill='white',stroke_width=1,stroke_fill='black')
            crop=im.crop(tuple(map(int,b)));crop.thumbnail((470,310));x=(j%4)*480;y=1140+(j//4)*350
            page.paste(crop,(x,y+30));pd.text((x,y),f'{e["event_id"]} seed {e["cell"].split("-")[-1]} {e["prediction"]["class_name"]} {e["prediction"]["confidence"]:.3f}',fill='black')
        page.paste(overlay,(0,40));pd.text((10,10),f'frame-{i:02} {source["view_id"]} {source["variant"]}',fill='black')
        path=root/f'frame-{i:02}.png'
        if path.exists():raise ValueError('Existing evidence must not be overwritten')
        page.save(path);frames.append(dict(frame_id=f'frame-{i:02}',image_path=source['image_path'],image_sha256=h,evidence_path=str(path.resolve()),evidence_sha256=file_sha256(path),events=events))
    return write_record(root/'evidence.json',dict(frames=frames,training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in (OUT/'summary.json',Path(__file__))}))

if __name__=='__main__':print(len(run()['frames']))
