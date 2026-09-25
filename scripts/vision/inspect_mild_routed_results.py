"""Current-arm evidence: all losses vs routed, original/light vs reference, all FP."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from scripts.vision import mild_routed_contrast_control as arm
from scripts.vision import inspect_dose_matched_results as prior
from scripts.vision.evaluate_reactor_visibility_expansion import _load_inputs
from scripts.vision.evaluate_reviewed_negative_order import compare_truth
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

OUT = arm.OUT / 'audit-v1'

def run():
    prior.TRAIN=arm.OUT; prior.OUT=OUT; prior.KEYS=arm.KEYS
    prior.SOURCE=arm.SOURCE; prior.SOURCE_KEYS=arm.SOURCE_KEYS
    prior.GLOBAL=arm.GLOBAL; prior.GLOBAL_KEYS=arm.GLOBAL_KEYS
    original=prior.run()
    dest=OUT/'material-evidence.json'
    if dest.exists(): return arm.checked(dest)
    pairs,_,deps=_load_inputs(); images={r['image_sha256']:r for r,_ in pairs}; groups={}
    for seed,key,oldkey in zip((7,17,27),arm.KEYS,arm.SOURCE_KEYS):
        records=[]
        for root,k in ((arm.SOURCE,oldkey),(arm.OUT,key)):
            cp=root/'evaluation-v1/units'/k/'completion.json'; c=arm.checked(cp)
            rp=Path(c['result']); records.append(arm.checked(rp))
            for p in (cp,rp): deps[str(p.resolve())]=file_sha256(p)
        old={(r['pair_id'],r['variant']):r for r in records[0]['rows']}
        for row in records[1]['rows']:
            if row['variant']!='material': continue
            states=compare_truth(old[row['pair_id'],row['variant']],row)
            for i,x in enumerate(states):
                if x['state']!='loss': continue
                groups.setdefault(row['image_sha256'],[]).append(dict(seed=seed,truth_index=i,truth=x['truth'],
                    diagnosis=next(m for m in row['misses'] if m['truth_index']==i),loss_against=['routed']))
    frames=[]
    for n,(digest,events) in enumerate(sorted(groups.items())):
        s=images[digest]; im=Image.open(s['image_path']).convert('RGB')
        if file_sha256(s['image_path'])!=digest: raise ValueError('Stale RGB')
        indices=sorted({x['truth_index'] for x in events}); overlay=im.copy(); d=ImageDraw.Draw(overlay)
        canvas=Image.new('RGB',(1200,740+300*((len(indices)+2)//3)),'white'); cd=ImageDraw.Draw(canvas)
        for j,idx in enumerate(indices):
            t=next(e['truth'] for e in events if e['truth_index']==idx); box=t['bbox_xyxy']
            d.rectangle(box,outline='red',width=4); d.text(tuple(box[:2]),str(idx),fill='red')
            x,y=(j%3)*400,740+(j//3)*300
            canvas.paste(ImageOps.contain(im.crop(box),(390,255)),(x,y+30))
            cd.text((x,y),f"GT {idx} {t['class_name']}",fill='black')
        canvas.paste(ImageOps.contain(overlay,(1200,700)),(0,30)); cd.text((5,5),f'material-{n:02}',fill='black')
        page=OUT/f'material-{n:02}.png'; canvas.save(page)
        frames.append(dict(frame_id=f'material-{n:02}',pair_id=s['pair_id'],image_sha256=digest,
            image_path=s['image_path'],events=events,evidence_path=str(page.resolve()),evidence_sha256=file_sha256(page)))
        deps[str(page.resolve())]=file_sha256(page)
    deps[str(Path(__file__).resolve())]=file_sha256(__file__)
    return write_record(dest,dict(frames=frames,status='visual_review_pending',training_admitted=False,promotable=False,inputs=deps))

if __name__=='__main__':
    r=run(); print('MATERIAL_FRAMES',len(r['frames']))
