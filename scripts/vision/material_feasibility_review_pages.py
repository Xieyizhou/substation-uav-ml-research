"""Prediction-free contact pages: individual, correctly resolved truth crops."""
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.material_control_feasibility import OUT,prior

def main():
    src=OUT/'initial-gate.json';p=prior.read(src);prior.verify(p)
    paths=[src,Path(__file__).resolve()];pages=[]
    folder=OUT/'review-pages';folder.mkdir(exist_ok=True)
    for n,view in enumerate(sorted({r['view_id'] for r in p['corrected_target_records']}),1):
        for group,variants in (('OM',('original','material')),('BL',('background','lighting'))):
            rows=[r for r in p['corrected_target_records'] if r['view_id']==view and r['variant'] in variants]
            ids=sorted({r['object_id'] for r in rows});canvas=Image.new('RGB',(1440,440+len(ids)*180),'white');d=ImageDraw.Draw(canvas)
            for col,v in enumerate(variants):
                a=[r for r in rows if r['variant']==v];im=Image.open(a[0]['image_path']).convert('RGB');im.thumbnail((720,405));canvas.paste(im,(720*col,25));d.text((720*col,5),f'V{n:02} {v}',fill='black')
                for j,oid in enumerate(ids):
                    r=next(r for r in a if r['object_id']==oid);crop=Image.open(r['crop_path']);crop.thumbnail((690,150));y=440+j*180;canvas.paste(crop,(720*col,y+20));d.text((720*col,y),f'{j+1} {oid}',fill='black')
            dest=folder/f'V{n:02}-{group}.png';canvas.save(dest);paths.append(dest)
            pages.append(dict(page=str(dest),sha256=prior.file_sha256(dest),view_id=view,object_order=ids,variants=variants))
    prior.frozen(OUT/'review-pages.json',dict(status='awaiting_explicit_visual_decisions',pages=pages,inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('24 pages generated')

if __name__=='__main__':main()
