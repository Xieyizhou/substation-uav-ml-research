"""Render extremes on deterministically selected material examples for review."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageOps
from scripts.vision import material_routed_contrast_control as source
from scripts.vision import routed_gamma_transform as gamma
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

OUT=source.OUT.parent/'routed-gamma-transfer-control-v1'

def run():
    p=source.checked(source.OUT/'protocol.json');rows=[];paths=[source.OUT/'protocol.json',Path(__file__),Path(gamma.__file__)]
    candidates=sorted((r for r in p['pool_rows'] if r['variant'] in source.MATERIAL),key=lambda r:r['member_id'])
    # One member for each registered material variant; no selection by prediction.
    members=[next(r for r in candidates if r['variant']==v) for v in sorted(source.MATERIAL)]
    folder=OUT/'preview-v1';folder.mkdir(parents=True,exist_ok=True)
    for i,r in enumerate(members):
        for kind in ('image','label'):
            if file_sha256(r[kind+'_path'])!=r[kind+'_sha256']:raise ValueError('Stale input')
            paths.append(Path(r[kind+'_path']))
        im=Image.open(r['image_path']).convert('RGB');im=ImageOps.contain(im,(640,640))
        x=torch.from_numpy(np.array(im)).permute(2,0,1).unsqueeze(0)
        canvas=Image.new('RGB',(1920,im.height+35),'white');draw=ImageDraw.Draw(canvas)
        for j,(c,g) in enumerate(((.75,.8),(1.,1.),(1.25,1.25))):
            z,_=source.transform_runtime.transform(x,[c]);z=gamma.transform(z,[g])
            canvas.paste(Image.fromarray(z[0].permute(1,2,0).numpy()),(j*640,35));draw.text((j*640,5),f'{r["variant"]} contrast={c} gamma={g}',fill='black')
        page=folder/f'frame-{i:02}.png';canvas.save(page);paths.append(page)
        rows.append(dict(member_id=r['member_id'],variant=r['variant'],image_sha256=r['image_sha256'],label_sha256=r['label_sha256'],page=str(page.resolve()),page_sha256=file_sha256(page)))
    return write_record(folder/'evidence.json',dict(status='preview_review_pending',rows=rows,training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in paths}))

if __name__=='__main__':print(len(run()['rows']))
