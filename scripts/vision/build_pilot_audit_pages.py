"""Render evidence only; never generate semantic review decisions."""
from pathlib import Path
from PIL import Image, ImageDraw
from src.vision.canonical.plan import read_record, write_record
from src.ml.artifacts import file_sha256

BASE = Path('data/research/ml_training_recovery_v1/reactor-visibility-expansion-v4')

def run():
    receipt = BASE/'pilot/collection-receipt.json'
    out = BASE/'pilot-full-label-audit-v1'
    out.mkdir(exist_ok=True)
    rows = []
    for i, v in enumerate(read_record(receipt)['views']):
        src = Path(v['rgb_path'])
        if file_sha256(src) != v['image_sha256']:
            raise ValueError('Changed original RGB')
        im = Image.open(src).convert('RGB')
        overlay = im.copy(); draw = ImageDraw.Draw(overlay)
        objects = v['truth']['objects']
        page = Image.new('RGB', (1920, 1140 + ((len(objects)+3)//4)*310), 'white')
        for j, obj in enumerate(objects):
            box = obj['bbox_xyxy']
            draw.rectangle(box, outline='red', width=3)
            draw.text((box[0]+3,box[1]+3),f'{j} {obj["class_name"]}', fill='black',stroke_width=1,stroke_fill='white')
            crop = im.crop(tuple(map(int,box)))
            crop.thumbnail((470,270))
            x,y=(j%4)*480,1140+(j//4)*310
            page.paste(crop,(x,y+25))
            ImageDraw.Draw(page).text((x,y),f'{j}: {obj["class_name"]}',fill='black')
        page.paste(overlay,(0,40))
        ImageDraw.Draw(page).text((10,10),f'Frame {i:02}: {v["view_id"]}',fill='black')
        dest=out/f'frame-{i:02}.png'
        if dest.exists(): raise ValueError('Do not overwrite evidence')
        page.save(dest)
        rows.append(dict(view_id=v['view_id'],page=str(dest.resolve()),page_sha256=file_sha256(dest),image_sha256=v['image_sha256'],objects=objects))
    write_record(out/'evidence.json',dict(rows=rows,training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in (receipt,Path(__file__))}))

if __name__=='__main__':run()
