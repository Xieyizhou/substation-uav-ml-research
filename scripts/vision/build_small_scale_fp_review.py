"""Freeze all six-cell false positives by unique image, without decisions."""
from pathlib import Path
from PIL import Image, ImageDraw
from scripts.vision.small_scale_control import OUT, KEYS, prior
from scripts.vision.evaluate_small_scale import complete

DEST = OUT/'evaluation/false-positive-review-v1'


def build():
    DEST.mkdir(parents=True, exist_ok=True)
    dest = DEST/'evidence.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    p = prior.read(OUT/'design.json'); prior.verify(p)
    np = Path(p['evaluation']['negative_review']); negatives = prior.read(np); prior.verify(negatives)
    sources = {r['image_sha256']:r for r in negatives['frames']}
    if len(sources) != 48: raise ValueError('Negative source identity collision')
    groups = {}; deps = [np, OUT/'design.json', Path(__file__).resolve()]
    for key in KEYS:
        complete(key); ep = OUT/'evaluation'/f'{key}.json'; r = prior.read(ep); prior.verify(r); deps.append(ep)
        for row in r['negative_rows']:
            src = sources[row['image_sha256']]
            if (src['view_id'],src['variant']) != (row['view_id'],row['variant']): raise ValueError('Source conflict')
            for n,pred in enumerate(row['predictions']):
                g = groups.setdefault(row['image_sha256'], dict(source=src, events=[]))
                g['events'].append(dict(event_id=f'{key}:{row["view_id"]}:{row["variant"]}:{n}',
                    cell=key, seed=int(key.split('-')[-1]), prediction=pred))
    pages = []; records = []
    for i,(sha,g) in enumerate(sorted(groups.items()),1):
        im = Image.open(g['source']['image_path']).convert('RGB')
        if prior.file_sha256(g['source']['image_path']) != sha: raise ValueError('Image drift')
        deps.append(Path(g['source']['image_path']))
        for start in range(0,len(g['events']),6):
            chunk = g['events'][start:start+6]
            overlay = im.copy(); draw = ImageDraw.Draw(overlay)
            for j,e in enumerate(chunk,start):
                b = e['prediction']['bbox_xyxy']; draw.rectangle(b,outline='red',width=3)
                draw.text((b[0],b[1]),str(j),fill='red')
            card = Image.new('RGB',(1440,1040),'white'); d = ImageDraw.Draw(card)
            overlay.thumbnail((900,500)); card.paste(overlay,(0,30))
            d.text((5,5),f'F{i:02} {g["source"]["variant"]} predictions {start}..{start+len(chunk)-1}',fill='black')
            path = DEST/f'F{i:02}-{start//6+1}.png'
            for j,e in enumerate(chunk):
                pred = e['prediction']; b = tuple(map(round,pred['bbox_xyxy']))
                if b[2] <= b[0] or b[3] <= b[1]: raise ValueError('Empty ROI requires explicit investigation')
                crop = im.crop(b); crop.thumbnail((470,210))
                x,y=(j%3)*480,550+(j//3)*245
                d.text((x,y),f'{start+j} {e["cell"]} {pred["class_name"]} {pred["confidence"]:.4f}',fill='black')
                card.paste(crop,(x,y+25)); e['page_path']=str(path); e['box_index']=start+j
            card.save(path); pages.append(str(path)); deps.append(path)
        records.append(dict(image_id=f'F{i:02}', **g))
    return prior.frozen(dest,dict(status='awaiting_explicit_AI_review', images=records,pages=pages,
        predictions=sum(len(g['events']) for g in records),unique_images=len(records),
        inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__ == '__main__':
    r=build(); print(r['predictions'],r['unique_images'],len(r['pages']))
