"""Render exact frozen targets for explicit condition review; never decides."""
import re
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from scripts.vision.condition_transfer_design import OUT as SOURCE,ROOT,read,verify,frozen,file_sha256
from scripts.vision.structure_fit import truth_for

OUT=SOURCE/'condition-review-v1'
VARIANTS=('original','material','background','lighting')
FONT=ImageFont.load_default(size=17)


def instance(t):
    m=re.search(r'-instance-(\d+)-',t['annotation_id'])
    if not m:raise ValueError('Unresolved development instance')
    return m.group(1),t['class_name']


def fitted(im,size):
    im=im.copy();im.thumbnail(size);return im


def crop(im,b):
    x1,y1,x2,y2=b
    if x2<=x1 or y2<=y1:raise ValueError('Empty box')
    return im.crop((max(0,x1),max(0,y1),min(im.width,x2),min(im.height,y2)))


def main():
    q=read(SOURCE/'review-queue.json');verify(q);verify(read(SOURCE/'design-receipt.json'))
    dest=OUT/'evidence.json'
    if dest.exists():verify(read(dest));return
    OUT.mkdir(exist_ok=True);paths=[SOURCE/'review-queue.json',SOURCE/'design-receipt.json',Path(__file__)];train=[]
    for i,r in enumerate(q['training_targets'],1):
        s=r['source'];t=r['target']['truth'];im=Image.open(s['image_path']).convert('RGB')
        if file_sha256(s['image_path'])!=s['image_sha256'] or file_sha256(s['label_path'])!=s['label_sha256']:raise ValueError('Stale training source')
        full=im.copy();d=ImageDraw.Draw(full)
        for a in truth_for(s):d.rectangle(a['bbox_xyxy'],outline='yellow',width=2)
        d.rectangle(t['bbox_xyxy'],outline='lime',width=4)
        tile=Image.new('RGB',(800,360),'white');tile.paste(fitted(full,(510,290)),(0,50));tile.paste(fitted(crop(im,t['bbox_xyxy']),(280,290)),(515,50))
        ImageDraw.Draw(tile).text((4,4),f"T{i:03} {t['class_name']} line={t['label_line_index']} {s['subset']}",font=FONT,fill='black')
        path=OUT/f'T{i:03}.png';tile.save(path);paths += [path,Path(s['image_path']),Path(s['label_path'])]
        train.append(dict(review_id=f'T{i:03}',**r,evidence_path=str(path),evidence_sha256=file_sha256(path),native_size=list(im.size)))
    train_pages=[]
    for n in range(0,len(train),6):
        page=Image.new('RGB',(1600,1080),'white')
        for j,r in enumerate(train[n:n+6]):page.paste(Image.open(r['evidence_path']),(j%2*800,j//2*360))
        path=OUT/f'train-page-{n//6+1:02}.png';page.save(path);paths.append(path);train_pages.append(str(path))
    groups={}
    for r in q['development_targets']:
        t=r['target'];key=(t['pair_id'],*instance(t['truth']))
        if t['variant'] in groups.setdefault(key,{}):raise ValueError('Duplicate same-pose instance')
        groups[key][t['variant']]=r
    dev=[];dev_pages=[];frame_groups={}
    for i,(key,variants) in enumerate(sorted(groups.items()),1):
        if set(variants)!=set(VARIANTS):raise ValueError('Incomplete four-condition target')
        tile=Image.new('RGB',(1600,270),'white');items=[]
        for j,v in enumerate(VARIANTS):
            r=variants[v];s=r['source'];t=r['target'];im=Image.open(s['image_path']).convert('RGB')
            if file_sha256(s['image_path'])!=s['image_sha256']:raise ValueError('Stale development source')
            tile.paste(fitted(crop(im,t['truth']['bbox_xyxy']),(395,220)),(j*400,45))
            label=f'D{i:03}-{v[0]} {v} {t["truth"]["class_name"]}'
            ImageDraw.Draw(tile).text((j*400+3,4),label,font=FONT,fill='black')
            items.append(dict(review_id=f'D{i:03}-{v}',**r));paths.append(Path(s['image_path']))
            frame_groups.setdefault(t['pair_id'],{})[v]=s
        path=OUT/f'D{i:03}.png';tile.save(path);paths.append(path)
        for r in items:r.update(evidence_path=str(path),evidence_sha256=file_sha256(path))
        dev.extend(items)
    for n in range(0,len(groups),4):
        page=Image.new('RGB',(1600,1080),'white')
        for j in range(min(4,len(groups)-n)):page.paste(Image.open(OUT/f'D{n+j+1:03}.png'),(0,j*270))
        path=OUT/f'dev-page-{n//4+1:02}.png';page.save(path);paths.append(path);dev_pages.append(str(path))
    frame_pages=[];frame_lookup={}
    pairs=sorted(frame_groups)
    for n in range(0,len(pairs),3):
        page=Image.new('RGB',(1600,810),'white')
        for j,pair in enumerate(pairs[n:n+3]):
            for k,v in enumerate(VARIANTS):
                s=frame_groups[pair][v];im=Image.open(s['image_path']).convert('RGB');d=ImageDraw.Draw(im)
                for r in dev:
                    if r['target']['pair_id']==pair and r['target']['variant']==v:d.rectangle(r['target']['truth']['bbox_xyxy'],outline='lime',width=3)
                page.paste(fitted(im,(395,220)),(k*400,j*270+40))
                ImageDraw.Draw(page).text((k*400+2,j*270+2),f'F{n+j+1:02} {v}',font=FONT,fill='black')
        path=OUT/f'frames-{n//3+1:02}.png';page.save(path);paths.append(path);frame_pages.append(str(path))
        for pair in pairs[n:n+3]:frame_lookup[pair]=str(path)
    for r in dev:r['full_context_path']=frame_lookup[r['target']['pair_id']];r['full_context_sha256']=file_sha256(r['full_context_path'])
    frozen(dest,dict(status='awaiting_explicit_visual_decisions',training=train,development=dev,train_pages=train_pages,dev_pages=dev_pages,frame_pages=frame_pages,
        inputs={str(x):file_sha256(x) for x in paths}))
    print('EVIDENCE',len(train),len(dev),'PAGES',len(train_pages),len(dev_pages),len(frame_pages))


if __name__=='__main__':main()
