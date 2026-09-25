"""Build current diagnostic review evidence, never reuse old decisions."""
from pathlib import Path
from PIL import Image, ImageDraw
from scripts.vision.diagnose_bn_statistics import OUT,SOURCE,prior
from scripts.vision import build_physical_low_light_review as renderer

DEST=OUT/'error-review-v1'
def render(x):
    source=x['source'];im=Image.open(source['image_path']).convert('RGB')
    if prior.file_sha256(source['image_path'])!=source['image_sha256']:raise ValueError('Changed RGB')
    boxes=[r['prediction']['bbox_xyxy'] for r in x['events']] if x['kind']=='FP' else [x['truth']['bbox_xyxy']]
    overlay=im.copy();draw=ImageDraw.Draw(overlay)
    for j,b in enumerate(boxes):draw.rectangle(b,outline='red',width=4)
    card=Image.new('RGB',(1000,370+220*len(boxes)),'white');d=ImageDraw.Draw(card)
    d.text((5,5),f"{x['event_id']} {x['kind']} {source.get('object_id','negative')} {source['variant']}",fill='black')
    overlay.thumbnail((980,330));card.paste(overlay,(5,30));x['empty_crop_indices']=[]
    for j,b in enumerate(boxes):
        bb=tuple(map(round,b));d.text((5,370+j*220),f"box {j} {b}",fill='black')
        if bb[2]<=bb[0] or bb[3]<=bb[1]:
            x['empty_crop_indices'].append(j);d.text((5,410+j*220),'EMPTY ROUNDED ROI - no content certification; prediction retained',fill='red')
        else:
            crop=im.crop(bb);crop.thumbnail((780,195));card.paste(crop,(5,390+j*220))
    cp=DEST/f"{x['event_id']}.png";card.save(cp);x['page_path']=str(cp);x['page_sha256']=prior.file_sha256(cp);return x
def main():
    DEST.mkdir(parents=True,exist_ok=True);renderer.DEST=DEST
    sp=OUT/'summary.json';s=prior.read(sp);prior.verify(s)
    oldp=SOURCE/'evaluation/error-review-v1/evidence.json';old=prior.read(oldp);prior.verify(old)
    identities={(x['source']['image_sha256'],x['truth']['class_name'],tuple(x['truth']['bbox_xyxy'])):x['source'] for x in old['events'] if x['kind']=='LOSS'}
    ip=renderer.IDENTITY/'initial-gate.json';identity=prior.read(ip);prior.verify(identity)
    for r in identity['corrected_target_records']:identities[r['image_sha256'],r['truth']['class_name'],tuple(r['truth']['bbox_xyxy'])]=r
    p=prior.read(SOURCE/'design.json');np=Path(p['evaluation']['negative_review']);neg=prior.read(np);prior.verify(neg)
    negatives={r['image_sha256']:r for r in neg['frames']};groups={}
    for r in s['review_queue']:
        if r['kind']=='FP':
            k=('FP',r['image_sha256']);source=negatives[r['image_sha256']]
            event=dict(seed=r['seed'],cell=f"{r['condition']}-{r['seed']}",prediction=r['prediction'])
            x=groups.setdefault(k,dict(kind='FP',source=source,events=[]))
        else:
            t=r['truth'];k=('LOSS',r['image_sha256'],t['class_name'],tuple(t['bbox_xyxy']))
            source=identities[k[1:]];event=r
            x=groups.setdefault(k,dict(kind='LOSS',source=source,truth=t,events=[]))
        x['events'].append(event)
    events=[];deps=[sp,ip,np,Path(__file__).resolve()]
    for i,x in enumerate(groups.values(),1):
        x['event_id']=f'E{i:02}';events.append(render(x));deps.extend([Path(x['source']['image_path']),Path(x['page_path'])])
    pages=[]
    for start in range(0,len(events),4):
        chunk=events[start:start+4];h=max(Image.open(x['page_path']).height for x in chunk)
        sheet=Image.new('RGB',(2000,h*2),'white')
        for i,x in enumerate(chunk):sheet.paste(Image.open(x['page_path']),((i%2)*1000,(i//2)*h))
        path=DEST/f'page-{start//4+1:02}.png';sheet.save(path);pages.append(str(path));deps.append(path)
    prior.frozen(DEST/'evidence.json',dict(status='awaiting_explicit_review',events=events,pages=pages,inputs={str(d):prior.file_sha256(d) for d in deps}))
    print(len(events),'cards',len(pages),'pages')

if __name__=='__main__':main()
