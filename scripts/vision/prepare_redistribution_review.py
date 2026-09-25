"""Increased-member source audit and visual evidence; no automatic admission."""
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.relax_hold_compensation import OUT as COUNTS,RUN,ROOT,ready,read,verify,frozen,file_sha256
from scripts.vision.structure_fit import OUT as FIT,truth_for
from scripts.vision.check_structure_fit_sources import equal_rgb,label_correspondence

OUT=RUN/'positive-redistribution-pretraining-v1'

def build():
    p=ready();c=read(COUNTS/'counts.json');verify(c);verify(read(COUNTS/'completion.json'))
    source=read(FIT/'member-source-trace.json');verify(source)
    idx={r['member_id']:r for r in p['pool_rows']};sources={r['member_id']:r for r in source['rows']}
    increased=sorted({d['member_id'] for r in c['results'].values() for d in r['changes'] if d['delta']>0})
    OUT.mkdir(exist_ok=True);events=[];paths=[COUNTS/'counts.json',COUNTS/'completion.json',FIT/'member-source-trace.json',Path(__file__)];tiles=[]
    for i,m in enumerate(increased):
        r=idx[m];s=sources[m];t=truth_for(r)
        for kind in ('image','label'):
            if file_sha256(r[kind+'_path'])!=r[kind+'_sha256']:raise ValueError('Stale input')
            paths.append(Path(r[kind+'_path']))
        equal_rgb(r['image_path'],s['source_image'])
        cp=Path(s['source_image']).parents[1]/'collection-receipt.json';v=[v for v in read(cp)['views'] if v['view_id']==s['source_view_id']]
        if len(v)!=1:raise ValueError('Source view collision')
        label_correspondence(t,v[0]['truth']['objects']);paths.extend([cp,Path(s['source_image'])])
        im=Image.open(r['image_path']).convert('RGB');d=ImageDraw.Draw(im)
        for j,x in enumerate(t):
            b=x['bbox_xyxy'];d.rectangle(b,outline='lime',width=4);d.text((max(0,b[0]),max(0,b[1])),f"{j}:{x['class_name']}",fill='red')
        im.thumbnail((1000,560));tile=Image.new('RGB',(1000,620),'white');tile.paste(im,(0,45))
        eid=f'P{i+1:02}';ImageDraw.Draw(tile).text((10,10),eid+' '+r['subset']+' labels='+str(len(t)),fill='black')
        ep=OUT/(eid+'.png');tile.save(ep);paths.append(ep);tiles.append(tile)
        events.append(dict(event_id=eid,member=r,truth=t,source=s,evidence_path=str(ep),evidence_sha256=file_sha256(ep),
            exact_RGB_and_full_labels_verified=True,exposures={seed:dict(before=next((d['before'] for d in x['changes'] if d['member_id']==m),x['counts'][m]),after=x['counts'][m]) for seed,x in c['results'].items()}))
    for i in range(0,len(tiles),4):
        page=Image.new('RGB',(2000,1240),'white')
        for j,t in enumerate(tiles[i:i+4]):page.paste(t,((j%2)*1000,(j//2)*620))
        path=OUT/f'page-{i//4+1:02}.png';page.save(path);paths.append(path)
    frozen(OUT/'evidence.json',dict(status='sources_verified_visual_review_pending',events=events,inputs={str(x):file_sha256(x) for x in paths}))
    print('SOURCE_VERIFIED',len(events))

if __name__=='__main__':build()
