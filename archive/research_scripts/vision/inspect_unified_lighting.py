"""Render new six-cell error evidence, never synthesizing review decisions."""
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.evaluate_unified_lighting import OUT,KEYS,prior,ready,validate_record
from scripts.vision.unified_hold_lighting_counts import SOURCE
from scripts.vision.analyze_recovery_paired_calibration import iou

def aligned(a,b):
    def index(rows):
        out={}
        for r in rows:
            k=(r['pair_id'],r['variant'])
            if k in out: raise ValueError('Duplicate frame identity')
            out[k]=r
        return out
    x,y=index(a),index(b)
    if x.keys()!=y.keys(): raise ValueError('Pair membership differs')
    for k,r in x.items():
        s=y[k]
        if r['image_sha256']!=s['image_sha256']: raise ValueError('Image identity changed')
        tx={t['annotation_id']:t for t in r['truth']};ty={t['annotation_id']:t for t in s['truth']}
        if len(tx)!=len(r['truth']) or len(ty)!=len(s['truth']) or tx!=ty: raise ValueError('Truth identity/coordinates conflict')
    return x,y

def main():
    p=ready();dest=OUT/'error-review/evidence.json'
    if dest.exists(): prior.verify(prior.read(dest));return
    sp=OUT/'evaluation/summary.json';prior.verify(prior.read(sp));paths=[sp,Path(__file__).resolve()]
    sources={}
    for field in ('paired_review','negative_review'):
        rp=Path(p['evaluation'][field]);r=prior.read(rp);prior.verify(r);paths.append(rp)
        sources.update({(x['view_id'],x['variant']):x for x in r['frames']})
    records={};fps={};losses={};transitions=[]
    for key in KEYS:
        ep=OUT/'evaluation'/f'{key}.json';r=prior.read(ep);validate_record(r,key);records[key]=r;paths.append(ep)
        for row in r['negative_rows']:
            k=(row['view_id'],row['variant'])
            for j,pred in enumerate(row['predictions']): fps.setdefault(k,[]).append(dict(**pred,cell=key,prediction_index=j))
    for key,n in records.items():
        seed=int(key.split('-')[-1]);bp=SOURCE/'evaluation'/f'brightness-450-{seed}.json'
        b=prior.read(bp);prior.verify(b);paths.append(bp)
        comparisons=[('previous-brightness',b)]
        if key.startswith('L-'): comparisons.append(('same-seed-R-clean',records[f'R-clean-{seed}']))
        for reference,b in comparisons:
            old,new=aligned(b['rows'],n['rows'])
            for k,r in new.items():
                a=old[k];ah={a['truth'][m['truth_index']]['annotation_id'] for m in a['matches']};bh={r['truth'][m['truth_index']]['annotation_id'] for m in r['matches']}
                for j,t in enumerate(r['truth']):
                    tid=t['annotation_id'];state='persistent_hit' if tid in ah and tid in bh else 'gain' if tid in bh else 'loss' if tid in ah else 'persistent_miss'
                    transitions.append(dict(cell=key,reference=reference,pair_id=r['pair_id'],variant=r['variant'],truth=t,state=state))
                    if state=='loss' and r['variant'] in ('original','lighting'):
                        ek=(r['view_id'],r['variant'],tid);event=losses.setdefault(ek,dict(truth=t,events=[]))
                        event['events'].append(dict(cell=key,reference=reference,miss=next(m for m in r['misses'] if m['truth_index']==j),
                            low=sorted([dict(**x,iou=iou(t['bbox_xyxy'],x['bbox_xyxy'])) for x in r['low_predictions']],key=lambda x:x['iou'],reverse=True)[:5]))
    folder=dest.parent;folder.mkdir(exist_ok=True);events=[]
    for kind,data in (('FP',fps),('LOSS',losses)):
        for num,(key,value) in enumerate(sorted(data.items()),1):
            source=sources[key[:2]];ip=Path(source['image_path'])
            if prior.file_sha256(ip)!=source['image_sha256']:raise ValueError('Image changed')
            im=Image.open(ip).convert('RGB');full=im.copy();draw=ImageDraw.Draw(full);boxes=value if kind=='FP' else [value['truth']]
            eid=f'{kind}{num:02}';crops=[]
            for j,item in enumerate(boxes):
                b=item['bbox_xyxy'];draw.rectangle(b,outline='red',width=3);cp=folder/f'{eid}-{j}.png';im.crop(b).save(cp);paths.append(cp)
                crops.append(dict(path=str(cp),sha256=prior.file_sha256(cp)))
            full.thumbnail((960,540));canvas=Image.new('RGB',(1200,590+((len(crops)+2)//3)*270),'white');canvas.paste(full,(0,30));d=ImageDraw.Draw(canvas)
            d.text((0,5),eid+' '+source['variant'],fill='black')
            for j,crop in enumerate(crops):
                thumb=Image.open(crop['path']);thumb.thumbnail((390,230));x=j%3*400;y=590+j//3*270;canvas.paste(thumb,(x,y+30))
                d.text((x,y),f"{j} {boxes[j].get('class_name')} {boxes[j].get('cell','GT')}",fill='black')
            page=folder/f'{eid}.png';canvas.save(page);paths += [page,ip]
            events.append(dict(event_id=eid,kind=kind,source=source,predictions=value if kind=='FP' else [],loss=value if kind=='LOSS' else None,crops=crops,page=str(page),page_sha256=prior.file_sha256(page)))
    prior.frozen(dest,dict(status='awaiting_explicit_review',events=events,all_transitions=transitions,inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('FP_IMAGES',len(fps),'LOSS_INSTANCES',len(losses),flush=True)

if __name__=='__main__':main()
