"""Freeze this experiment's actual FP boxes and paired instance transitions."""
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.lineage_capped_control import OUT as CONTROL,RUN,ROOT,read,verify,frozen,file_sha256
from scripts.vision.analyze_recovery_paired_calibration import iou

OUT=CONTROL/'error-review-v1'

def main():
    p=read(CONTROL/'protocol.json');verify(p);verify(read(CONTROL/'evaluation/summary.json'))
    sources={}
    for field in ('paired_review','negative_review'):
        r=read(p['evaluation'][field]);verify(r)
        sources.update({(f['view_id'],f['variant']):f for f in r['frames']})
    paths=[CONTROL/'protocol.json',CONTROL/'evaluation/summary.json',Path(__file__)]
    negatives={};losses={};transitions=[]
    for seed in (7,17,27):
        np=CONTROL/'evaluation'/f'lineage-capped-450-{seed}.json';new=read(np);verify(new);paths.append(np)
        for r in new['negative_rows']:
            if not r['predictions']:continue
            key=(r['view_id'],r['variant']);f=negatives.setdefault(key,dict(source=sources[key],predictions=[]))
            f['predictions'].extend(dict(**x,seed=seed,prediction_index=i) for i,x in enumerate(r['predictions']))
        for arm in ('reference','hold'):
            op=RUN/'evaluation'/f'{arm}-450-{seed}.json';old=read(op);verify(old);paths.append(op)
            oldrows={(r['view_id'],r['variant']):r for r in old['rows']}
            for r in new['rows']:
                key=(r['view_id'],r['variant']);a=oldrows[key]
                if a['image_sha256']!=r['image_sha256'] or a['truth']!=r['truth']:raise ValueError('Paired identities changed')
                ah={m['truth_index'] for m in a['matches']};bh={m['truth_index'] for m in r['matches']}
                for i,t in enumerate(r['truth']):
                    state='persistent_hit' if i in ah and i in bh else 'gain' if i in bh else 'loss' if i in ah else 'persistent_miss'
                    transitions.append(dict(seed=seed,comparison=arm,view_id=key[0],variant=key[1],annotation_id=t['annotation_id'],truth=t,state=state))
                    if state!='loss' or key[1] not in ('original','lighting') or t['class_name'] not in ('reactor','capacitor_bank'):continue
                    lk=(*key,t['annotation_id']);f=losses.setdefault(lk,dict(source=sources[key],truth=t,events=[]))
                    reason=next(x for x in r['misses'] if x['truth_index']==i)
                    low=sorted([dict(**x,iou=iou(t['bbox_xyxy'],x['bbox_xyxy'])) for x in r['low_predictions']],key=lambda x:x['iou'],reverse=True)[:3]
                    f['events'].append(dict(seed=seed,comparison=arm,reason=reason,best_low=low,planned=i==r['planned_truth_index']))
    OUT.mkdir(exist_ok=True);events=[];tiles=[]
    for kind,entries in [('FP',sorted(negatives.items())),('LOSS',sorted(losses.items()))]:
        for number,(_,f) in enumerate(entries,1):
            eid=f'{kind}{number:02}';s=f['source'];im=Image.open(s['image_path']).convert('RGB')
            if file_sha256(s['image_path'])!=s['image_sha256']:raise ValueError('Stale source')
            paths.append(Path(s['image_path']));full=im.copy();draw=ImageDraw.Draw(full)
            boxes=[x['bbox_xyxy'] for x in f['predictions']] if kind=='FP' else [f['truth']['bbox_xyxy']]
            crops=[]
            for j,b in enumerate(boxes):
                draw.rectangle(b,outline='red' if kind=='FP' else 'lime',width=3);draw.text((b[0],b[1]),str(j),fill='red')
                cp=OUT/f'{eid}-box{j}.png';im.crop(b).save(cp);paths.append(cp);crops.append(dict(path=str(cp),sha256=file_sha256(cp)))
            ep=OUT/(eid+'-full.png');full.save(ep);paths.append(ep)
            union=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
            crop=im.crop(union);crop.thumbnail((395,400));full.thumbnail((790,445));tile=Image.new('RGB',(1200,500),'white');tile.paste(full,(0,45));tile.paste(crop,(800,45));ImageDraw.Draw(tile).text((5,5),eid+' '+s['variant']+((' '+f['truth']['class_name']) if kind=='LOSS' else ''),fill='black')
            tilep=OUT/(eid+'.png');tile.save(tilep);paths.append(tilep);tiles.append(tile)
            events.append(dict(event_id=eid,kind=kind,**f,crops=crops,full_path=str(ep),full_sha256=file_sha256(ep),evidence_path=str(tilep),evidence_sha256=file_sha256(tilep)))
    for i in range(0,len(tiles),4):
        page=Image.new('RGB',(2400,1000),'white')
        for j,t in enumerate(tiles[i:i+4]):page.paste(t,((j%2)*1200,(j//2)*500))
        path=OUT/f'page-{i//4+1:02}.png';page.save(path);paths.append(path)
    frozen(OUT/'evidence.json',dict(status='awaiting_explicit_AI_review',events=events,all_instance_transitions=transitions,
        inputs={str(x):file_sha256(x) for x in paths}))
    print('REVIEW_FRAMES',len(events),'FP_IMAGES',len(negatives),'TARGET_LOSS_INSTANCES',len(losses))

if __name__=='__main__':main()
