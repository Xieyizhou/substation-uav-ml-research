"""Identity-resolved error evidence and transitions; no automatic review decisions."""
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw
from scripts.vision.evaluate_frozen_multiscale import OUT,KEYS,prior,validate_record
from scripts.vision.analyze_recovery_paired_calibration import iou

DEST=OUT/'error-diagnosis'

def truth_key(t): return (t['class_name'],tuple(t['bbox_xyxy']))

def main():
    DEST.mkdir(exist_ok=True)
    src=OUT.parent/'material-control-feasibility-v1/initial-gate.json'
    identity=prior.read(src);prior.verify(identity)
    paths=[src,Path(__file__).resolve()]
    records={}
    for key in KEYS:
        path=OUT/'evaluation'/f'{key}.json';r=prior.read(path);validate_record(r,key);records[key]=r;paths.append(path)
    identities={}
    for r in identity['corrected_target_records']:
        k=(r['pair_id'],r['variant'],truth_key(r['truth']))
        if k in identities: raise ValueError('Duplicate resolved identity')
        identities[k]=r
    transitions=[];losses={}
    for seed in (7,17,27):
        a={(r['pair_id'],r['variant']):r for r in records[f'fixed-{seed}']['rows']}
        for b in records[f'multiscale-{seed}']['rows']:
            old=a[(b['pair_id'],b['variant'])]
            ot={truth_key(t):i for i,t in enumerate(old['truth'])}
            if len(ot)!=len(old['truth']) or set(ot)!={truth_key(t) for t in b['truth']}:raise ValueError('Truth identity mismatch')
            hit_a={m['truth_index'] for m in old['matches']};hit_b={m['truth_index'] for m in b['matches']}
            for j,t in enumerate(b['truth']):
                k=(b['pair_id'],b['variant'],truth_key(t));source=identities[k]
                if source['image_sha256']!=b['image_sha256'] or source['truth']!=t:raise ValueError('Source mismatch')
                before=ot[truth_key(t)] in hit_a;after=j in hit_b
                state='retained_hit' if before and after else 'loss' if before else 'gain' if after else 'retained_miss'
                best={}
                for name,row in (('fixed',old),('multiscale',b)):
                    same=[p for p in row['low_predictions'] if p['class_name']==t['class_name'] and iou(p['bbox_xyxy'],t['bbox_xyxy'])>=.5]
                    best[name]=max(same,key=lambda p:p['confidence'],default=None)
                miss=next((m for m in b['misses'] if m['truth_index']==j),None)
                item=dict(seed=seed,pair_id=b['pair_id'],variant=b['variant'],object_id=source['object_id'],truth=t,state=state,miss=miss,best_matching_same_class=best)
                transitions.append(item)
                if state=='loss' and (b['variant'] in ('original','lighting') or (b['variant']=='material' and t['class_name']=='reactor')):
                    loss=losses.setdefault(k,dict(kind='LOSS',source=source,truth=t,events=[]))
                    loss['events'].append(item)
    _,p,_=__import__('scripts.vision.train_frozen_multiscale',fromlist=['contract']).contract(KEYS[0])
    np=Path(p['evaluation']['negative_review']);negative=prior.read(np);prior.verify(negative);paths.append(np)
    negmap={(r['view_id'],r['variant']):r for r in negative['frames']}
    fps={}
    for seed in (7,17,27):
        for r in records[f'multiscale-{seed}']['negative_rows']:
            for prediction in r['predictions']:
                k=(r['view_id'],r['variant']);source=negmap[k]
                fp=fps.setdefault(k,dict(kind='FP',source=source,events=[]))
                fp['events'].append(dict(seed=seed,prediction=prediction))
    events=[]
    for n,x in enumerate(list(fps.values())+list(losses.values()),1):
        x['event_id']=f'E{n:02}'
        source=x['source'];path=Path(source['image_path'])
        if prior.file_sha256(path)!=source['image_sha256']:raise ValueError('Changed image')
        paths.append(path)
        im=Image.open(path).convert('RGB');overlay=im.copy();d=ImageDraw.Draw(overlay)
        boxes=[y['prediction']['bbox_xyxy'] for y in x['events']] if x['kind']=='FP' else [x['truth']['bbox_xyxy']]
        for j,box in enumerate(boxes):d.rectangle(box,outline='red',width=5);d.text((box[0],box[1]),str(j),fill='red')
        card=Image.new('RGB',(1000,370+220*len(boxes)),'white');cd=ImageDraw.Draw(card)
        label=source.get('object_id','negative')
        cd.text((5,5),f"{x['event_id']} {x['kind']} {source['variant']} {label}",fill='black')
        overlay.thumbnail((980,330));card.paste(overlay,(5,30))
        for j,box in enumerate(boxes):
            crop=im.crop(tuple(map(round,box)));crop.thumbnail((780,195));card.paste(crop,(5,390+j*220))
            desc=str([(e['seed'],e.get('miss',{}).get('reason')) for e in x['events']]) if x['kind']=='LOSS' else str(x['events'][j])
            cd.text((5,370+j*220),desc[:155],fill='black')
        cp=DEST/f"{x['event_id']}.png";card.save(cp);paths.append(cp)
        x['page_path']=str(cp);x['page_sha256']=prior.file_sha256(cp);events.append(x)
    pages=[]
    for start in range(0,len(events),4):
        chunk=events[start:start+4];heights=[Image.open(x['page_path']).height for x in chunk];h=max(heights)
        sheet=Image.new('RGB',(2000,h*2),'white')
        for i,x in enumerate(chunk):sheet.paste(Image.open(x['page_path']),((i%2)*1000,(i//2)*h))
        path=DEST/f'page-{start//4+1:02}.png';sheet.save(path);paths.append(path);pages.append(str(path))
    counts=Counter(f"{x['variant']}:{x['state']}" for x in transitions)
    prior.frozen(DEST/'evidence.json',dict(status='awaiting_explicit_visual_review',events=events,transitions=transitions,pages=pages,
        transition_counts=dict(counts),inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('EVENTS',len(events),'FP frames',len(fps),'LOSS boxes',len(losses),'PAGES',len(pages))
    for x in events: print(x['event_id'],x['kind'],x['source']['variant'],x['source'].get('object_id'),len(x['events']))

if __name__=='__main__':main()
