"""Freeze all paired losses and focused visual evidence, without review decisions."""
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.train_whole_image_hold import OUT as RUN,read,verify,frozen,file_sha256,ROOT
from scripts.vision.analyze_recovery_paired_calibration import iou

OUT=RUN/'error-localization-v1'

def generate():
    OUT.mkdir(exist_ok=True);p=read(RUN/'protocol.json');verify(p)
    frames=read(p['evaluation']['paired_review'])['frames'];negative=read(p['evaluation']['negative_review'])['frames']
    sources={(r['view_id'],r['variant']):r for r in frames+negative}
    losses={};neg={};paths=[RUN/'protocol.json',RUN/'evaluation/completion.json',Path(__file__)]
    for seed in (7,17,27):
        ap=RUN/'evaluation'/f'reference-450-{seed}.json';bp=RUN/'evaluation'/f'hold-450-{seed}.json'
        a=read(ap);b=read(bp);verify(a);verify(b);paths.extend([ap,bp])
        aa={(r['view_id'],r['variant']):r for r in a['rows']}
        for r in b['rows']:
            old=aa[r['view_id'],r['variant']]
            if old['truth']!=r['truth'] or old['image_sha256']!=r['image_sha256']:raise ValueError('Changed truth identity')
            ah={m['truth_index'] for m in old['matches']};bh={m['truth_index'] for m in r['matches']}
            for i in ah-bh:
                t=r['truth'][i];key=(r['view_id'],r['variant'],t['annotation_id'])
                if key not in losses:losses[key]=dict(view_id=r['view_id'],variant=r['variant'],truth=t,events=[],source=sources[r['view_id'],r['variant']])
                reason=next(x for x in r['misses'] if x['truth_index']==i)
                relevant=sorted([dict(**x,iou=iou(t['bbox_xyxy'],x['bbox_xyxy'])) for x in r['low_predictions']],key=lambda x:x['iou'],reverse=True)[:3]
                losses[key]['events'].append(dict(seed=seed,planned=i==r['planned_truth_index'],**reason,best_low=relevant))
        for arm,record in [('reference',a),('hold',b)]:
            for r in record['negative_rows']:
                if not r['predictions']:continue
                key=(r['view_id'],r['variant'])
                if key not in neg:neg[key]=dict(view_id=key[0],variant=key[1],source=sources[key],predictions=[])
                neg[key]['predictions'] += [dict(**x,arm=arm,seed=seed) for x in r['predictions']]
    lossrows=sorted(losses.values(),key=lambda r:(r['variant'],r['view_id'],r['truth']['annotation_id']))
    for i,r in enumerate(lossrows):r['event_id']=f'L{i+1:03}'
    focus=[r for r in lossrows if r['variant']=='background' or (r['variant']=='original' and r['truth']['class_name'] in ('reactor','capacitor_bank'))]
    negrows=sorted(neg.values(),key=lambda r:(r['view_id'],r['variant']))
    for i,r in enumerate(negrows):r['event_id']=f'N{i+1:02}'
    def render(rows,kind):
        tiles=[]
        for r in rows:
            s=r['source'];im=Image.open(s['image_path']).convert('RGB')
            if file_sha256(s['image_path'])!=s['image_sha256']:raise ValueError('Changed image')
            paths.append(Path(s['image_path']));full=im.copy();d=ImageDraw.Draw(full)
            boxes=[r['truth']['bbox_xyxy']] if kind=='loss' else [x['bbox_xyxy'] for x in r['predictions']]
            for j,box in enumerate(boxes):d.rectangle(box,outline='lime' if kind=='loss' else 'red',width=4);d.text((box[0],box[1]),str(j),fill='red')
            # Union crop preserves exact box coverage; each negative's per-box evidence also saved.
            box=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
            tile=Image.new('RGB',(1200,500),'white');full.thumbnail((790,445));tile.paste(full,(0,45))
            crop=im.crop(box);crop.thumbnail((395,440));tile.paste(crop,(800,45))
            title=r['event_id']+' '+r['variant']+' '+(r['truth']['class_name'] if kind=='loss' else 'negative')
            ImageDraw.Draw(tile).text((8,10),title,fill='black');path=OUT/(r['event_id']+'.png');tile.save(path);paths.append(path)
            r['evidence_path']=str(path);r['evidence_sha256']=file_sha256(path);tiles.append(tile)
        for i in range(0,len(tiles),4):
            page=Image.new('RGB',(2400,1000),'white')
            for j,t in enumerate(tiles[i:i+4]):page.paste(t,((j%2)*1200,(j//2)*500))
            path=OUT/f'{kind}-page-{i//4+1:02}.png';page.save(path);paths.append(path)
    render(focus,'loss');render(negrows,'negative')
    return frozen(OUT/'evidence.json',dict(status='evidence_only_not_reviewed',all_losses=lossrows,focus_ids=[r['event_id'] for r in focus],negative=negrows,
        inputs={str(x):file_sha256(x) for x in paths}))

if __name__=='__main__':
    r=generate();print('LOSSES',len(r['all_losses']),'FOCUS',len(r['focus_ids']),'NEGATIVE_IMAGES',len(r['negative']))
