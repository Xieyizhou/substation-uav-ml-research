"""Identity-matched whole-frame transitions and new-loss visual evidence."""
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.interleaved_small_scale_control import OUT,KEYS,prior,source

DEST=OUT/'evaluation/loss-review-v1'


def truth_key(t):
    return t['annotation_id'],t['class_name'],tuple(t['bbox_xyxy'])


def transitions(a,b):
    if a['image_sha256']!=b['image_sha256'] or (a['pair_id'],a['variant'])!=(b['pair_id'],b['variant']):
        raise ValueError('Frame pairing conflict')
    left={truth_key(t):(i,t) for i,t in enumerate(a['truth'])}
    right={truth_key(t):(i,t) for i,t in enumerate(b['truth'])}
    if len(left)!=len(a['truth']) or len(right)!=len(b['truth']) or set(left)!=set(right):
        raise ValueError('Truth identity/coordinate conflict')
    ah={m['truth_index'] for m in a['matches']}; bh={m['truth_index'] for m in b['matches']}
    result=[]
    for k,(i,t) in left.items():
        j,_=right[k]; before=i in ah; after=j in bh
        state='retained_hit' if before and after else 'loss' if before else 'gain' if after else 'persistent_miss'
        misses=[m for m in b['misses'] if m['truth_index']==j]
        if not after and len(misses)!=1: raise ValueError('Missing/duplicate miss diagnosis')
        result.append(dict(truth=t,transition=state,before_hit=before,after_hit=after,
            miss=misses[0] if misses else None,planned=j==b['planned_truth_index']))
    return result


def build():
    DEST.mkdir(exist_ok=True); dest=DEST/'evidence.json'
    if dest.exists():
        e=prior.read(dest); prior.verify(e); return e
    p=prior.read(OUT/'design.json'); prior.verify(p)
    rp=Path(p['evaluation']['paired_review']); review=prior.read(rp); prior.verify(review)
    src={r['image_sha256']:r for r in review['frames']}
    deps=[rp,OUT/'design.json',Path(__file__).resolve()]; matrix=[]; grouped={}
    for key in KEYS:
        bp=source.OUT/'evaluation'/f'{key[1:]}.json'; ap=OUT/'evaluation'/f'{key}.json'
        before=prior.read(bp); after=prior.read(ap)
        for r in (before,after): prior.verify(r)
        deps += [bp,ap]
        old={r['image_sha256']:r for r in before['rows']}
        if len(old)!=48 or len(after['rows'])!=48: raise ValueError('Incomplete paired data')
        for row in after['rows']:
            if row['matching_conflict']: raise ValueError('Unresolved matching conflict')
            for item in transitions(old[row['image_sha256']],row):
                event=dict(cell=key,image_sha256=row['image_sha256'],variant=row['variant'],pair_id=row['pair_id'],**item)
                matrix.append(event)
                if item['transition']!='loss' or row['variant'] not in ('original','lighting'): continue
                identity=(row['image_sha256'],truth_key(item['truth']))
                g=grouped.setdefault(identity,dict(source=src[row['image_sha256']],truth=item['truth'],events=[]))
                g['events'].append(dict(event,low_predictions=row['low_predictions']))
    cards=[]; objects=[]
    for i,g in enumerate(grouped.values(),1):
        ip=Path(g['source']['image_path']); deps.append(ip)
        if prior.file_sha256(ip)!=g['source']['image_sha256']: raise ValueError('Image drift')
        im=Image.open(ip).convert('RGB'); t=g['truth']; box=t['bbox_xyxy']
        full=im.copy(); d=ImageDraw.Draw(full); d.rectangle(box,outline='lime',width=4)
        full.thumbnail((940,530)); crop=im.crop(tuple(map(round,box))); crop.thumbnail((900,450))
        page=Image.new('RGB',(1600,1100),'white'); page.paste(full,(0,30)); page.paste(crop,(0,590)); d=ImageDraw.Draw(page)
        d.text((5,5),f'L{i:02} {g["source"]["variant"]} {t["class_name"]} {t["annotation_id"]}',fill='black')
        for j,event in enumerate(g['events']):
            d.text((960,30+j*150),f'{event["cell"]}\n{event["miss"]["reason"]}\nplanned={event["planned"]}',fill='black')
        path=DEST/f'L{i:02}.png'; page.save(path); deps.append(path); cards.append(str(path))
        objects.append(dict(review_id=f'L{i:02}',page_path=str(path),**g))
    return prior.frozen(dest,dict(status='explicit_loss_review_pending',matrix=matrix,objects=objects,pages=cards,
        new_loss_events=sum(len(g['events']) for g in objects),unique_loss_targets=len(objects),
        transition_counts=dict(Counter(e['transition'] for e in matrix)),
        inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__=='__main__':
    r=build(); print(r['new_loss_events'],r['unique_loss_targets'],r['transition_counts'])
