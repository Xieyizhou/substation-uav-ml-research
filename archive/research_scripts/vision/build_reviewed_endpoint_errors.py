"""Freeze same-image truth correspondence and review pages; never decides review."""
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.train_order_fit_reviewed import DEST,KEYS,prior
from scripts.vision.diagnose_small_scale_order_fit import TRAIN

OUT=DEST/'evaluation/endpoint-review-v1'
def truth_key(t): return (t['class_name'],tuple(t['bbox_xyxy']),t.get('annotation_id'))

def build():
    OUT.mkdir(exist_ok=True); dest=OUT/'evidence.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    p=prior.read(DEST/'design.json');prior.verify(p);rp=Path(p['evaluation']['paired_review']);rv=prior.read(rp);prior.verify(rv)
    sources={r['image_sha256']:r for r in rv['frames']}; groups={}; transitions=[]; deps=[rp,DEST/'design.json',Path(__file__).resolve()]
    for cell in KEYS:
        paths=[root/'evaluation'/f'{cell}.json' for root in (TRAIN,DEST)]
        old,new=[prior.read(x) for x in paths]
        for r in (old,new):prior.verify(r)
        deps+=paths; index={(r['pair_id'],r['variant'],r['image_sha256']):r for r in old['rows']}
        if len(index)!=len(old['rows']):raise ValueError('Duplicate frame')
        for row in new['rows']:
            if row['variant'] not in ('original','lighting'):continue
            before=index[(row['pair_id'],row['variant'],row['image_sha256'])]
            a={truth_key(t):i for i,t in enumerate(before['truth'])};b={truth_key(t):i for i,t in enumerate(row['truth'])}
            if len(a)!=len(before['truth']) or a.keys()!=b.keys():raise ValueError('Truth identity/coordinate conflict')
            oldhit={m['truth_index'] for m in before['matches']};newhit={m['truth_index'] for m in row['matches']}
            for tk,j in b.items():
                state=('persistent_hit' if j in newhit else 'new_miss') if a[tk] in oldhit else ('gain' if j in newhit else 'persistent_miss')
                tr=dict(cell=cell,seed=int(cell.split('-')[-1]),pair_id=row['pair_id'],variant=row['variant'],image_sha256=row['image_sha256'],truth=row['truth'][j],state=state)
                transitions.append(tr)
                if state!='new_miss':continue
                gkey=(row['image_sha256'],tk)
                g=groups.setdefault(gkey,dict(source=sources[row['image_sha256']],truth=row['truth'][j],events=[]))
                g['events'].append(dict(tr,miss=next((m for m in row['misses'] if m.get('truth_index')==j),None),predictions=row['predictions'],low_predictions=row['low_predictions']))
    records=[]
    for i,(_,g) in enumerate(sorted(groups.items(),key=lambda x:str(x[0])),1):
        im=Image.open(g['source']['image_path']).convert('RGB');sha=g['source']['image_sha256']
        if prior.file_sha256(g['source']['image_path'])!=sha:raise ValueError('Image drift')
        box=g['truth']['bbox_xyxy']; overlay=im.copy();ImageDraw.Draw(overlay).rectangle(box,outline='red',width=5)
        overlay.thumbnail((960,540));card=Image.new('RGB',(1440,820),'white');card.paste(overlay,(0,35));d=ImageDraw.Draw(card)
        name=f'M{i:02}';d.text((5,5),f'{name} {g["source"]["variant"]} {g["truth"]["class_name"]} events={len(g["events"])}',fill='black')
        crop=im.crop(tuple(map(round,box)));crop.thumbnail((460,560));card.paste(crop,(975,35))
        for n,e in enumerate(g['events']):d.text((5,590+n*22),f'{e["cell"]} miss={e["miss"]}',fill='black')
        page=OUT/f'{name}.png';card.save(page);deps += [Path(g['source']['image_path']),page]
        records.append(dict(review_id=name,page=str(page),**g))
    return prior.frozen(dest,dict(status='explicit_review_pending',new_miss_events=sum(len(r['events']) for r in records),unique_targets=len(records),targets=records,
        transitions=transitions,transition_counts=dict(Counter(t['state'] for t in transitions)),
        identity_scope='Same immutable image, complete truth class/coordinate/annotation identity; no cross-world runtime-ID inference.',
        inputs={str(d):prior.file_sha256(d) for d in deps}))

if __name__=='__main__':
    r=build();print(r['new_miss_events'],r['unique_targets'],r['transition_counts'])
