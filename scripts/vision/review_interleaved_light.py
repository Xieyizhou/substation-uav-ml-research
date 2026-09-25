"""Current-run evidence; explicit decisions only, no training or label writes."""
from pathlib import Path
from scripts.vision.interleaved_light_control import OUT,REF,prior,KEYS
from scripts.vision.evaluate_interleaved_light import configure,adapter
configure()
complete=adapter.complete
from scripts.vision import build_bn_review as renderer
from scripts.vision.build_physical_low_light_review import IDENTITY,truth_key

DEST=OUT/'evaluation/error-review-v1'
def build():
    DEST.mkdir(parents=True,exist_ok=True);renderer.DEST=DEST
    ip=IDENTITY/'initial-gate.json';identity=prior.read(ip);prior.verify(identity)
    identities={}
    for x in identity['corrected_target_records']:
        k=(x['pair_id'],x['variant'],truth_key(x['truth']))
        if k in identities:raise ValueError('Duplicate identity')
        identities[k]=x
    p=prior.read(OUT/'design.json');prior.verify(p)
    np=Path(p['evaluation']['negative_review']);neg=prior.read(np);prior.verify(neg)
    negatives={(x['view_id'],x['variant']):x for x in neg['frames']}
    groups={};transitions=[];deps=[ip,np,OUT/'design.json',Path(__file__).resolve(),Path(renderer.__file__).resolve()]
    for key in KEYS:
        complete(key);seed=int(key.split('-')[-1]);ep=OUT/'evaluation'/f'{key}.json'
        r=prior.read(ep);prior.verify(r);deps.append(ep)
        for row in r['negative_rows']:
            src=negatives[row['view_id'],row['variant']]
            if src['image_sha256']!=row['image_sha256']:raise ValueError('Negative identity mismatch')
            for pred in row['predictions']:
                x=groups.setdefault(('FP',row['image_sha256']),dict(kind='FP',source=src,events=[]))
                x['events'].append(dict(seed=seed,cell=key,prediction=pred))
        for family,base in [('I1000',REF)]:
            rp=base/'evaluation'/f'{family}-{seed}.json';ref=prior.read(rp);prior.verify(ref);deps.append(rp)
            old={(x['pair_id'],x['variant']):x for x in ref['rows']}
            for row in r['rows']:
                a=old[row['pair_id'],row['variant']];indices={truth_key(t):j for j,t in enumerate(a['truth'])}
                if len(indices)!=len(a['truth']) or set(indices)!={truth_key(t) for t in row['truth']}:raise ValueError('Truth mismatch')
                ah={m['truth_index'] for m in a['matches']};bh={m['truth_index'] for m in row['matches']}
                for j,t in enumerate(row['truth']):
                    k=(row['pair_id'],row['variant'],truth_key(t));src=identities[k]
                    if src['image_sha256']!=row['image_sha256'] or src['truth']!=t:raise ValueError('Source mismatch')
                    before=indices[truth_key(t)] in ah;after=j in bh
                    state=('retained_hit' if after else 'loss') if before else ('gain' if after else 'retained_miss')
                    e=dict(comparison=family+'->IL1000',seed=seed,pair_id=row['pair_id'],variant=row['variant'],object_id=src['object_id'],truth=t,state=state,miss=next((m for m in row['misses'] if m['truth_index']==j),None),low_predictions=row['low_predictions'])
                    transitions.append(e)
                    if state=='loss':groups.setdefault(('LOSS',)+k,dict(kind='LOSS',source=src,truth=t,events=[]))['events'].append(e)
    events=[]
    for i,x in enumerate(groups.values(),1):
        x['event_id']=f'E{i:02}';events.append(renderer.render(x));deps.extend([Path(x['source']['image_path']),Path(x['page_path'])])
    prior.frozen(DEST/'evidence.json',dict(status='awaiting_explicit_review',events=events,transitions=transitions,inputs={str(d):prior.file_sha256(d) for d in deps}))
    for x in events:print(x['event_id'],x['kind'],x['source']['variant'],x['source'].get('object_id'),len(x['events']),x['page_path'])
if __name__=='__main__':build()
