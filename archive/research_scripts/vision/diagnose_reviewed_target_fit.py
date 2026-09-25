"""Bounded target-structure fit check using existing six endpoints only."""
import argparse
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.train_order_fit_reviewed import DEST,KEYS,contract,bind,prior
from scripts.vision.diagnose_small_scale_order_fit import OUT as OLD
from scripts.vision.structure_fit import truth_for,scoring
from scripts.vision import diagnose_material_late_rehearsal_fit as runtime

OUT=DEST/'targeted-fit-v1'
def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    p,_,deps,_=contract();OUT.mkdir(exist_ok=True);models={};exposures={}
    for k in KEYS:
        c=bind().complete(k);xp=Path(c['exposure_path']);x=prior.read(xp);prior.verify(x)
        exposures[k]=dict(Counter(x['actual']));models[k]=dict(weights=c['weights'],weights_sha256=c['weights_sha256'])
        deps += [DEST/'training'/k/'completion.json',xp,Path(c['weights'])]
    rows={m['member_id']:dict(m,truth=truth_for(m)) for m in p['pool_rows']};selected={};targets=[];gaps=[]
    for cls in ('switchgear','reactor'):
        for subset in ('base','regular','bridge_positive'):
            candidates=[]
            for mid,m in rows.items():
                if m['subset']!=subset or not all(exposures[k].get(mid,0)>0 for k in KEYS):continue
                boxes=[t for t in m['truth'] if t['class_name']==cls]
                if not boxes:continue
                t=max(boxes,key=lambda t:(t['bbox_xyxy'][2]-t['bbox_xyxy'][0])*(t['bbox_xyxy'][3]-t['bbox_xyxy'][1]))
                w,h=Image.open(m['image_path']).size;b=t['bbox_xyxy'];short=min(b[2]-b[0],b[3]-b[1])*640/max(w,h)
                candidates.append((short,mid,t))
            candidates.sort(key=lambda x:(x[0],x[1]))
            if not candidates:gaps.append(dict(class_name=cls,subset=subset,reason='No common-exposed member'));continue
            for rank in sorted({len(candidates)//2,len(candidates)-1}):
                size,mid,t=candidates[rank];m=rows[mid];selected[mid]=m
                targets.append(dict(target_id=f'T{len(targets)+1:02}',member_id=mid,truth=t,short_side_640=size,
                    class_name=cls,subset=subset,selection_rank=rank,candidate_count=len(candidates)))
    for m in selected.values():
        for kind in ('image','label'):
            path=Path(m[kind+'_path'])
            if prior.file_sha256(path)!=m[kind+'_sha256']:raise ValueError('Member drift')
            deps.append(path)
    deps += [Path(__file__).resolve(),Path(runtime.__file__).resolve(),Path(__file__).with_name('structure_fit.py'),Path(__file__).with_name('evaluate_exposure_diagnosis.py'),DEST/'evaluation/endpoint-review-v1/review.json']
    return prior.frozen(dest,dict(status='bounded_target_fit_frozen',members=list(selected.values()),targets=targets,gaps=gaps,
        models=models,actual_exposures=exposures,environment={k:p['environment'][k] for k in ('torch','ultralytics')},
        inference=dict(device='cpu',imgsz=640,confidence=[.37,.001],iou=.7,max_det=300,match_iou=.5),
        selection='Before new predictions: per class/subset, largest class box per common-exposed image; select median and maximum short-side ranks. No performance-based replacement.',
        scope='Bounded training-fit probe, not whole-pool accuracy; structural analogy requires explicit visual review, not class identity alone.',
        inputs={str(d):prior.file_sha256(d) for d in deps}))

def evidence(p):
    dest=OUT/'evidence.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    members={m['member_id']:m for m in p['members']};pages=[]
    for t in p['targets']:
        m=members[t['member_id']];im=Image.open(m['image_path']).convert('RGB');crop=im.crop(t['truth']['bbox_xyxy']);d=ImageDraw.Draw(im)
        for full in m['truth']:d.rectangle(full['bbox_xyxy'],outline='gray',width=2)
        d.rectangle(t['truth']['bbox_xyxy'],outline='red',width=5);im.thumbnail((960,540));crop.thumbnail((470,540))
        card=Image.new('RGB',(1440,650),'white');card.paste(im,(0,40));card.paste(crop,(970,40))
        ImageDraw.Draw(card).text((5,5),f'{t["target_id"]} {m["member_id"]} {t["class_name"]} {m["variant"]} short640={t["short_side_640"]:.1f}',fill='black')
        path=OUT/(t['target_id']+'.png');card.save(path);pages.append(path)
    return prior.frozen(dest,dict(pages=list(map(str,pages)),inputs={str(x):prior.file_sha256(x) for x in [OUT/'protocol.json',*pages]}))

def finish(p):
    data={}
    for k in KEYS:
        r=prior.read(OUT/(k+'.json'));runtime.validate(r,k,p);data[k]={r['member_id']:r for r in r['rows']}
    results=[];deps=[OUT/'protocol.json']+[OUT/(k+'.json') for k in KEYS]
    for t in p['targets']:
        outcomes={}
        for k in KEYS:
            row=data[k][t['member_id']];j=row['truth'].index(t['truth']);hit=any(m['truth_index']==j for m in row['matches'])
            outcomes[k]=dict(hit=hit,exposures=p['actual_exposures'][k][t['member_id']],miss=next((m for m in row['misses'] if m['truth_index']==j),None))
        results.append(dict(t,outcomes=outcomes))
    dest=OUT/'summary.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='bounded_fit_inference_complete_visual_correspondence_pending',targets=results,
        inputs={str(x):prior.file_sha256(x) for x in deps}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');a=ap.parse_args();p=freeze();evidence(p)
    print('FROZEN',len(p['members']),len(p['targets']),p['gaps'],flush=True)
    if a.infer:
        runtime.OUT=OUT
        for k in KEYS:runtime.infer(k,p)
        print(finish(p)['status'],flush=True)
