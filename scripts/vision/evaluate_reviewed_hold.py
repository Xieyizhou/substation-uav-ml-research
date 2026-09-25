"""Fixed formal/low-confidence development evaluation, never trains."""
import argparse
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.train_reviewed_hold import OUT,KEYS,ready,complete,prior
from scripts.vision import evaluate_brightness_lr_retention as engine
from scripts.vision.finalize_exposure_diagnosis import aggregate,policy_checks
from scripts.vision.exposure_order_retention import PRIOR as RETAINED
from scripts.vision.analyze_recovery_paired_calibration import iou


def evaluate():
    p=ready();engine.OUT=OUT;engine.complete=complete
    for key in KEYS:
        r=engine.evaluate(key,p);print('EVALUATED',key,r['negative_summary'],flush=True)
    dest=OUT/'evaluation/summary.json'
    if dest.exists():prior.verify(prior.read(dest));return
    new=[prior.read(OUT/'evaluation'/f'{k}.json') for k in KEYS]
    refs=[RETAINED/f'evaluation-retained_reference-450-{s}.json' for s in (7,17,27)]
    hp=Path(p['evaluation']['historical_reference']);historical=prior.read(hp);prior.verify(historical)
    controls=[prior.read(p['baselines'][str(s)]['evaluation']) for s in (7,17,27)]
    reference=[prior.read(x) for x in refs]
    for r in new+controls+reference:prior.verify(r)
    checks=policy_checks(aggregate(new),aggregate(reference),historical['historical_A'],p)
    paths=[OUT/'evaluation'/f'{k}.json' for k in KEYS]+refs+[hp,Path(__file__),OUT/'protocol.json']+[Path(p['baselines'][str(s)]['evaluation']) for s in (7,17,27)]
    prior.frozen(dest,dict(status='numerical_complete_explicit_error_review_pending',aggregate=aggregate(new),direct_reference=aggregate(controls),policy_results=checks,
        matching_conflicts=sum(r['matching_conflicts'] for r in new),selected_candidate=None,
        inputs={str(x):prior.file_sha256(x) for x in paths}))


def evidence():
    p=ready();sp=OUT/'evaluation/summary.json';prior.verify(prior.read(sp));dest=OUT/'error-review/evidence.json'
    if dest.exists():prior.verify(prior.read(dest));return
    sources={};paths=[sp,Path(__file__)];fps={};losses={};transitions=[]
    for field in ('paired_review','negative_review'):
        rp=Path(p['evaluation'][field]);r=prior.read(rp);prior.verify(r);paths.append(rp)
        sources.update({(x['view_id'],x['variant']):x for x in r['frames']})
    for seed,key in zip((7,17,27),KEYS):
        np=OUT/'evaluation'/f'{key}.json';bp=Path(p['baselines'][str(seed)]['evaluation']);n,b=prior.read(np),prior.read(bp)
        for r in (n,b):prior.verify(r)
        paths += [np,bp];old={(r['view_id'],r['variant']):r for r in b['rows']}
        for r in n['negative_rows']:
            k=(r['view_id'],r['variant'])
            for j,pred in enumerate(r['predictions']):fps.setdefault(k,[]).append(dict(**pred,seed=seed,prediction_index=j))
        for r in n['rows']:
            k=(r['view_id'],r['variant']);a=old[k]
            if r['truth']!=a['truth'] or r['image_sha256']!=a['image_sha256']:raise ValueError('Paired identity mismatch')
            ah={x['truth_index'] for x in a['matches']};bh={x['truth_index'] for x in r['matches']}
            for j,t in enumerate(r['truth']):
                state='persistent_hit' if j in ah and j in bh else 'gain' if j in bh else 'loss' if j in ah else 'persistent_miss'
                transitions.append(dict(seed=seed,pair_id=r['pair_id'],variant=r['variant'],truth=t,state=state))
                if state=='loss' and r['variant'] in ('original','lighting'):
                    ek=(*k,t['annotation_id']);event=losses.setdefault(ek,dict(truth=t,seeds=[]))
                    event['seeds'].append(dict(seed=seed,miss=next(x for x in r['misses'] if x['truth_index']==j),low=sorted([dict(**x,iou=iou(t['bbox_xyxy'],x['bbox_xyxy'])) for x in r['low_predictions']],key=lambda x:x['iou'],reverse=True)[:3]))
    folder=dest.parent;folder.mkdir(exist_ok=True);events=[]
    for kind,data in (('FP',fps),('LOSS',losses)):
        for number,(key,value) in enumerate(sorted(data.items()),1):
            source=sources[key[:2]];ip=Path(source['image_path'])
            if prior.file_sha256(ip)!=source['image_sha256']:raise ValueError('Image hash changed')
            im=Image.open(ip).convert('RGB');full=im.copy();draw=ImageDraw.Draw(full);boxes=value if kind=='FP' else [value['truth']]
            eid=f'{kind}{number:02}';crops=[]
            for j,item in enumerate(boxes):
                b=item['bbox_xyxy'];draw.rectangle(b,outline='red',width=3);cp=folder/f'{eid}-{j}.png';im.crop(b).save(cp);paths.append(cp)
                crops.append(dict(path=str(cp),sha256=prior.file_sha256(cp)))
            full.thumbnail((960,540));canvas=Image.new('RGB',(1200,600+((len(crops)+2)//3)*250),'white');canvas.paste(full,(0,30))
            d=ImageDraw.Draw(canvas);d.text((0,5),eid+' '+source['variant'],fill='black')
            for j,crop in enumerate(crops):
                thumb=Image.open(crop['path']);thumb.thumbnail((390,210));x=(j%3)*400;y=600+(j//3)*250;canvas.paste(thumb,(x,y+25));d.text((x,y),str(j)+' '+str(boxes[j].get('class_name'))+' seed='+str(boxes[j].get('seed','multiple')),fill='black')
            page=folder/f'{eid}.png';canvas.save(page);paths += [page,ip]
            events.append(dict(event_id=eid,kind=kind,source=source,predictions=value if kind=='FP' else [],loss=value if kind=='LOSS' else None,crops=crops,page=str(page),page_sha256=prior.file_sha256(page)))
    prior.frozen(dest,dict(status='awaiting_explicit_review',events=events,all_transitions=transitions,inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('FP_IMAGES',len(fps),'LOSS_INSTANCES',len(losses),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--evaluate',action='store_true');a=ap.parse_args()
    if a.evaluate:evaluate();evidence()
    else:print('NO_INFERENCE_NO_TRAINING')
