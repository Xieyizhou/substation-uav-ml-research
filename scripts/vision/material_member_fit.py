"""Four-lineage diagnostic. Default checks only; no training entry exists."""
import argparse
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.material_control_feasibility import OUT as PRIOR,RUN,prior
from scripts.vision.structure_fit import truth_for,scoring
OUT=PRIOR.parent/'material-member-fit-v1'

def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():prior.verify(prior.read(dest));return
    cp=PRIOR/'continuation-completion.json';xp=PRIOR/'coverage-census.json';qp=PRIOR/'risk-containment.json'
    c,x,q=map(prior.read,(cp,xp,qp));p=prior.read(RUN/'protocol.json')
    for v in (c,x,q,p):prior.verify(v)
    selected=set(c['next_stage']['first_members']);lineages={r['lineage_id'] for r in p['pool_rows'] if r['member_id'] in selected}
    rows=[r for r in p['pool_rows'] if r['lineage_id'] in lineages]
    if len(rows)!=12 or set(q['denied_members'])&{r['member_id'] for r in rows}:raise ValueError('Unexpected selection or held group')
    paths=[cp,xp,qp,RUN/'protocol.json',Path(__file__).resolve()];models={}
    models['v2.11']=dict(weights=p['initialization']['path'],weights_sha256=p['initialization']['sha256'],counts={},role='initialization_comparator_not_training_fit')
    for key in c['next_stage']['weights']:
        tp=RUN/'training'/key/'completion.json';t=prior.read(tp);prior.verify(t);ap=Path(t['exposure_path']);a=prior.read(ap);prior.verify(a)
        if a['draws']!=p['schedules'][key]:raise ValueError('Actual exposure changed')
        models[key]=dict(weights=t['weights'],weights_sha256=t['weights_sha256'],counts=dict(Counter(a['draws'])),role='actual_exposure_training_fit_only')
        paths += [tp,ap]
    for model in models.values():
        if prior.file_sha256(model['weights'])!=model['weights_sha256']:raise ValueError('Changed weight')
        paths.append(Path(model['weights']))
    OUT.mkdir(exist_ok=True);(OUT/'evidence').mkdir(exist_ok=True);events=[];members=[]
    for n,m in enumerate(rows,1):
        truth=truth_for(m);source=next(r for r in x['members'] if r['member_id']==m['member_id'])
        if len(truth)!=len(source['instances']):raise ValueError('Full source identity missing')
        im=Image.open(m['image_path']).convert('RGB');canvas=Image.new('RGB',(1000,590+len(truth)*200),'white');d=ImageDraw.Draw(canvas)
        thumb=im.copy();thumb.thumbnail((1000,560));canvas.paste(thumb,(0,25));d.text((0,5),f'M{n:02} {source["source_variant"]}',fill='black')
        for i,t in enumerate(truth):
            matches=[s for s in source['instances'] if s['class_name']==t['class_name'] and max(abs(a-b) for a,b in zip(s['bbox_xyxy'],t['bbox_xyxy']))<1e-4]
            if len(matches)!=1:raise ValueError('Nonunique source target')
            s=matches[0];eid=f'M{n:02}-{i:02}';crop=OUT/'evidence'/f'{eid}.png';im.crop(t['bbox_xyxy']).save(crop)
            ct=Image.open(crop);ct.thumbnail((980,170));canvas.paste(ct,(0,615+i*200));d.text((0,590+i*200),f'{eid} {s["object_id"]} {t["class_name"]}',fill='black')
            events.append(dict(event_id=eid,member_id=m['member_id'],truth=t,object_id=s['object_id'],crop_path=str(crop),crop_sha256=prior.file_sha256(crop),image_path=m['image_path'],image_sha256=m['image_sha256']))
            paths.append(crop)
        page=OUT/'evidence'/f'M{n:02}.png';canvas.save(page);paths.append(page)
        members.append(dict(m,source_variant=source['source_variant'],truth=truth,page=str(page),page_sha256=prior.file_sha256(page),primary_gray=m['member_id'] in selected))
        paths += [Path(m['image_path']),Path(m['label_path'])]
    for name in ('structure_fit.py','exposure_metrics.py','evaluate_exposure_diagnosis.py','evaluate_paired_visual_factors.py'):
        paths.append(prior.ROOT/'scripts/vision'/name)
    prior.frozen(dest,dict(status='frozen_review_required',members=members,events=events,models=models,
        inference_protocol=c['next_stage']['protocol'],independent_pose_groups=4,training_allowed=False,
        inputs={str(z):prior.file_sha256(z) for z in paths}))
    print('FROZEN',len(members),'members',len(events),'labels; review required')

def preflight():
    p=prior.read(OUT/'protocol.json');prior.verify(p)
    for m in p['members']:
        if truth_for(m)!=m['truth']:raise ValueError('Changed labels')
    return p

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');a=ap.parse_args()
    if a.freeze:freeze()
    else:preflight();print('PREFLIGHT_ONLY_NO_INFERENCE_NO_TRAINING')
