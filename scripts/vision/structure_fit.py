"""Independent frozen, inference-only structure/fit diagnosis."""
import argparse
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw
from scripts.vision.exposure_order_retention import OUT as SOURCE, PRIOR, ROOT, read, frozen, file_sha256, verify_tree, identity
from scripts.vision.audit_structure_exposure import OUT as CENSUS, pixels, label_counts
from scripts.vision.exposure_protocol import verify, NAMES
from scripts.vision.run_exposure_diagnosis import checked_cell
from scripts.vision.exposure_metrics import missed_reason
from scripts.vision.evaluate_paired_visual_factors import match
from scripts.vision.analyze_recovery_paired_calibration import iou
from src.ml.artifacts import object_sha256

OUT=SOURCE/'structure-fit-diagnosis-v1'
STRUCTURES=('front_panel','side_back','top','base_fragment','thin_pole_crossbar','edge_pole','building_facade','wall_sky_ground','shadow','mixed_overlap')

def unique(rows,key):
    out={}
    for r in rows:
        k=key(r)
        if k in out:raise ValueError('Ambiguous source or member')
        out[k]=r
    return out

def truth_for(r):
    with Image.open(r['image_path']) as im:w,h=im.size
    truth=[]
    for n,line in enumerate(Path(r['label_path']).read_text().splitlines()):
        cls,x,y,bw,bh=map(float,line.split())
        truth.append(dict(class_name=NAMES[int(cls)],bbox_xyxy=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h],
            annotation_id=object_sha256([r['member_id'],r['label_sha256'],n]),label_line_index=n))
    return truth

def scoring(truth,formal,low):
    matches,up,ut=match(formal,truth)
    return dict(truth=truth,predictions=formal,low_predictions=low,matches=matches,
        unmatched_prediction_count=len(formal)-len(up),misses=[dict(truth_index=i,class_name=t['class_name'],reason=missed_reason(t,low),
        formal_matching_competition=any(p['class_name']==t['class_name'] and iou(p['bbox_xyxy'],t['bbox_xyxy'])>=.5 for p in formal)) for i,t in enumerate(truth) if i not in ut])

def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():verify_tree(dest);return read(dest)
    OUT.mkdir(exist_ok=True)
    paths=[CENSUS/'completion.json',SOURCE/'post-training-review-v1/completion.json']
    for path in paths:verify_tree(path)
    p=read(SOURCE/'protocol.json');prior=read(PRIOR/'protocol.json');census=read(CENSUS/'census.json')
    paths.extend([SOURCE/'protocol.json',PRIOR/'protocol.json',CENSUS/'census.json',Path(__file__),ROOT/'scripts/vision/exposure_metrics.py',ROOT/'scripts/vision/evaluate_exposure_diagnosis.py',ROOT/'scripts/vision/evaluate_paired_visual_factors.py',ROOT/'scripts/vision/run_structure_fit.py'])
    rows=p['pool_rows'];unique(rows,lambda r:r['member_id']);inputs={}
    for r in rows:
        for kind in ('image','label'):
            if file_sha256(r[kind+'_path'])!=r[kind+'_sha256']:raise ValueError('Stale member')
            inputs[r[kind+'_path']]=r[kind+'_sha256']
        if label_counts(r['label_path'],NAMES)!=r['class_instances']:raise ValueError('Full labels changed')
    models={'v2.11':dict(weights=p['initialization']['path'],weights_sha256=p['initialization']['sha256'],draws=[],exposure_role='initialization_not_new_training_fit')}
    for seed in (7,17,27):
        for arm,folder,prot in [('retained_reference',PRIOR,prior),('staged',SOURCE/'training',p),('interleaved',SOURCE/'training',p)]:
            key=f'{arm}-450-{seed}';cp=folder/key/'completion.json';cell=checked_cell(cp,prot);paths.extend([cp,Path(cell['exposure_path'])])
            ep=folder/f'evaluation-{key}.json';verify(read(ep));paths.append(ep)
            models[key]=dict(weights=cell['weights'],weights_sha256=cell['weights_sha256'],draws=read(cell['exposure_path'])['draws'],
                exposure_role='per_model_actual_draws',development_evaluation=str(ep))
    for model in models.values():
        if file_sha256(model['weights'])!=model['weights_sha256']:raise ValueError('Stale weights')
        inputs[model['weights']]=model['weights_sha256']
    from scripts.vision.evaluate_visual_augmentation_abcd import paired_truth
    paired_review=Path(p['evaluation']['paired_review']);negative_review=Path(p['evaluation']['negative_review'])
    paired,bind=paired_truth(read(paired_review)['frames']);inputs.update(bind);paths.extend([paired_review,negative_review])
    dev=[dict(**r,truth=t) for r,t in paired]+[dict(**r,truth=[]) for r in read(negative_review)['frames']]
    for r in dev:
        if file_sha256(r['image_path'])!=r['image_sha256'] or r['decision']!='accepted':raise ValueError('Stale development input')
        inputs[r['image_path']]=r['image_sha256']
    from scripts.vision.hard_negative_coverage import OUT as COVERAGE
    from scripts.vision.prepare_visual_bridge_negative_v2 import BASE as BRIDGE
    ap=COVERAGE/'final-admission.json';a=read(ap);paths.append(ap)
    decisions=unique(a['decisions'],lambda x:x['view_id']);sources={}
    for f in a['frames']:sources[f['view_id']]=dict(**f,source_decision=decisions[f['view_id']],source_kind='coverage')
    for stage in ('pilot-v1','remaining-v1'):
        path=BRIDGE/stage/'review-v1/semantic-review.json';paths.append(path)
        for f in read(path)['frames']:
            if f['view_id'] in sources:raise ValueError('Source collision')
            sources[f['view_id']]=dict(**f,source_kind='bridge')
    cm=unique(census['rows'],lambda r:r['member_id']);negative=[]
    for r in rows:
        if r['subset']!='hard_negative':continue
        f=sources[cm[r['member_id']]['source_view_id']]
        if file_sha256(f['image_path'])!=f['image_sha256'] or pixels(f['image_path'])!=pixels(r['image_path']):raise ValueError('RGB source mismatch')
        inputs[f['image_path']]=f['image_sha256'];negative.append(dict(member=r,source=f,census=cm[r['member_id']]))
    negative.sort(key=lambda r:(r['source']['source_kind'],r['source']['map_id'],r['source']['pair_id'],r['source']['lighting_id']))
    for i,r in enumerate(negative):r['event_id']=f'N{i+1:03}'
    reactors=[]
    for r in rows:
        for t in truth_for(r):
            if t['class_name']=='reactor':reactors.append(dict(event_id=f'T{len(reactors)+1:02}',member=r,truth=t))
    dr=[]
    for r in dev:
        if r['variant'] not in ('original','lighting'):continue
        for t in r['truth']:
            if t['class_name']=='reactor':dr.append(dict(event_id=f'D{len(dr)+1:02}',frame=r,truth=t,instance_identity=identity(t)))
    if (len(rows),len(negative),len(reactors),len(dev),len(models))!=(236,120,35,96,10):raise ValueError('Scope changed')
    inputs.update({str(x):file_sha256(x) for x in paths})
    result=frozen(dest,dict(status='frozen_review_and_inference_pending',rows=rows,negative=negative,reactors=reactors,dev_reactors=dr,development=dev,models=models,
        controls=dict(device='cpu',imgsz=640,formal_confidence=.37,diagnostic_confidence=.001,nms_iou=.7,agnostic_nms=False,max_det=300,matching_iou=.5),
        structures=list(STRUCTURES),inputs=inputs))
    print('FROZEN',len(dr),'development reactors',flush=True);return result

def evidence():
    p=read(OUT/'protocol.json');verify(p);dest=OUT/'evidence.json'
    if dest.exists():verify(read(dest));return
    events=[];pages=[]
    for group in ('negative','reactors','dev_reactors'):
        tiles=[]
        for event in p[group]:
            r=event.get('member',event.get('frame'));im=Image.open(r['image_path']).convert('RGB');original=im.copy()
            tile=Image.new('RGB',(1000,430),'white');d=ImageDraw.Draw(im)
            if group=='negative':
                source=event['source'];rois=source.get('source_decision',{}).get('rois',[])
                box=rois[0]['bbox_xyxy'] if rois else [0,0,im.width,im.height]
                subtitle=source['map_id']+' '+source['lighting_id']
            else:
                box=event['truth']['bbox_xyxy'];d.rectangle(box,outline='lime',width=4);subtitle='reactor; full label box'
            im.thumbnail((680,383));tile.paste(im,(0,40));crop=original.crop(box);crop.thumbnail((310,383));tile.paste(crop,(690,40))
            ImageDraw.Draw(tile).text((8,10),event['event_id']+' '+subtitle,fill='black')
            path=OUT/(event['event_id']+'.png');tile.save(path)
            events.append(dict(event_id=event['event_id'],group=group,image_path=r['image_path'],image_sha256=r['image_sha256'],
                evidence_path=str(path),evidence_sha256=file_sha256(path),source_event_identity=object_sha256(event)))
            tiles.append(tile)
        for i in range(0,len(tiles),4):
            page=Image.new('RGB',(2000,860),'white')
            for j,tile in enumerate(tiles[i:i+4]):page.paste(tile,((j%2)*1000,(j//2)*430))
            path=OUT/f'{group}-page-{i//4+1:02}.png';page.save(path);pages.append(path)
    paths=[OUT/'protocol.json']+pages+[Path(e['evidence_path']) for e in events]
    frozen(dest,dict(status='evidence_ready_not_reviewed',events=events,inputs={str(x):file_sha256(x) for x in paths}))
    print('EVIDENCE',len(events),flush=True)

def validate_review(evidence,decisions):
    es=unique(evidence['events'],lambda x:x['event_id']);ds=unique(decisions,lambda x:x['event_id'])
    if set(es)!=set(ds):raise ValueError('Missing/unexpected review')
    for eid,d in ds.items():
        e=es[eid]
        if d.get('review_nature')!='AI辅助审核' or not d.get('reason') or not d.get('reviewed_at'):raise ValueError('Nonexplicit review')
        for key in ('source_event_identity','image_sha256','evidence_sha256'):
            if d.get(key)!=e[key]:raise ValueError('Stale decision')
        for kind in ('image','evidence'):
            if file_sha256(e[kind+'_path'])!=e[kind+'_sha256']:raise ValueError('Stale evidence')
        if e['group']=='negative':
            if set(d.get('structures',{}))!=set(STRUCTURES):raise ValueError('Incomplete structure inventory')
            for s,v in d['structures'].items():
                if v['state'] not in ('present','not_seen','unknown'):raise ValueError('Invalid state')
                if v['state']=='not_seen' and not d.get('full_image_inspected'):raise ValueError('Absence not inspected')
                if v['state']=='present' and (not v.get('roi_xyxy') or not v.get('reason')):raise ValueError('Missing ROI evidence')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--evidence',action='store_true');args=ap.parse_args()
    if args.freeze:freeze()
    elif args.evidence:evidence()
    elif (OUT/'protocol.json').exists():verify(read(OUT/'protocol.json'));print('PREFLIGHT_ONLY_NO_TRAINING')
    else:print('NOT_FROZEN_NO_ACTION')

if __name__=='__main__':main()
