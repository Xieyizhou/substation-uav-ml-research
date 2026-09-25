"""Source-bound bridge strata and evidence for all common clear-target misses."""
from pathlib import Path
from collections import Counter
from PIL import Image,ImageDraw
from scripts.vision.check_reviewed_hold_fit import OUT,prior,base
from scripts.vision.check_structure_fit_sources import equal_rgb
from scripts.vision.analyze_recovery_paired_calibration import iou

def main():
    p=prior.read(OUT/'protocol.json');s=prior.read(OUT/'summary.json')
    for r in (p,s):prior.verify(r)
    idx={r['member_id']:r for r in p['rows']};sources={};paths=[OUT/'protocol.json',OUT/'summary.json',Path(__file__).resolve()]
    root=prior.ROOT/'data/research/ml_training_recovery_v1/visual-bridge-supplement-v2'
    for stage in ('pilot-v1','remaining-positive-v1'):
        path=root/stage/'review-v1/semantic-review.json';r=prior.read(path);prior.verify(r);paths.append(path)
        for frame in r['frames']:
            mid='bridge:'+frame['view_id']
            if mid in sources:raise ValueError('Ambiguous bridge source')
            sources[mid]=frame
    bridge={}
    for mid,r in idx.items():
        if r['subset']!='bridge_positive':continue
        src=sources[mid]
        if prior.file_sha256(src['image_path'])!=src['image_sha256']:raise ValueError('Stale source RGB')
        equal_rgb(src['image_path'],r['image_path']);paths.append(Path(src['image_path']))
        bridge[mid]=dict(variant=src['variant'],source_image=src['image_path'],source_image_sha256=src['image_sha256'],derivation_group=r.get('derivation_group','unknown'))
    metrics={}
    for key in p['models']:
        metrics[key]={}
        for variant in sorted({r['variant'] for r in bridge.values()}):
            ds=[d for d in s['details'] if d['model']==key and d['common_exposed'] and d['member_id'] in bridge and bridge[d['member_id']]['variant']==variant]
            metrics[key][variant]={}
            for cls in ('all','reactor','capacitor_bank','switchgear','transformer'):
                es=[d for d in ds if cls=='all' or d['truth']['class_name']==cls]
                metrics[key][variant][cls]=dict(instances=len(es),hits=sum(e['hit'] for e in es),recall=sum(e['hit'] for e in es)/len(es) if es else None)
    events=[];folder=OUT/'visual-crosscheck';folder.mkdir(exist_ok=True)
    targets={(t['review']['member_id'],t['review']['truth']['annotation_id']):t for t in p['targets']}
    subjects=sorted({(d['member_id'],d['truth']['annotation_id']) for d in s['details'] if d['model'].startswith('reviewed-') and d['common_exposed'] and d['clear_primary'] and not d['hit']})
    for n,(mid,aid) in enumerate(subjects,1):
        target=targets[mid,aid];truth=target['review']['truth'];image=Image.open(idx[mid]['image_path']).convert('RGB');full=image.copy();draw=ImageDraw.Draw(full)
        for t in base.truth_for(idx[mid]):draw.rectangle(t['bbox_xyxy'],outline='yellow',width=2)
        draw.rectangle(truth['bbox_xyxy'],outline='red',width=5);full.thumbnail((960,540));crop=image.crop(truth['bbox_xyxy']);crop.thumbnail((470,460))
        page=Image.new('RGB',(1460,600),'white');page.paste(full,(0,35));page.paste(crop,(980,35));eid=f'F{n:02}'
        ImageDraw.Draw(page).text((5,5),eid+' '+target['review']['event_id']+' '+truth['class_name'],fill='black');path=folder/f'{eid}.png';page.save(path);paths.append(path)
        ds=[d for d in s['details'] if d['member_id']==mid and d['truth']['annotation_id']==aid]
        events.append(dict(event_id=eid,prior_review=target['review'],source=idx[mid],page=str(path),page_sha256=prior.file_sha256(path),results=ds))
    prior.frozen(OUT/'crosscheck.json',dict(status='explicit_visual_review_pending',bridge_sources=bridge,bridge_metrics=metrics,events=events,inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('CLEAR_TARGET_PAGES',len(events))

if __name__=='__main__':main()
