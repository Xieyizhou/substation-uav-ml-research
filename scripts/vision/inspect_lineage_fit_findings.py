"""Bind current fit misses and full-label metrics for human/AI inspection."""
from pathlib import Path
from collections import Counter
from PIL import Image,ImageDraw
from scripts.vision.check_lineage_training_fit import OUT,ROOT,read,verify,frozen,file_sha256
from scripts.vision.analyze_recovery_paired_calibration import iou

def main():
    p=read(OUT/'protocol.json');s=read(OUT/'summary.json');verify(p);verify(s)
    rows={r['member_id']:r for r in p['rows']};targets={t['review']['event_id']:t for t in p['targets']};paths=[OUT/'protocol.json',OUT/'summary.json',Path(__file__)];events=[];metrics={}
    records={k:read(OUT/'inference'/f'{k}.json') for k in p['models']}
    for key,u in records.items():
        verify(u);paths.append(OUT/'inference'/f'{key}.json');sc=[r['scoring'] for r in u['rows'] if r['exposures']>0]
        totals=Counter(t['class_name'] for r in sc for t in r['truth']);hits=Counter(t['class_name'] for r in sc for t in r['matches'])
        metrics[key]={c:dict(instances=n,hits=hits[c],recall=hits[c]/n) for c,n in totals.items()}
    for d in s['details']:
        if d['hit']:continue
        r=rows[d['member_id']];t=targets[d['event_id']]['review']['truth'];u=next(r for r in records[d['model']]['rows'] if r['member_id']==d['member_id'])['scoring']
        low=sorted([dict(**x,iou=iou(x['bbox_xyxy'],t['bbox_xyxy'])) for x in u['low_predictions']],key=lambda x:x['iou'],reverse=True)[:5]
        im=Image.open(r['image_path']).convert('RGB');full=im.copy();draw=ImageDraw.Draw(full)
        for tr in u['truth']:draw.rectangle(tr['bbox_xyxy'],outline='yellow',width=2)
        draw.rectangle(t['bbox_xyxy'],outline='lime',width=5)
        crop=im.crop(t['bbox_xyxy']);crop.thumbnail((480,480));full.thumbnail((960,540))
        tile=Image.new('RGB',(1460,600),'white');tile.paste(full,(0,45));tile.paste(crop,(975,45));ImageDraw.Draw(tile).text((5,5),d['model']+' '+d['event_id']+' '+d['miss']['reason'],fill='black')
        path=OUT/(d['model']+'-'+d['event_id']+'.png');tile.save(path);paths.append(path)
        comparisons=[x for x in s['details'] if x['event_id']==d['event_id']]
        events.append(dict(detail=d,source=r,truth=t,top_low_predictions=low,evidence_path=str(path),evidence_sha256=file_sha256(path),all_models= comparisons))
    frozen(OUT/'findings.json',dict(status='awaiting_explicit_inspection',events=events,full_label_fit_metrics=metrics,inputs={str(x):file_sha256(x) for x in paths}))
    for e in events:print(e['evidence_path'])

if __name__=='__main__':main()
