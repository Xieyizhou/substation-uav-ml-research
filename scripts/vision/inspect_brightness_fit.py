"""Evidence for every current miss in the pre-frozen clear training subset."""
from pathlib import Path
from PIL import Image,ImageDraw
from scripts.vision.prepare_brightness_transfer import OUT,read,verify,frozen,file_sha256
from scripts.vision.check_lineage_training_fit import OUT as FIT
from scripts.vision.analyze_recovery_paired_calibration import iou

def main():
    p=read(FIT/'protocol.json');verify(p);idx={r['member_id']:r for r in p['rows']};events=[];paths=[FIT/'protocol.json',Path(__file__)];dest=OUT/'fit-review';dest.mkdir(exist_ok=True)
    for seed in (7,17,27):
        path=OUT/'fit'/f'brightness-450-{seed}.json';f=read(path);verify(f);paths.append(path);sc={r['member_id']:r['scoring'] for r in f['rows']}
        for target in p['targets']:
            if not target['clear_primary']:continue
            d=target['review'];s=sc[d['member_id']];ti=d['truth']['label_line_index']
            if any(m['truth_index']==ti for m in s['matches']):continue
            row=idx[d['member_id']];truth=d['truth'];im=Image.open(row['image_path']).convert('RGB');full=im.copy();draw=ImageDraw.Draw(full)
            for t in s['truth']:draw.rectangle(t['bbox_xyxy'],outline='yellow',width=2)
            draw.rectangle(truth['bbox_xyxy'],outline='lime',width=5);full.thumbnail((960,540));crop=im.crop(truth['bbox_xyxy']);crop.thumbnail((470,480));tile=Image.new('RGB',(1460,600),'white');tile.paste(full,(0,45));tile.paste(crop,(980,45));ImageDraw.Draw(tile).text((5,5),f"seed {seed} {d['event_id']}",fill='black');ep=dest/f"{seed}-{d['event_id']}.png";tile.save(ep);paths.extend([ep,Path(row['image_path'])])
            events.append(dict(review_id=f"{seed}:{d['event_id']}",seed=seed,source=row,truth=truth,prior_review=d,evidence_path=str(ep),evidence_sha256=file_sha256(ep),miss=next(m for m in s['misses'] if m['truth_index']==ti),low=sorted([dict(**x,iou=iou(truth['bbox_xyxy'],x['bbox_xyxy'])) for x in s['low_predictions']],key=lambda x:x['iou'],reverse=True)[:3]))
    frozen(dest/'evidence.json',dict(status='awaiting_explicit_clear_miss_review',events=events,inputs={str(x):file_sha256(x) for x in paths}))
    print('CLEAR_MISSES',len(events))
if __name__=='__main__':main()
