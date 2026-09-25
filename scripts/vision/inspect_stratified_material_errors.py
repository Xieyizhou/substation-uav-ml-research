"""Paired material/original evidence for explicit review, not automatic approval."""
import sys
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.run_stratified_negative_control import OUT as RUN,REFERENCE,read,save,file_sha256,verify_tree,checked_rows
from scripts.vision.analyze_recovery_paired_calibration import iou
OUT=RUN/'material-error-review-v1'

def main():
    verify_tree(RUN/'report-receipt.json');current=read(RUN/'evaluation.json');prior=read(REFERENCE/'evaluation-CF-100-7.json')
    sources,_=checked_rows();src={(r['pair_id'],r['variant']):r for r in sources}
    orig={r['pair_id']:r for r in current['rows'] if r['variant']=='original'}
    prev={r['pair_id']:r for r in prior['rows'] if r['variant']=='material'};items=[]
    for row in current['rows']:
        if row['variant']!='material':continue
        used={m['prediction_index'] for m in row['matches']}
        for idx,p in enumerate(row['predictions']):
            if idx in used or p['class_name']!='capacitor_bank':continue
            overlaps=sorted([dict(truth_index=j,class_name=t['class_name'],iou=iou(p['bbox_xyxy'],t['bbox_xyxy'])) for j,t in enumerate(row['truth'])],key=lambda x:x['iou'],reverse=True)
            items.append(dict(item_id=f'P{len(items)+1:02}',kind='unmatched_capacitor_prediction',pair_id=row['pair_id'],bbox_xyxy=p['bbox_xyxy'],prediction=p,truth_overlaps=overlaps,
                original_predictions=[q for q in orig[row['pair_id']]['predictions'] if iou(q['bbox_xyxy'],p['bbox_xyxy'])>=.1]))
        if prev[row['pair_id']]['planned_assigned_hit'] and not row['planned_assigned_hit']:
            t=row['truth'][row['planned_truth_index']]
            items.append(dict(item_id='L'+str(sum(x['kind']=='lost_target' for x in items)+1),kind='lost_target',pair_id=row['pair_id'],bbox_xyxy=t['bbox_xyxy'],truth=t,
                diagnosis=next(m for m in row['misses'] if m['truth_index']==row['planned_truth_index']),original_hit=orig[row['pair_id']]['planned_assigned_hit'],
                low_predictions=[q for q in row['low_predictions'] if iou(q['bbox_xyxy'],t['bbox_xyxy'])>=.5]))
    if sum(x['kind']=='unmatched_capacitor_prediction' for x in items)!=20 or sum(x['kind']=='lost_target' for x in items)!=3:raise ValueError('Unexpected review scope')
    OUT.mkdir(parents=True,exist_ok=True);inputs={str(RUN/'report-receipt.json'):file_sha256(RUN/'report-receipt.json'),str(Path(__file__)):file_sha256(Path(__file__))}
    for offset in range(0,len(items),3):
        canvas=Image.new('RGB',(1200,900),'white');d=ImageDraw.Draw(canvas)
        for n,item in enumerate(items[offset:offset+3]):
            item['sources']={}
            for j,variant in enumerate(('material','original')):
                source=src[item['pair_id'],variant];assert file_sha256(source['image_path'])==source['image_sha256']
                inputs[source['image_path']]=source['image_sha256'];item['sources'][variant]=source
                im=Image.open(source['image_path']).convert('RGB');x1,y1,x2,y2=item['bbox_xyxy'];crop=im.crop((max(0,int(x1)),max(0,int(y1)),min(im.width,int(x2)+1),min(im.height,int(y2)+1)));crop.thumbnail((590,235));canvas.paste(crop,(j*600,n*300+60))
                detail=item['truth']['class_name']+' '+item['diagnosis']['reason'] if item['kind']=='lost_target' else f"conf={item['prediction']['confidence']:.3f} top_truth={item['truth_overlaps'][0]}"
                caption=detail if variant=='material' else 'Same pixel region; original predictions are recorded in manifest, not copied from material.'
                d.text((j*600+5,n*300+5),f"{item['item_id']} {variant}\n{caption}",fill='black')
        page=OUT/f'page-{offset//3+1:02}.jpg';canvas.save(page);inputs[str(page)]=file_sha256(page)
        for item in items[offset:offset+3]:item.update(evidence_path=str(page),evidence_sha256=file_sha256(page))
    save(OUT/'manifest.json',dict(status='pending_AI_assisted_review',items=items,inputs=inputs));print('ITEMS',[(i['item_id'],i['kind']) for i in items])

if __name__=='__main__':main()
