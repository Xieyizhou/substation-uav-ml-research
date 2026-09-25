"""Build hash-bound crops for explicit review, never automatic approval."""
import sys
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.run_fixed_budget_diagnosis import OUT as RUN, REFERENCE, BASE, KEY, read,save,file_sha256,verify_tree,checked_rows
from scripts.vision.analyze_recovery_paired_calibration import iou
OUT=RUN/'error-coverage-v1'

def main():
    verify_tree(RUN/'report-receipt.json')
    current=read(RUN/'evaluation.json');old=read(REFERENCE/'evaluation-CF-100-7.json')
    sources={(r['view_id'],r['variant']):r for r in read(BASE/'hard-negative-isolated-v2/semantic-review.json')['frames']}
    positives,_=checked_rows();sources.update({(r['view_id'],r['variant']):r for r in positives})
    items=[]
    previous={(r['view_id'],r['variant']):r for r in old['negative_rows']}
    for r in current['negative_rows']:
        key=r['view_id'],r['variant']
        for j,p in enumerate(r['predictions']):
            items.append(dict(kind='negative',item_id=f'N{len(items)+1:02}',source=sources[key],prediction=p,
                bbox_xyxy=p['bbox_xyxy'],present_at_100=any(q['class_name']==p['class_name'] and iou(q['bbox_xyxy'],p['bbox_xyxy'])>=.5 for q in previous[key]['predictions'])))
    previous={(r['view_id'],r['variant']):r for r in old['rows']}
    for r in current['rows']:
        if r['variant']!='lighting':continue
        oldrow=previous[r['view_id'],r['variant']]
        before={m['truth_index'] for m in oldrow['matches']};after={m['truth_index'] for m in r['matches']}
        for idx in sorted(before-after):
            t=r['truth'][idx];miss=next(m for m in r['misses'] if m['truth_index']==idx)
            items.append(dict(kind='lighting_lost',item_id=f'L{sum(x['kind']=='lighting_lost' for x in items)+1:02}',source=sources[r['view_id'],r['variant']],bbox_xyxy=t['bbox_xyxy'],truth=t,diagnosis=miss,truth_index=idx))
    OUT.mkdir(parents=True,exist_ok=True);pages=[]
    for offset in range(0,len(items),6):
        canvas=Image.new('RGB',(1200,900),'white');d=ImageDraw.Draw(canvas)
        for n,item in enumerate(items[offset:offset+6]):
            src=item['source'];assert file_sha256(src['image_path'])==src['image_sha256']
            im=Image.open(src['image_path']).convert('RGB');x1,y1,x2,y2=item['bbox_xyxy']
            crop=im.crop((max(0,int(x1)),max(0,int(y1)),min(im.width,int(x2)+1),min(im.height,int(y2)+1)));crop.thumbnail((570,225))
            x=(n%2)*600;y=(n//2)*300;canvas.paste(crop,(x,y+65))
            detail=item.get('prediction',item.get('truth'));d.text((x+5,y+5),f"{item['item_id']} {detail['class_name']} {src['variant']}\n"+ (f"conf={detail['confidence']:.4f} previous={item['present_at_100']}" if item['kind']=='negative' else str(item['diagnosis'])),fill='black')
            item['crop_pixels']=[x2-x1,y2-y1];item['model_640_pixels']=[(x2-x1)*640/im.width,(y2-y1)*640/im.width]
        path=OUT/f'page-{offset//6+1:02}.jpg';canvas.save(path);pages.append(str(path))
        for item in items[offset:offset+6]:item.update(evidence_path=str(path),evidence_sha256=file_sha256(path))
    p=read(RUN/'protocol.json');members={r['member_id']:r for r in p['pool_rows']}
    counts=Counter(members[mid].get('coverage_unit','old_negative' if members[mid]['subset']=='hard_negative' else members[mid]['subset']) for mid in p['schedules'][KEY])
    save(OUT/'manifest.json',dict(status='pending_AI_assisted_review',items=items,exposure_by_coverage_unit=dict(counts),pages=pages,
        inputs={str(q):file_sha256(q) for q in [RUN/'report-receipt.json',Path(__file__)]+[Path(x) for x in pages]}))
    print('ITEMS',len(items),'PAGES',pages,'EXPOSURE',counts)

if __name__=='__main__':main()
