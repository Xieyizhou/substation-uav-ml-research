"""Read-only diagnosis of completed runs; derived evidence does not alter inputs."""
import sys
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.evaluate_hard_negative_coverage import EVAL,TRAIN,BASE,KEYS,SEEDS,read,save,file_sha256,verify_tree
from scripts.vision.analyze_recovery_paired_calibration import iou
OUT=EVAL/'negative-diagnosis-v1'

def main():
    verify_tree(EVAL/'report-receipt.json')
    protocol=read(TRAIN/'protocol.json');lookup={r['member_id']:r for r in protocol['pool_rows']}
    exposure={};members=[]
    for seed in SEEDS:
        counts={a:Counter(protocol['schedules'][f'{a}-100-{seed}']) for a in ('O','N')}
        exposure[str(seed)]={a:dict(Counter({'old_negative':sum(n for mid,n in counts[a].items() if lookup[mid]['subset']=='hard_negative' and not mid.startswith('coverage:')),
            'new_negative':sum(n for mid,n in counts[a].items() if mid.startswith('coverage:'))})) for a in ('O','N')}
        for mid,r in lookup.items():
            if r['subset']=='hard_negative':members.append(dict(seed=seed,member_id=mid,lineage_id=r['lineage_id'],coverage_unit=r.get('coverage_unit','old'),O=counts['O'][mid],N=counts['N'][mid]))
    sources={(r['view_id'],r['variant']):r for r in read(BASE/'hard-negative-isolated-v2/semantic-review.json')['frames']}
    records={k:read(EVAL/f'{k}.json') for k in KEYS}
    grouped={}
    transitions=[]
    for seed in SEEDS:
        old={(r['view_id'],r['variant']):r for r in records[f'O-100-{seed}']['negative_rows']}
        for row in records[f'N-100-{seed}']['negative_rows']:
            key=row['view_id'],row['variant'];o=old[key]
            transitions.append(dict(seed=seed,view_id=key[0],variant=key[1],O=bool(o['predictions']),N=bool(row['predictions'])))
            for index,p in enumerate(row['predictions']):
                peers=[s for s in SEEDS if any(q['class_name']==p['class_name'] and iou(q['bbox_xyxy'],p['bbox_xyxy'])>=.5 for r in records[f'N-100-{s}']['negative_rows'] if (r['view_id'],r['variant'])==key for q in r['predictions'])]
                old_match=any(q['class_name']==p['class_name'] and iou(q['bbox_xyxy'],p['bbox_xyxy'])>=.5 for q in o['predictions'])
                grouped.setdefault(key,[]).append(dict(prediction_id=f'{seed}:{key[0]}:{key[1]}:{index}',seed=seed,**p,
                    corresponding_O_box=old_match,N_consistent_seeds=peers))
    frames=[]
    OUT.mkdir(parents=True,exist_ok=True)
    for ordinal,(key,predictions) in enumerate(sorted(grouped.items()),1):
        source=sources[key];im=Image.open(source['image_path']).convert('RGB')
        canvas=Image.new('RGB',(1000,260+240*len(predictions)), 'white');draw=ImageDraw.Draw(canvas)
        thumb=im.copy();thumb.thumbnail((460,250));canvas.paste(thumb,(0,0))
        draw.text((470,10),f'Frame {ordinal} / {key[1]}\n{key[0][:20]}',fill='black')
        for j,p in enumerate(predictions):
            x1,y1,x2,y2=p['bbox_xyxy'];crop=im.crop((max(0,int(x1)),max(0,int(y1)),min(im.width,int(x2)+1),min(im.height,int(y2)+1)))
            crop.thumbnail((480,225));y=260+j*240;canvas.paste(crop,(0,y))
            draw.text((490,y+10),f'Box {j+1} seed={p["seed"]} {p["class_name"]}\nconf={p["confidence"]:.6f}\nO overlap={p["corresponding_O_box"]}\nN seeds={p["N_consistent_seeds"]}\nbbox={[round(x,1) for x in p["bbox_xyxy"]]}',fill='black')
            scale=thumb.width/im.width;draw.rectangle((x1*scale,y1*scale,x2*scale,y2*scale),outline='red',width=2)
        evidence=OUT/f'frame-{ordinal:02d}.jpg';canvas.save(evidence)
        frames.append(dict(ordinal=ordinal,view_id=key[0],variant=key[1],image_path=source['image_path'],image_sha256=source['image_sha256'],evidence_path=str(evidence),evidence_sha256=file_sha256(evidence),predictions=predictions))
    save(OUT/'manifest.json',dict(status='pending_visual_review',frames=frames,exposure=exposure,member_exposures=members,frame_transitions=transitions,
        inputs={str(p):file_sha256(p) for p in [Path(__file__),EVAL/'report-receipt.json',TRAIN/'protocol.json']}))
    print(exposure);print('frames',len(frames),'boxes',sum(len(f['predictions']) for f in frames))

if __name__=='__main__':main()
