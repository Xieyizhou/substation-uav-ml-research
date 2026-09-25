"""Hash-bound H/O/N error evidence and seed diagnostics; no training changes."""
import sys
from pathlib import Path
from collections import Counter
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.run_negative_anchor import OUT as RUN,PREV,SOURCE,SEEDS,read,save,file_sha256,verify_tree
from scripts.vision.evaluate_hard_negative_coverage import BASE
from scripts.vision.analyze_recovery_paired_calibration import iou
from scripts.vision.finalize_hard_negative_coverage_training import losses
OUT=RUN/'error-diagnosis-v1'

def main():
    verify_tree(RUN/'report-receipt.json');verify_tree(PREV/'report-receipt.json')
    protocol=read(RUN/'protocol.json');lookup={r['member_id']:r for r in protocol['pool_rows']}
    sources={(r['view_id'],r['variant']):r for r in read(BASE/'hard-negative-isolated-v2/semantic-review.json')['frames']}
    records={(a,s):read(RUN/f'evaluation-H-100-{s}.json' if a=='H' else PREV/f'{a}-100-{s}.json') for a in ('O','N','H') for s in SEEDS}
    group={};transitions=[];fits={};exposure={};misses={}
    for s in SEEDS:
        for a in ('O','N','H'):
            cp=(RUN if a=='H' else SOURCE)/f'{a}-100-{s}/completion.json';cell=read(cp)
            curve=losses(Path(cell['exposure_path']).parent/'results.csv');fits[f'{a}-{s}']=curve
            misses[f'{a}-{s}']={v:dict(Counter(m['reason'] for r in records[a,s]['rows'] if r['variant']==v for m in r['misses'])) for v in ('original','material','background','lighting')}
        exposure[str(s)]=dict(Counter(lookup[m].get('coverage_unit','old' if lookup[m]['subset']=='hard_negative' else lookup[m]['subset']) for m in protocol['schedules'][f'H-100-{s}']))
        old={a:{(r['view_id'],r['variant']):r for r in records[a,s]['negative_rows']} for a in ('O','N')}
        for r in records['H',s]['negative_rows']:
            key=r['view_id'],r['variant']
            transitions.append(dict(seed=s,view_id=key[0],variant=key[1],H=bool(r['predictions']),**{a:bool(old[a][key]['predictions']) for a in old}))
            for j,p in enumerate(r['predictions']):
                overlaps={a:any(q['class_name']==p['class_name'] and iou(q['bbox_xyxy'],p['bbox_xyxy'])>=.5 for q in old[a][key]['predictions']) for a in old}
                peers=[z for z in SEEDS if any(q['class_name']==p['class_name'] and iou(q['bbox_xyxy'],p['bbox_xyxy'])>=.5 for r2 in records['H',z]['negative_rows'] if (r2['view_id'],r2['variant'])==key for q in r2['predictions'])]
                group.setdefault(key,[]).append(dict(prediction_id=f'{s}:{key[0]}:{key[1]}:{j}',seed=s,**p,correspondence=overlaps,consistent_seeds=peers))
    OUT.mkdir(parents=True,exist_ok=True);frames=[]
    for ordinal,(key,ps) in enumerate(sorted(group.items()),1):
        src=sources[key]
        if file_sha256(src['image_path'])!=src['image_sha256']:raise ValueError('Stale image')
        im=Image.open(src['image_path']).convert('RGB');thumb=im.copy();thumb.thumbnail((460,250))
        canvas=Image.new('RGB',(1000,260+240*len(ps)),'white');canvas.paste(thumb,(0,0));d=ImageDraw.Draw(canvas)
        d.text((470,10),f'Frame {ordinal} {key[1]}\n{key[0][:24]}',fill='black')
        for j,p in enumerate(ps):
            x1,y1,x2,y2=p['bbox_xyxy'];crop=im.crop((max(0,int(x1)),max(0,int(y1)),min(im.width,int(x2)+1),min(im.height,int(y2)+1)));crop.thumbnail((480,225));y=260+j*240;canvas.paste(crop,(0,y))
            d.text((490,y+10),f'Box {j+1} seed={p["seed"]} {p["class_name"]}\nconf={p["confidence"]:.6f}\nO/N={p["correspondence"]}\nH seeds={p["consistent_seeds"]}',fill='black')
            scale=thumb.width/im.width;d.rectangle((x1*scale,y1*scale,x2*scale,y2*scale),outline='red',width=2)
        path=OUT/f'frame-{ordinal:02}.jpg';canvas.save(path)
        frames.append(dict(ordinal=ordinal,view_id=key[0],variant=key[1],image_path=src['image_path'],image_sha256=src['image_sha256'],evidence_path=str(path),evidence_sha256=file_sha256(path),predictions=ps))
    save(OUT/'manifest.json',dict(status='pending_visual_review',frames=frames,frame_transitions=transitions,loss_curves=fits,exposure=exposure,misses=misses,
        inputs={str(p):file_sha256(p) for p in (Path(__file__),RUN/'report-receipt.json',PREV/'report-receipt.json')}))
    print('frames',len(frames),'boxes',sum(len(r['predictions']) for r in frames));print('exposure',exposure)

if __name__=='__main__':main()
