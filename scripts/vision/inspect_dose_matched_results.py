"""Freeze dose-control transitions, fixed gates and current visual evidence."""
from pathlib import Path
from collections import Counter
from PIL import Image,ImageDraw
from scripts.vision.dose_matched_contrast_control import OUT as TRAIN,SOURCE,KEYS,SOURCE_KEYS,GLOBAL,GLOBAL_KEYS,checked
from scripts.vision.evaluate_reactor_visibility_expansion import _load_inputs
from scripts.vision.evaluate_reviewed_negative_order import compare_truth
from scripts.vision import build_clear_context_error_evidence as fp
from scripts.vision.exposure_order_retention import PRIOR
from scripts.vision.finalize_exposure_diagnosis import aggregate,policy_checks
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

OUT=TRAIN/'audit-v1'

def run():
    dest=OUT/'evidence.json'
    if dest.exists():return checked(dest)
    OUT.mkdir(exist_ok=True);pairs,_,deps=_load_inputs();sources={r['image_sha256']:dict(r,truth=t) for r,t in pairs}
    groups={};transitions=[]
    for seed,key,prior,gkey in zip((7,17,27),KEYS,SOURCE_KEYS,GLOBAL_KEYS):
        cp=TRAIN/'evaluation-v1/units'/key/'completion.json';c=checked(cp);new=checked(c['result']);paths=[cp,Path(c['result'])];refs={}
        for name,root,oldkey in (('routed',SOURCE,prior),('reference',GLOBAL.parent/'reference',gkey)):
            op=root/'evaluation-v1/units'/oldkey/'completion.json';o=checked(op);refs[name]={(x['pair_id'],x['variant']):x for x in checked(o['result'])['rows']};paths += [op,Path(o['result'])]
        for row in new['rows']:
            states={name:compare_truth(by[row['pair_id'],row['variant']],row) for name,by in refs.items()}
            for name,ss in states.items():transitions.append(dict(seed=seed,reference=name,pair_id=row['pair_id'],variant=row['variant'],instances=ss))
            if row['variant'] not in ('original','lighting'):continue
            for idx,t in enumerate(row['truth']):
                losses=[name for name,ss in states.items() if ss[idx]['state']=='loss']
                if not losses:continue
                groups.setdefault(row['image_sha256'],[]).append(dict(seed=seed,truth_index=idx,truth=t,loss_against=losses,
                    diagnosis=next(m for m in row['misses'] if m['truth_index']==idx),predictions=row['predictions'],low_predictions=row['low_predictions']))
        for p in paths:deps[str(p.resolve())]=file_sha256(p)
    frames=[]
    for i,(digest,events) in enumerate(sorted(groups.items())):
        s=sources[digest]
        if file_sha256(s['image_path'])!=digest:raise ValueError('Stale RGB')
        im=Image.open(s['image_path']).convert('RGB');overlay=im.copy();draw=ImageDraw.Draw(overlay)
        indices=sorted({x['truth_index'] for x in events});page=Image.new('RGB',(1600,950+300*((len(indices)+3)//4)),'white');pd=ImageDraw.Draw(page)
        for j,idx in enumerate(indices):
            t=s['truth'][idx]
            if any(e['truth']!=t for e in events if e['truth_index']==idx):raise ValueError('Truth identity drift')
            b=t['bbox_xyxy'];draw.rectangle(b,outline='red',width=4);draw.text(tuple(b[:2]),str(idx),fill='white',stroke_width=1,stroke_fill='black')
            crop=im.crop(tuple(map(int,b)));crop.thumbnail((390,255));x,y=j%4*400,950+j//4*300
            page.paste(crop,(x,y+35));pd.text((x,y),f"GT {idx} {t['class_name']}",fill='black')
        overlay.thumbnail((1600,900));page.paste(overlay,(0,40));pd.text((10,10),f"loss-{i:02} {s['variant']}",fill='black');p=OUT/f'loss-{i:02}.png';page.save(p)
        frames.append(dict(frame_id=f'loss-{i:02}',variant=s['variant'],pair_id=s['pair_id'],image_path=s['image_path'],image_sha256=digest,truth=s['truth'],events=events,evidence_path=str(p.resolve()),evidence_sha256=file_sha256(p)))
        deps[str(p.resolve())]=file_sha256(p)
    sp=TRAIN/'evaluation-v1/summary.json';summary=checked(sp)
    if any(summary['matching_conflicts'].values()):raise ValueError('Matching conflict')
    pp=PRIOR/'protocol.json';policy=checked(pp);hp=Path(policy['evaluation']['historical_reference']);hist=checked(hp);rp=[PRIOR/f'evaluation-retained_reference-450-{s}.json' for s in (7,17,27)]
    gates=policy_checks(summary['group'],aggregate([checked(p) for p in rp]),hist['historical_A'],policy)
    for c in gates['checks']:
        if c.get('reference')=='same_budget_R':c['reference']='fixed_retained_reference_450_not_same_budget'
    fp.OUT=TRAIN;n=fp.run();np=TRAIN/'evaluation-v1/error-review-v1/evidence.json'
    for p in (sp,pp,hp,*rp,np,Path(__file__)):deps[str(p.resolve())]=file_sha256(p)
    return write_record(dest,dict(status='numerical_checked_visual_review_pending',frames=frames,all_transitions=transitions,gates=gates,
        events=sum(len(f['events']) for f in frames),reasons=dict(Counter(e['diagnosis']['reason'] for f in frames for e in f['events'])),negative_frames=len(n['frames']),
        training_admitted=False,promotable=False,inputs=deps))

if __name__=='__main__':
    r=run();print({k:r[k] for k in ('events','reasons','negative_frames')});print('positive_frames',len(r['frames']))
