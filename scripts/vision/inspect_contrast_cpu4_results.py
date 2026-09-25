"""Freeze complete transition evidence and fixed gates; never invent reviews."""
from pathlib import Path
from collections import Counter
from PIL import Image,ImageDraw
from scripts.vision.contrast_cpu4_control import ROOT,KEYS,checked
from scripts.vision.evaluate_reviewed_negative_order import compare_truth
from scripts.vision.evaluate_reactor_visibility_expansion import _load_inputs
from scripts.vision.exposure_order_retention import PRIOR
from scripts.vision.finalize_exposure_diagnosis import aggregate,policy_checks
from scripts.vision import build_clear_context_error_evidence as fp
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

OUT=ROOT/'contrast/audit-v1'

def run():
    OUT.mkdir(exist_ok=True);dest=OUT/'evidence.json'
    if dest.exists():return checked(dest)
    paired,_,deps=_load_inputs();sources={r['image_sha256']:dict(r,truth=t) for r,t in paired}
    groups={};transitions=[]
    for key in KEYS:
        records=[]
        for arm in ('reference','contrast'):
            cp=ROOT/arm/'evaluation-v1/units'/key/'completion.json';c=checked(cp);r=checked(c['result']);records.append(r)
            for p in (cp,Path(c['result']),ROOT/arm/'training'/key/'thread-verification.json',ROOT/arm/'training'/key/'tensor-verification.json'):
                checked(p);deps[str(p.resolve())]=file_sha256(p)
        old,new=records;by={(r['pair_id'],r['variant']):r for r in old['rows']}
        for row in new['rows']:
            states=compare_truth(by[row['pair_id'],row['variant']],row)
            transitions.append(dict(seed=int(key.split('-')[-1]),pair_id=row['pair_id'],variant=row['variant'],instances=states))
            if row['variant'] not in ('original','lighting'):continue
            for idx,state in enumerate(states):
                if state['state']!='loss':continue
                groups.setdefault(row['image_sha256'],[]).append(dict(seed=int(key.split('-')[-1]),truth_index=idx,truth=state['truth'],
                    diagnosis=next(m for m in row['misses'] if m['truth_index']==idx),predictions=row['predictions'],low_predictions=row['low_predictions']))
    frames=[]
    for i,(digest,events) in enumerate(sorted(groups.items())):
        s=sources[digest]
        if file_sha256(s['image_path'])!=digest:raise ValueError('Stale RGB')
        im=Image.open(s['image_path']).convert('RGB');overlay=im.copy();d=ImageDraw.Draw(overlay)
        indices=sorted({e['truth_index'] for e in events});page=Image.new('RGB',(1600,950+300*((len(indices)+3)//4)),'white');pd=ImageDraw.Draw(page)
        for j,idx in enumerate(indices):
            t=s['truth'][idx]
            if any(e['truth']!=t for e in events if e['truth_index']==idx):raise ValueError('Truth correspondence mismatch')
            b=t['bbox_xyxy'];d.rectangle(b,outline='red',width=4);d.text(tuple(b[:2]),str(idx),fill='white',stroke_width=1,stroke_fill='black')
            crop=im.crop(tuple(map(int,b)));crop.thumbnail((390,255));x,y=j%4*400,950+j//4*300
            page.paste(crop,(x,y+35));pd.text((x,y),f"GT {idx} {t['class_name']} seeds "+','.join(str(e['seed']) for e in events if e['truth_index']==idx),fill='black')
        overlay.thumbnail((1600,900));page.paste(overlay,(0,40));pd.text((10,10),f"loss-{i:02} {s['variant']}",fill='black')
        path=OUT/f'loss-{i:02}.png';page.save(path)
        frames.append(dict(frame_id=f'loss-{i:02}',variant=s['variant'],pair_id=s['pair_id'],image_path=s['image_path'],image_sha256=digest,
            truth=s['truth'],events=events,evidence_path=str(path.resolve()),evidence_sha256=file_sha256(path)))
        deps[str(path.resolve())]=file_sha256(path)
    policy_path=PRIOR/'protocol.json';policy=checked(policy_path);hp=Path(policy['evaluation']['historical_reference']);h=checked(hp)
    refs=[PRIOR/f'evaluation-retained_reference-450-{s}.json' for s in (7,17,27)]
    hist=[next(Path(p) for p in h['inputs'] if p.endswith(f'/historical-A-{s}.json')) for s in (7,17,27)]
    if aggregate([checked(p) for p in hist])!=h['historical_A']:raise ValueError('Historical identity mismatch')
    gates={}
    for arm in ('reference','contrast'):
        sp=ROOT/arm/'evaluation-v1/summary.json';summary=checked(sp)
        if any(summary['matching_conflicts'].values()):raise ValueError('Unresolved matching conflict')
        gates[arm]=policy_checks(summary['group'],aggregate([checked(p) for p in refs]),h['historical_A'],policy)
        for c in gates[arm]['checks']:
            if c.get('reference')=='same_budget_R':c['reference']='fixed_retained_reference_450_not_same_budget'
        deps[str(sp.resolve())]=file_sha256(sp)
    fp.OUT=ROOT/'contrast';negative=fp.run();fp_path=ROOT/'contrast/evaluation-v1/error-review-v1/evidence.json'
    for p in (policy_path,hp,*refs,*hist,fp_path,Path(__file__)):deps[str(p.resolve())]=file_sha256(p)
    return write_record(dest,dict(status='complete_numerical_transitions_visual_review_pending',frames=frames,all_transitions=transitions,gates=gates,
        loss_events=sum(len(f['events']) for f in frames),loss_truths=sum(len({e['truth_index'] for e in f['events']}) for f in frames),
        loss_reasons=dict(Counter(e['diagnosis']['reason'] for f in frames for e in f['events'])),negative_frames=len(negative['frames']),
        training_admitted=False,promotable=False,inputs=deps))

if __name__=='__main__':
    r=run();print({k:r[k] for k in ('loss_events','loss_truths','loss_reasons','negative_frames')})
    for arm,g in r['gates'].items():print(arm,[c for c in g['checks'] if not c['passed']])
