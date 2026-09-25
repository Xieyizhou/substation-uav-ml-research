"""Read-only model checks and review evidence; no training or auto-approval."""
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision.clear_context_lr_control import OUT,PREVIOUS,KEYS,checked
from scripts.vision.evaluate_reactor_visibility_expansion import _load_inputs
from scripts.vision.exposure_order_retention import PRIOR
from scripts.vision.finalize_exposure_diagnosis import aggregate,policy_checks
from scripts.vision import build_clear_context_error_evidence as fp_pages


def run():
    root=OUT/'verification-v1';root.mkdir(exist_ok=True);target=root/'evidence.json'
    if target.exists():return checked(target)
    summary_path=OUT/'evaluation-v1/summary.json';summary=checked(summary_path)
    paired,_,deps=_load_inputs();sources={(r['pair_id'],r['variant']):(r,t) for r,t in paired}
    records=[]
    for key in KEYS:
        cp=OUT/'evaluation-v1/units'/key/'completion.json';c=checked(cp);r=checked(c['result']);records.append(r)
        for p in (cp,Path(c['result']),OUT/'training'/key/'lr-verification.json',OUT/'training'/key/'tensor-verification.json'):
            checked(p);deps[str(p.resolve())]=file_sha256(p)
    policy_path=PRIOR/'protocol.json';policy=checked(policy_path);hp=Path(policy['evaluation']['historical_reference']);h=checked(hp)
    refs_paths=[PRIOR/f'evaluation-retained_reference-450-{s}.json' for s in (7,17,27)]
    hist_paths=[next(Path(x) for x in h['inputs'] if x.endswith(f'/historical-A-{s}.json')) for s in (7,17,27)]
    if aggregate([checked(p) for p in hist_paths])!=h['historical_A']:raise ValueError('Historical aggregate mismatch')
    gates=policy_checks(summary['group'],aggregate([checked(p) for p in refs_paths]),h['historical_A'],policy)
    for c in gates['checks']:
        if c.get('reference')=='same_budget_R':c['reference']='fixed_retained_reference_450_not_same_budget'
    frames=[];counts={};all_seeds=[]
    for key,r in zip(KEYS,records):
        counts[key]=dict(Counter(m['reason'] for row in r['rows'] if row['variant']=='material' for m in row['misses']))
    for i,row in enumerate(r for r in records[0]['rows'] if r['variant']=='material'):
        source,truth=sources[row['pair_id'],'material'];original,_=sources[row['pair_id'],'original']
        if truth!=row['truth']:raise ValueError('Full truth differs')
        seed_rows=[next(x for x in r['rows'] if x['pair_id']==row['pair_id'] and x['variant']=='material') for r in records]
        if any(x['truth']!=truth or x['image_sha256']!=row['image_sha256'] for x in seed_rows):raise ValueError('Seed source mismatch')
        im=Image.open(source['image_path']).convert('RGB');orig=Image.open(original['image_path']).convert('RGB')
        page=Image.new('RGB',(1600,500+270*((len(truth)+3)//4)),'white');pd=ImageDraw.Draw(page)
        for j,img in enumerate((orig,im)):
            overlay=img.copy();d=ImageDraw.Draw(overlay)
            for idx,t in enumerate(truth):
                b=t['bbox_xyxy'];d.rectangle(b,outline='red',width=3);d.text((b[0],b[1]),str(idx),fill='white',stroke_width=1,stroke_fill='black')
            overlay.thumbnail((800,450));page.paste(overlay,(j*800,30))
        pd.text((10,5),f'material-{i:02} ORIGINAL | MATERIAL; truth-local indices, not plan device IDs',fill='black')
        for idx,t in enumerate(truth):
            crop=im.crop(tuple(map(int,t['bbox_xyxy'])));crop.thumbnail((390,215));x,y=idx%4*400,500+idx//4*270
            hits=[idx in {m['truth_index'] for m in r['matches']} for r in seed_rows]
            pd.text((x,y),f"GT {idx} {t['class_name']} hits 7/17/27 {hits}",fill='black');page.paste(crop,(x,y+35))
            if not any(hits):all_seeds.append(dict(pair_id=row['pair_id'],truth_index=idx,truth=t))
        path=root/f'material-{i:02}.png'
        if path.exists():raise ValueError('Unreceipted page exists')
        page.save(path);frames.append(dict(frame_id=f'material-{i:02}',pair_id=row['pair_id'],image_path=source['image_path'],
            image_sha256=row['image_sha256'],original_image_path=original['image_path'],original_image_sha256=original['image_sha256'],
            truth=truth,seed_rows=seed_rows,evidence_path=str(path.resolve()),evidence_sha256=file_sha256(path)))
    fp_pages.OUT=OUT;fp=fp_pages.run()
    for p in [summary_path,policy_path,hp,*refs_paths,*hist_paths,Path(__file__),OUT/'evaluation-v1/error-review-v1/evidence.json']:
        deps[str(p.resolve())]=file_sha256(p)
    return write_record(target,dict(status='numerical_checked_visual_review_pending',frames=frames,gates=gates,
        material_miss_reasons=counts,all_seed_material_misses=all_seeds,fp_images=len(fp['frames']),
        matching_conflicts=summary['matching_conflicts'],training_admitted=False,promotable=False,inputs=deps))


if __name__=='__main__':print(run()['material_miss_reasons'])
