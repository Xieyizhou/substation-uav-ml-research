"""Explicit endpoint fit diagnosis on all actually exposed training members."""
import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from PIL import Image
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision.evaluate_reviewed_negative_order import checked,TRAIN,ev
from scripts.vision.evaluate_paired_visual_factors import match
from scripts.vision.exposure_metrics import missed_reason

OUT=TRAIN/'training-fit-diagnosis-v1'
SEEDS=(7,17,27)

def truth(row,names):
    if file_sha256(row['label_path'])!=row['label_sha256']:raise ValueError('Changed label')
    with Image.open(row['image_path']) as im:w,h=im.size
    out=[]
    for index,line in enumerate(Path(row['label_path']).read_text().splitlines()):
        cls,x,y,bw,bh=map(float,line.split());out.append(dict(class_name=names[int(cls)],label_line_index=index,bbox_xyxy=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]))
    if dict(Counter(t['class_name'] for t in out))!=row['class_instances']:raise ValueError('Instance count mismatch')
    return out

def cell(seed):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True);dest=OUT/f'seed-{seed}.json'
    if dest.exists():return checked(dest)
    p=checked(TRAIN/'protocol.json');cp=TRAIN/'training'/f'reviewed-interleaved-480-{seed}'/'completion.json';c=checked(cp);ex=checked(c['exposure_path'])
    if ex['actual']!=p['schedules'][f'reviewed-interleaved-480-{seed}']:raise ValueError('Actual exposure mismatch')
    counts=Counter(ex['actual']);rows=[];deps={str(x.resolve()):file_sha256(x) for x in (TRAIN/'protocol.json',cp,Path(c['exposure_path']),Path(c['weights']),Path(__file__),Path(ev.__file__))}
    with ExitStack() as stack:
        for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(YOLO,'train'),(YOLO,'val')):
            stack.enter_context(patch.object(obj,name,side_effect=AssertionError('Training forbidden')))
        model=YOLO(c['weights'])
        for i,row in enumerate(p['pool_rows']):
            for kind in ('image','label'):
                if file_sha256(row[kind+'_path'])!=row[kind+'_sha256']:raise ValueError('Member changed')
                deps[row[kind+'_path']]=row[kind+'_sha256']
            gt=truth(row,p['names']);pred=ev.predict(model,row['image_path'],.37);low=ev.predict(model,row['image_path'],.001)
            matches,used_p,used_t=match(pred,gt)
            rows.append(dict(member_id=row['member_id'],subset=row['subset'],lineage_id=row['lineage_id'],variant=row['variant'],actual_exposures=counts[row['member_id']],
                image_path=row['image_path'],image_sha256=row['image_sha256'],label_sha256=row['label_sha256'],truth=gt,predictions=pred,low_predictions=low,matches=matches,
                misses=[dict(truth_index=j,reason=missed_reason(t,low),class_name=t['class_name']) for j,t in enumerate(gt) if j not in used_t],unmatched_predictions=len(pred)-len(used_p)))
            if (i+1)%50==0:print('FIT_PROGRESS',seed,i+1,'/380',flush=True)
    if len(rows)!=380 or any(r['actual_exposures']==0 for r in rows):raise ValueError('Incomplete actual-exposure diagnosis')
    return write_record(dest,dict(status='training_fit_diagnosis_complete_not_validation',seed=seed,rows=rows,training_admitted=False,promotable=False,inputs=deps))

def summarize():
    paths=[OUT/f'seed-{s}.json' for s in SEEDS];records=[checked(p) for p in paths];summaries=[]
    for r in records:
        classes={}
        for name in ev.NAMES:
            total=sum(t['class_name']==name for row in r['rows'] for t in row['truth']);hits=sum(m['class_name']==name for row in r['rows'] for m in row['matches'])
            classes[name]=dict(truth=total,matched=hits,recall=hits/total if total else None)
        negative=[row for row in r['rows'] if not row['truth']]
        summaries.append(dict(seed=r['seed'],classes=classes,negative_frames=len(negative),negative_error_frames=sum(bool(row['predictions']) for row in negative),negative_predictions=sum(len(row['predictions']) for row in negative)))
    path=OUT/'summary.json'
    if path.exists():return checked(path)
    return write_record(path,dict(status='fit_numerical_complete_visual_checks_pending',seeds=summaries,training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in paths+[Path(__file__)]}))

def run():
    OUT.mkdir(exist_ok=True)
    for group in ((7,17),(27,)):
        jobs=[]
        try:
            for seed in group:
                log=(OUT/f'seed-{seed}.log').open('a');proc=subprocess.Popen([sys.executable,'-u','-m','scripts.vision.diagnose_reviewed_training_fit','--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT);jobs.append((proc,log))
            for proc,_ in jobs:
                proc.wait(timeout=3600)
                if proc.returncode:raise RuntimeError('Fit worker failed; retain log, no automatic semantic retry')
        finally:
            for proc,log in jobs:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                log.close()
    print(summarize()['seeds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--infer',action='store_true');p.add_argument('--seed',type=int,choices=SEEDS);a=p.parse_args()
    if a.seed:cell(a.seed)
    elif a.infer:run()
    else:print('NO_INFERENCE_NO_TRAINING')
