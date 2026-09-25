"""Full-label fitting on exposed switchgear frames and all negative members."""
import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from scripts.vision import routed_late_decay_control as arm
from scripts.vision.diagnose_reviewed_training_fit import truth
from scripts.vision.evaluate_reactor_visibility_expansion import predict
from scripts.vision.evaluate_paired_visual_factors import match
from scripts.vision.exposure_metrics import missed_reason
from scripts.vision.locked_cpu_threads import locked_threads
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

OUT=arm.OUT.parent/'routed-retention-fit-diagnosis-v1'
MODELS={**{k:arm.SOURCE for k in arm.SOURCE_KEYS},**{k:arm.OUT for k in arm.KEYS}}
KEYS=tuple(MODELS)

def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():return arm.checked(dest)
    paths=[Path(__file__),arm.OUT/'audit-v1/completion.json']
    arm.checked(paths[-1]);source=arm.checked(arm.SOURCE/'protocol.json');members=[]
    for r in source['pool_rows']:
        gt=truth(r,source['names'])
        if not gt or any(t['class_name']=='switchgear' for t in gt):
            members.append(dict(r,truth=gt))
            paths.extend(Path(r[k+'_path']) for k in ('image','label'))
    for key,root in MODELS.items():
        p=arm.checked(root/'protocol.json');c=arm.checked(root/'training'/key/'completion.json');e=arm.checked(c['exposure_path']);counts=Counter(e['actual'])
        if p['pool_rows']!=source['pool_rows'] or e['actual']!=p['schedules'][key] or any(counts[r['member_id']]==0 for r in members):raise ValueError('Not common exposed members')
        paths.extend([root/'protocol.json',root/'training'/key/'completion.json',Path(c['exposure_path']),Path(c['weights'])])
    OUT.mkdir(exist_ok=True)
    return write_record(dest,dict(status='frozen_fit_only',members=members,models=list(KEYS),selection='all full-label switchgear-containing frames plus all empty-label frames; common actual exposure required',confidence=[.37,.001],training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in paths}))

def worker(key):
    import torch
    from ultralytics import YOLO
    p=freeze();dest=OUT/f'{key}.json'
    if dest.exists():return arm.checked(dest)
    root=MODELS[key];c=arm.checked(root/'training'/key/'completion.json');counts=Counter(arm.checked(c['exposure_path'])['actual']);rows=[]
    with ExitStack() as stack:
        for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(YOLO,'train'),(YOLO,'val')):
            stack.enter_context(patch.object(obj,name,side_effect=AssertionError('Training forbidden')))
        events=stack.enter_context(locked_threads(4));model=YOLO(c['weights'])
        for i,r in enumerate(p['members']):
            gt=r['truth'];pred=predict(model,r['image_path'],.37);low=predict(model,r['image_path'],.001);matches,used_p,used_t=match(pred,gt)
            rows.append(dict(member_id=r['member_id'],image_path=r['image_path'],image_sha256=r['image_sha256'],label_sha256=r['label_sha256'],lineage_id=r['lineage_id'],variant=r['variant'],actual_exposures=counts[r['member_id']],truth=gt,predictions=pred,low_predictions=low,matches=matches,misses=[dict(truth_index=j,class_name=t['class_name'],reason=missed_reason(t,low)) for j,t in enumerate(gt) if j not in used_t],unmatched_predictions=len(pred)-len(used_p)))
            if (i+1)%50==0:print(key,i+1,len(p['members']),flush=True)
    return write_record(dest,dict(status='common_exposed_training_fit_not_generalization',key=key,rows=rows,thread_events=events,training_admitted=False,promotable=False,inputs={str((OUT/'protocol.json').resolve()):file_sha256(OUT/'protocol.json')}))

def summarize():
    results={}
    for key in KEYS:
        r=arm.checked(OUT/f'{key}.json');counts=Counter(t['class_name'] for x in r['rows'] for t in x['truth']);hits=Counter(x['truth'][m['truth_index']]['class_name'] for x in r['rows'] for m in x['matches']);neg=[x for x in r['rows'] if not x['truth']]
        results[key]=dict(classes={c:dict(truth=n,hits=hits[c]) for c,n in counts.items()},negative_frames=len(neg),false_positive_frames=sum(bool(x['predictions']) for x in neg),misses=dict(Counter(m['reason'] for x in r['rows'] for m in x['misses'])))
    return write_record(OUT/'summary.json',dict(status='fit_numerical_complete',results=results,training_admitted=False,promotable=False,inputs={str((OUT/f'{k}.json').resolve()):file_sha256(OUT/f'{k}.json') for k in KEYS}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');ap.add_argument('--worker',choices=KEYS);a=ap.parse_args()
    if a.worker:worker(a.worker)
    elif a.infer:
        freeze();arm.runtime.KEYS=KEYS;arm.runtime.MODULE='scripts.vision.diagnose_routed_retention_fit';arm.runtime.parallel('--worker',OUT/'logs');print(summarize()['results'])
    else:print(freeze()['status'])
