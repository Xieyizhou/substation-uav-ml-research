"""Infer on every actually exposed material variant; never train or select rows."""
import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision.clear_context_lr_control import OUT as TRAIN,KEYS,checked
from scripts.vision.diagnose_reviewed_training_fit import truth
from scripts.vision.evaluate_reactor_visibility_expansion import predict
from scripts.vision.evaluate_paired_visual_factors import match
from scripts.vision.exposure_metrics import missed_reason

OUT=TRAIN/'material-fit-v1'
VARIANTS=('neutral','cool','warm','gray_all_body','gray_target_body','gray035')


def worker(key):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True);dest=OUT/f'{key}.json'
    if dest.exists():return checked(dest)
    protocol_path=TRAIN/'protocol.json';p=checked(protocol_path)
    cp=TRAIN/'training'/key/'completion.json';c=checked(cp);xp=Path(c['exposure_path']);ex=checked(xp)
    if ex['actual']!=p['schedules'][key]:raise ValueError('Actual schedule differs')
    counts=Counter(ex['actual']);members=[r for r in p['pool_rows'] if r['variant'] in VARIANTS]
    if len(members)!=144 or any(counts[r['member_id']]==0 for r in members):raise ValueError('Incomplete material population')
    deps={str(x.resolve()):file_sha256(x) for x in (protocol_path,cp,xp,Path(c['weights']),Path(__file__))};rows=[]
    with ExitStack() as stack:
        for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(YOLO,'train'),(YOLO,'val')):
            stack.enter_context(patch.object(obj,name,side_effect=AssertionError('Training forbidden')))
        model=YOLO(c['weights'])
        for i,r in enumerate(members):
            for kind in ('image','label'):
                if file_sha256(r[kind+'_path'])!=r[kind+'_sha256']:raise ValueError('Stale member')
                deps[r[kind+'_path']]=r[kind+'_sha256']
            gt=truth(r,p['names']);pred=predict(model,r['image_path'],.37);low=predict(model,r['image_path'],.001);matched,used_p,used_t=match(pred,gt)
            rows.append(dict(member_id=r['member_id'],variant=r['variant'],lineage_id=r['lineage_id'],image_path=r['image_path'],image_sha256=r['image_sha256'],
                label_sha256=r['label_sha256'],actual_exposures=counts[r['member_id']],truth=gt,predictions=pred,low_predictions=low,matches=matched,
                misses=[dict(truth_index=j,class_name=t['class_name'],reason=missed_reason(t,low)) for j,t in enumerate(gt) if j not in used_t],unmatched_predictions=len(pred)-len(used_p)))
            if (i+1)%36==0:print('MATERIAL_FIT',key,i+1,'/144',flush=True)
    return write_record(dest,dict(status='actual_exposed_material_training_fit_not_validation',key=key,rows=rows,training_admitted=False,promotable=False,inputs=deps))


def summarize():
    records=[checked(OUT/f'{k}.json') for k in KEYS];groups={}
    for record in records:
        groups[record['key']]={}
        for variant in ('all',)+VARIANTS:
            rows=[r for r in record['rows'] if variant=='all' or r['variant']==variant]
            counts=Counter(t['class_name'] for r in rows for t in r['truth']);hits=Counter(r['truth'][m['truth_index']]['class_name'] for r in rows for m in r['matches'])
            groups[record['key']][variant]=dict(images=len(rows),lineages=len({r['lineage_id'] for r in rows}),
                classes={c:dict(truth=n,hits=hits[c],recall=hits[c]/n) for c,n in counts.items()},
                reasons=dict(Counter(m['reason'] for r in rows for m in r['misses'])))
    path=OUT/'summary.json'
    if path.exists():return checked(path)
    return write_record(path,dict(status='material_fit_numerical_complete_visual_checks_pending',groups=groups,
        limits='Actual exposure required. Lineage IDs do not establish independent layouts/assets. Full labels, not only planned targets. No selection by score.',
        training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in [OUT/f'{k}.json' for k in KEYS]+[Path(__file__)]}))


def run():
    from scripts.vision import clear_context_training as runtime
    runtime.KEYS=KEYS;runtime.MODULE='scripts.vision.diagnose_lr_material_fit'
    runtime.parallel('--worker',OUT/'logs');print(summarize()['groups'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');ap.add_argument('--worker',choices=KEYS);a=ap.parse_args()
    if a.worker:worker(a.worker)
    elif a.infer:run()
    else:print('NO_INFERENCE_NO_TRAINING; explicit --infer required')
