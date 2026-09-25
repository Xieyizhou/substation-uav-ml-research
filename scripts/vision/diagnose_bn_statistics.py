"""Explicit, no-gradient BN buffer intervention. Defaults to preflight only."""
import argparse
import traceback
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
from scripts.vision.physical_low_light_capture import OUT as SOURCE,prior
from scripts.vision.run_fixed_budget_diagnosis import predict,paired_truth,score,summary,VARIANTS
from scripts.vision.evaluate_unified_lighting import forbidden
OUT=SOURCE/'bn-statistics-diagnosis-v1'

def exchange(receiver,donor):
    import torch
    aa,bb=receiver.state_dict(),donor.state_dict()
    names=[n for n,m in receiver.named_modules() if isinstance(m,torch.nn.BatchNorm2d)]
    other=[n for n,m in donor.named_modules() if isinstance(m,torch.nn.BatchNorm2d)]
    if len(names)!=81 or names!=other or set(aa)!=set(bb):raise ValueError('Unfused BN schema mismatch')
    allowed={n+'.'+k for n in names for k in ('running_mean','running_var')}
    before={n:t.clone() for n,t in aa.items()}
    with torch.no_grad():
        for n in allowed:
            if aa[n].shape!=bb[n].shape or not torch.isfinite(bb[n]).all():raise ValueError('Invalid donor buffer')
            aa[n].copy_(bb[n])
    for n,t in receiver.state_dict().items():
        if not torch.equal(t,bb[n] if n in allowed else before[n]):raise ValueError('Disallowed state change')
    return sorted(allowed)

def sources(seed):
    result={};deps=[]
    for family,root in (('B900',SOURCE.parent),('R1000',SOURCE)):
        cp=root/'training'/f'{family}-{seed}'/'completion.json';c=prior.read(cp);prior.verify(c)
        ep=root/'evaluation'/f'{family}-{seed}.json';e=prior.read(ep);prior.verify(e)
        wp=Path(c['weights'])
        if prior.file_sha256(wp)!=c['weights_sha256'] or e['inputs'].get(str(wp))!=c['weights_sha256']:raise ValueError('Weight binding')
        result[family]=(wp,e);deps.extend([cp,ep,wp])
    return result,deps

def run(seed):
    import torch
    from ultralytics import YOLO
    dest=OUT/f'seed-{seed}.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    root=OUT/f'seed-{seed}';root.mkdir(parents=True,exist_ok=True)
    n=len(list(root.glob('attempt-*')))+1
    if n>3:raise ValueError('Three technical attempts exhausted')
    attempt=root/f'attempt-{n:03}';attempt.mkdir()
    try:
        torch.set_num_threads(4);src,deps=sources(seed)
        dp=SOURCE/'design.json';p=prior.read(dp);prior.verify(p)
        rp,np=map(Path,(p['evaluation']['paired_review'],p['evaluation']['negative_review']))
        review,negative=prior.read(rp),prior.read(np)
        for r in (review,negative):prior.verify(r)
        pairs,inputs=paired_truth(review['frames'])
        if len(pairs)!=48 or len(negative['frames'])!=48:raise ValueError('Development coverage')
        for row in review['frames']+negative['frames']:
            if row['decision']!='accepted' or prior.file_sha256(row['image_path'])!=row['image_sha256']:raise ValueError('Stale development image')
            inputs[row['image_path']]=row['image_sha256']
        outputs={}
        with ExitStack() as stack:
            for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):
                stack.enter_context(patch.object(obj,name,forbidden))
            # Both originals must exactly reproduce all saved outputs before any hybrid is interpreted.
            for condition,recipient,donor in (('B','B900',None),('R','R1000',None),('R_Bstats','R1000','B900'),('B_Rstats','B900','R1000')):
                model=YOLO(str(src[recipient][0]));model.model.eval();changed=[]
                if donor:
                    source=YOLO(str(src[donor][0]));source.model.eval()
                    changed=exchange(model.model,source.model);del source
                rows=[];negs=[]
                with torch.inference_mode():
                    predict(model,review['frames'][0]['image_path'],.37);torch.set_num_threads(4)
                    for row,truth in pairs:
                        rows.append(score(row,truth,predict(model,row['image_path'],.37),predict(model,row['image_path'],.001)))
                    for row in negative['frames']:
                        a,b=predict(model,row['image_path'],.37),predict(model,row['image_path'],.001)
                        negs.append(dict(view_id=row['view_id'],variant=row['variant'],image_sha256=row['image_sha256'],predictions=a,diagnostic_predictions=b,frame_has_prediction=bool(a)))
                if not donor and (rows!=src[recipient][1]['rows'] or negs!=src[recipient][1]['negative_rows']):raise ValueError('Original prediction reproduction failed '+condition)
                outputs[condition]=dict(rows=rows,negative_rows=negs,allowed_replaced_buffers=changed,
                    summary={v:summary([x for x in rows if x['variant']==v]) for v in VARIANTS},
                    negative_summary=dict(frame_false_positive_rate=sum(bool(x['predictions']) for x in negs)/48,unmatched_predictions=sum(len(x['predictions']) for x in negs)))
                print(seed,condition,'complete',flush=True);del model
        deps.extend([dp,rp,np,Path(__file__).resolve()])
        deps.extend(prior.ROOT/'scripts/vision'/n for n in ('run_fixed_budget_diagnosis.py','run_order_diagnosis.py','evaluate_hard_negative_coverage.py','evaluate_exposure_diagnosis.py','exposure_metrics.py'))
        inputs.update({str(d):prior.file_sha256(d) for d in deps})
        return prior.frozen(dest,dict(status='inference_complete_review_pending',seed=seed,outputs=outputs,
            original_reproduction_exact=True,optimizer_created=False,backward_executed=False,validation_run=False,
            diagnostic_only=True,selected_candidate=None,inputs=inputs))
    except BaseException:
        prior.frozen(attempt/'failure.json',dict(error=traceback.format_exc(),child_processes_started=0));raise

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--infer',action='store_true');a.add_argument('--seed',type=int,choices=(7,17,27),default=7);args=a.parse_args()
    if args.infer:run(args.seed)
    else:
        for seed in (7,17,27):sources(seed)
        print('PREFLIGHT_ONLY_NO_MODEL_MUTATION_NO_INFERENCE')
