"""Fixed-protocol development evaluation for the endpoint decomposition."""
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from scripts.vision.scale_endpoint_control import OUT,KEYS,prior,complete,inputs,attempt_folder,failure
from scripts.vision.evaluate_frozen_multiscale import validate_record,forbidden,predict,paired_truth,score,summary,VARIANTS,aggregate,policy_checks,direct_checks,PRIOR,baseline_verify
from scripts.vision import train_frozen_multiscale as old

def evaluate(key):
    cell=complete(key);_,p,_,_=inputs(key)
    dest=OUT/'evaluation'/f'{key}.json'
    if dest.exists():r=prior.read(dest);validate_record(r,key);return r
    attempt=attempt_folder(OUT/'evaluation'/key)
    try:
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        rp,np=map(Path,(p['evaluation']['paired_review'],p['evaluation']['negative_review']))
        rev,neg=prior.read(rp),prior.read(np)
        for x in (rev,neg):prior.verify(x)
        paired,hashes=paired_truth(rev['frames'])
        if len(paired)!=48 or len(neg['frames'])!=48:raise ValueError('Missing frames')
        for x in rev['frames']+neg['frames']:
            if x['decision']!='accepted' or prior.file_sha256(x['image_path'])!=x['image_sha256']:raise ValueError('Stale review')
            hashes[x['image_path']]=x['image_sha256']
        rows=[];negative=[]
        with ExitStack() as stack:
            for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):stack.enter_context(patch.object(obj,name,forbidden))
            model=YOLO(cell['weights'])
            with torch.inference_mode():
                for x,t in paired:rows.append(score(x,t,predict(model,x['image_path'],.37),predict(model,x['image_path'],.001)))
                for x in neg['frames']:
                    a,b=predict(model,x['image_path'],.37),predict(model,x['image_path'],.001)
                    negative.append(dict(view_id=x['view_id'],variant=x['variant'],image_sha256=x['image_sha256'],predictions=a,diagnostic_predictions=b,frame_has_prediction=bool(a)))
        paths=[OUT/'protocol.json',OUT/'training'/key/'completion.json',rp,np,Path(cell['weights']),Path(__file__).resolve()]
        paths += [prior.ROOT/'scripts/vision'/n for n in ('exposure_metrics.py','evaluate_exposure_diagnosis.py','evaluate_visual_augmentation_abcd.py','evaluate_paired_visual_factors.py')]
        hashes.update({str(x):prior.file_sha256(x) for x in paths})
        r=prior.frozen(dest,dict(status='complete',cell=key,rows=rows,negative_rows=negative,
            matching_conflicts=sum(bool(x['matching_conflict']) for x in rows),
            summary={v:summary([x for x in rows if x['variant']==v]) for v in VARIANTS},
            negative_summary=dict(frame_false_positive_rate=sum(x['frame_has_prediction'] for x in negative)/48,unmatched_predictions=sum(len(x['predictions']) for x in negative)),
            optimizer_created=False,backward_executed=False,validation_run=False,inputs=hashes))
        validate_record(r,key);return r
    except BaseException as ex:failure(attempt,ex);raise

def transitions(reference,current):
    from scripts.vision.diagnose_multiscale_errors import truth_key
    a={(r['pair_id'],r['variant']):r for r in reference['rows']};result=[]
    for b in current['rows']:
        oldrow=a[(b['pair_id'],b['variant'])]
        truth={truth_key(t):i for i,t in enumerate(oldrow['truth'])}
        if len(truth)!=len(oldrow['truth']) or set(truth)!={truth_key(t) for t in b['truth']}:raise ValueError('Truth correspondence failed')
        ha={m['truth_index'] for m in oldrow['matches']};hb={m['truth_index'] for m in b['matches']}
        for j,t in enumerate(b['truth']):
            before=truth[truth_key(t)] in ha;after=j in hb
            state='retained_hit' if before and after else 'loss' if before else 'gain' if after else 'retained_miss'
            result.append(dict(pair_id=b['pair_id'],variant=b['variant'],truth=t,state=state,miss=next((m for m in b['misses'] if m['truth_index']==j),None)))
    return result

def run():
    records={}
    for key in KEYS:records[key]=evaluate(key);print('EVALUATED',key,records[key]['negative_summary'],flush=True)
    paths=[OUT/'evaluation'/f'{k}.json' for k in KEYS]
    for key in old.KEYS:
        old.complete(key);path=old.OUT/'evaluation'/f'{key}.json';records[key]=prior.read(path);validate_record(records[key],key);paths.append(path)
    _,p,_,_=inputs(KEYS[0])
    groups={a:aggregate([records[f'{a}-{s}'] for s in (7,17,27)]) for a in ('fixed','multiscale','small','large')}
    refs=[PRIOR/f'evaluation-retained_reference-450-{s}.json' for s in (7,17,27)]
    hp=Path(p['evaluation']['historical_reference']);history=prior.read(hp);reference=[prior.read(x) for x in refs]
    for x in reference+[history]:prior.verify(x)
    baseline=baseline_verify(prior.ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40:raise ValueError('Baseline integrity failed')
    paths += refs+[hp,OUT/'protocol.json',Path(__file__).resolve()]
    prior.frozen(OUT/'evaluation/summary.json',dict(status='numerical_complete_error_review_pending',aggregate=groups,
        policy_results={a:policy_checks(groups[a],aggregate(reference),history['historical_A'],p) for a in ('small','large')},
        direct_retention={a:direct_checks(groups[a],groups['fixed']) for a in ('small','large')},
        transitions={k:transitions(records[f"fixed-{k.rsplit('-',1)[1]}"],records[k]) for k in KEYS},
        matching_conflicts=sum(records[k]['matching_conflicts'] for k in KEYS),selected_candidate=None,baseline=baseline,
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print('NUMERICAL_COMPLETE_ERROR_REVIEW_PENDING',flush=True)

if __name__=='__main__':run()
