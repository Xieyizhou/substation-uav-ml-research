"""Explicit inference-only fitting diagnosis for all twelve new members."""
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision.prepare_clear_context_increment import OUT as DATA, checked
from scripts.vision.freeze_clear_context_training import OUT, KEYS
from scripts.vision.evaluate_reactor_visibility_expansion import predict
from scripts.vision.evaluate_paired_visual_factors import match


def run():
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    dest=OUT/'new-member-fit-v1.json'
    if dest.exists():return checked(dest)
    mp=DATA/'dataset/manifest.json'; members=checked(mp)['members']
    before_path=DATA/'pretraining-fit.json'; before=checked(before_path)
    dependencies=[mp,before_path,Path(__file__)]; rows=[]
    with ExitStack() as stack:
        for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(YOLO,'train'),(YOLO,'val')):
            stack.enter_context(patch.object(obj,name,side_effect=AssertionError('Training forbidden in diagnosis')))
        for seed,key in zip((7,17,27),KEYS):
            cp=OUT/'training'/key/'completion.json'; completion=checked(cp)
            exposure_path=Path(completion['exposure_path']); exposure=checked(exposure_path)
            counts=Counter(exposure['actual']); weights=Path(completion['weights'])
            dependencies.extend([cp,exposure_path,weights]); model=YOLO(str(weights))
            for member in members:
                mid=member['member_id']
                if counts[mid]!=10:raise ValueError('Unexpected actual new-member exposure')
                for path_field,hash_field in [('image_path','image_sha256'),('label_path','label_sha256')]:
                    if file_sha256(member[path_field])!=member[hash_field]:raise ValueError('Stale member')
                    dependencies.append(Path(member[path_field]))
                pred=predict(model,member['image_path'],.37); low=predict(model,member['image_path'],.001)
                matched,_,used=match(pred,member['truth'])
                previous=next(r for r in before['rows'] if r['seed']==seed and r['member_id']==mid)
                if previous['truth']!=member['truth']:raise ValueError('Truth mismatch')
                old=set(previous['matched_truth_indices'])
                rows.append(dict(seed=seed,member_id=mid,truth=member['truth'],predictions=pred,low_predictions=low,
                    matches=matched,matched_truth_indices=sorted(used),previous_matched_truth_indices=sorted(old),
                    actual_exposures=counts[mid],gains=sorted(used-old),losses=sorted(old-used)))
            print('FIT_COMPLETE',seed,flush=True)
    summary={}
    for seed in (7,17,27):
        subset=[r for r in rows if r['seed']==seed]
        truth=Counter(t['class_name'] for r in subset for t in r['truth'])
        hits=Counter(r['truth'][i]['class_name'] for r in subset for i in r['matched_truth_indices'])
        summary[str(seed)]={c:dict(truth=n,matched=hits[c],recall=hits[c]/n) for c,n in truth.items()}
    return write_record(dest,dict(status='training_member_fit_diagnosis_complete',rows=rows,summary=summary,
        limits='Training fit only; not development/generalization. Twelve already-frozen members, all seeds retained.',
        training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in dependencies}))


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');args=ap.parse_args()
    if args.infer:print(run()['summary'])
    else:print('Preflight only; use --infer for diagnostic inference. No training entry.')
