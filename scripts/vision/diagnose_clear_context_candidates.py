"""Pre-training inference on the already frozen new candidate set; no selection."""
from collections import Counter
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision.prepare_clear_context_increment import OUT,checked
from scripts.vision.reviewed_negative_order_control import OUT as CONTROL
from scripts.vision.evaluate_reactor_visibility_expansion import predict
from scripts.vision.evaluate_paired_visual_factors import match


def run():
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    dest=OUT/'pretraining-fit.json'
    if dest.exists():return checked(dest)
    mp=OUT/'dataset/manifest.json';p=checked(mp);rows=[];deps=[mp,Path(__file__)]
    with ExitStack() as stack:
        for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(YOLO,'train'),(YOLO,'val')):
            stack.enter_context(patch.object(obj,name,side_effect=AssertionError('Training forbidden')))
        for seed in (7,17,27):
            cp=CONTROL/'training'/f'reviewed-interleaved-480-{seed}'/'completion.json';c=checked(cp);deps.extend([cp,Path(c['weights'])])
            model=YOLO(c['weights'])
            for row in p['members']:
                pred=predict(model,row['image_path'],.37);low=predict(model,row['image_path'],.001)
                matched,_,used=match(pred,row['truth'])
                rows.append(dict(seed=seed,member_id=row['member_id'],review_id=row['review_id'],truth=row['truth'],predictions=pred,low_predictions=low,matches=matched,
                    matched_truth_indices=sorted(used),image_sha256=row['image_sha256'],label_sha256=row['label_sha256']))
            print('CANDIDATE_DIAGNOSIS',seed,'12 frames',flush=True)
    summary={}
    for seed in (7,17,27):
        units=[r for r in rows if r['seed']==seed];truth=Counter(o['class_name'] for r in units for o in r['truth']);hits=Counter(r['truth'][i]['class_name'] for r in units for i in r['matched_truth_indices'])
        summary[str(seed)]={k:dict(truth=v,matched=hits[k],recall=hits[k]/v) for k,v in truth.items()}
    return write_record(dest,dict(status='frozen_candidate_diagnosis_complete_no_training_or_member_selection',rows=rows,summary=summary,
        limits='All twelve members and quality decisions frozen before this inference. Results describe training candidates, not validation; seeds do not add independent data. No candidate is removed based on predictions.',
        training_admitted=False,promotable=False,inputs={str(x.resolve()):file_sha256(x) for x in deps}))


if __name__=='__main__':print(run()['summary'])
