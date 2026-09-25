"""Two independent full-length runs of the same canonicalized batch inputs."""
import sys,json,subprocess
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import scripts.vision.train_hard_negative_coverage as trainer
from scripts.vision.run_order_diagnosis import OUT as ORDER,REFERENCE,batches,read,save,file_sha256,verify_tree,exposures,predict,checked_rows,paired_truth,score,summary,VARIANTS,object_sha256
from scripts.vision.run_fixed_sequence_diagnosis import parameter_hash
from scripts.vision.canonical_batch_order import canonicalize
from scripts.vision.finalize_hard_negative_coverage_training import losses
ROOT_CAUSE=ORDER/'permutation-root-cause-v1'
OUT=ROOT_CAUSE/'canonical-fix-validation-v1'
KEYS=('CF-100-7','CW-100-7')

def prepare():
    path=OUT/'protocol.json'
    if path.exists():verify_tree(path);return read(path)
    verify_tree(ROOT_CAUSE/'trace.json');ref=read(REFERENCE/'protocol.json');other=read(ORDER/'protocol.json')
    raw=dict(zip(KEYS,(ref['schedules']['F-100-7'],other['schedules']['W-100-7'])))
    schedules={k:canonicalize(v) for k,v in raw.items()}
    if schedules[KEYS[0]]!=schedules[KEYS[1]]:raise ValueError('Canonical control differs')
    for k in KEYS:
        if any(Counter(a)!=Counter(b) for a,b in zip(batches(raw[k]),batches(schedules[k]))):raise ValueError('Batch membership changed')
    paths=[Path(__file__),ROOT/'scripts/vision/canonical_batch_order.py',ROOT/'tests/test_canonical_batch_order.py',ROOT_CAUSE/'trace.json']
    return save(path,dict(status='frozen',pool_rows=ref['pool_rows'],requested_schedules=raw,schedules=schedules,
        exposures={k:exposures(ref['pool_rows'],s) for k,s in schedules.items()},datasets={k:ref['datasets']['F-100-7'] for k in KEYS},controls=ref['controls'],
        policy='Canonical stable member-ID sorting BEFORE fetching/collation; preserve batch membership/order; record both requested and effective draws.',
        judgment='Fix passes only if both independent 100-step runs have exact state_dict, loss curves and all recorded predictions. Quality is separate; no candidate selection.',
        inputs={str(p):file_sha256(p) for p in paths}))

def main():
    from ultralytics import YOLO
    p=prepare();trainer.TRAIN=OUT
    for k in KEYS:trainer.train(k,p)
    reviewed,rpath=checked_rows();paired,receipt_inputs=paired_truth(reviewed)
    npath=trainer.BASE/'hard-negative-isolated-v2/semantic-review.json';neg=read(npath)
    if neg['status']!='reviewed' or neg['accepted']!=48 or neg['held']:raise ValueError('Review incomplete')
    inputs={str(q):file_sha256(q) for q in (OUT/'protocol.json',rpath,npath)};inputs.update(receipt_inputs)
    for row in reviewed+neg['frames']:
        if row['decision']!='accepted' or file_sha256(row['image_path'])!=row['image_sha256']:raise ValueError('Stale review')
        inputs[row['image_path']]=row['image_sha256']
    results={};curves={}
    for k in KEYS:
        cp=OUT/k/'completion.json';cell=trainer.checked_cell(cp,p);curves[k]=losses(Path(cell['exposure_path']).parent/'results.csv')
        ep=OUT/f'evaluation-{k}.json';sources={**inputs,str(cp):file_sha256(cp)}
        if ep.exists():
            verify_tree(ep);r=read(ep)
            if r['inputs']!=sources:raise ValueError('Stale evaluation')
        else:
            print('EVALUATE',k,flush=True);m=YOLO(cell['weights']);digest=parameter_hash(m)
            rows=[score(row,t,predict(m,row['image_path'],.37),predict(m,row['image_path'],.001)) for row,t in paired]
            negatives=[]
            for row in neg['frames']:
                preds=predict(m,row['image_path'],.37)
                negatives.append(dict(view_id=row['view_id'],variant=row['variant'],image_sha256=row['image_sha256'],predictions=preds,frame_has_prediction=bool(preds)))
            r=save(ep,dict(status='complete',inputs=sources,rows=rows,negative_rows=negatives,parameter_sha256=digest,
                summary={v:summary([x for x in rows if x['variant']==v]) for v in VARIANTS},
                negative_summary=dict(frame_false_positive_rate=sum(x['frame_has_prediction'] for x in negatives)/48,unmatched_predictions=sum(len(x['predictions']) for x in negatives)),
                matching_conflicts=sum(x['matching_conflict'] for x in rows)))
        if r['status']!='complete' or len(r['rows'])!=48 or len(r['negative_rows'])!=48 or r['matching_conflicts']:raise ValueError('Incomplete/conflicting evaluation')
        results[k]=r
    fields=('train/box_loss','train/cls_loss','train/dfl_loss')
    loss_equal=all([[r[n] for n in fields] for r in curves[k]]==[[r[n] for n in fields] for r in curves[KEYS[0]]] for k in KEYS)
    parameters_equal=len({r['parameter_sha256'] for r in results.values()})==1
    predictions_equal=len({object_sha256(dict(rows=r['rows'],negative_rows=r['negative_rows'])) for r in results.values()})==1
    if not (loss_equal and parameters_equal and predictions_equal):raise ValueError('Full-run invariance fix failed')
    verify_tree(OUT/'protocol.json')
    save(OUT/'completion.json',dict(status='invariance_fix_verified',selected_candidate=None,optimizer_steps=200,image_exposures=1200,
        identical_parameters=parameters_equal,identical_loss_curves=loss_equal,identical_predictions=predictions_equal,loss_curves=curves,
        inputs={str(q):file_sha256(q) for q in [OUT/'protocol.json']+[OUT/f'evaluation-{k}.json' for k in KEYS]}))
    print('FIX_VERIFIED',results[KEYS[0]]['negative_summary'],results[KEYS[0]]['summary'],flush=True)

if __name__=='__main__':main()
