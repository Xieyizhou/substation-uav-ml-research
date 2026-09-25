"""Frozen half-exposure grayscale ablation; sampling and geometry unchanged."""
import sys,random,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.run_stratified_negative_control import REFERENCE,BASE,read,save,file_sha256,verify_tree,exposures,predict,checked_rows,paired_truth,score,summary,VARIANTS,parameter_hash,retention
from scripts.vision.record_stratified_material_review import OUT as REVIEW
OUT=REFERENCE.parent/'grayscale-exposure-control-v1'
KEY='G-100-7'

def frozen_mask():
    slots=list(range(600));random.Random('grayscale-exposure-control-v1:mask').shuffle(slots)
    selected=set(slots[:300]);return [i in selected for i in range(600)]

def grayscale(images,mask):
    import torch
    if images.ndim!=4 or images.shape[1]!=3 or len(mask)!=len(images) or not images.is_floating_point():raise ValueError('Invalid RGB batch')
    output=images.clone();indices=[i for i,v in enumerate(mask) if v]
    if indices:
        rgb=images[indices];gray=(rgb[:,0:1]*.299+rgb[:,1:2]*.587+rgb[:,2:3]*.114)
        output[indices]=gray.repeat(1,3,1,1)
    return output

def tensor_hash(t):
    t=t.detach().cpu().contiguous();return hashlib.sha256(str(t.dtype).encode()+str(tuple(t.shape)).encode()+t.numpy().tobytes()).hexdigest()

def prepare():
    path=OUT/'protocol.json'
    if path.exists():verify_tree(path);return read(path)
    verify_tree(REVIEW/'completion.json');verify_tree(REFERENCE/'completion.json')
    ref=read(REFERENCE/'protocol.json');draws=ref['schedules']['CF-100-7'];mask=frozen_mask()
    paths=[Path(__file__),ROOT/'tests/test_grayscale_control.py',REVIEW/'completion.json',REFERENCE/'completion.json']
    return save(path,dict(status='frozen',pool_rows=ref['pool_rows'],schedules={KEY:draws},datasets={KEY:ref['datasets']['CF-100-7']},controls=ref['controls'],
        exposures={KEY:exposures(ref['pool_rows'],draws)},gray_mask=mask,grayscale_exposures=exposures(ref['pool_rows'],[m for m,on in zip(draws,mask) if on]),
        augmentation='Exactly 300 of 600 fixed exposures: RGB float grayscale Y=.299R+.587G+.114B, replicated to three channels AFTER standard normalization. No pixel geometry changes. Independent deterministic mask, no runtime RNG.',
        reference='CF-100-7, not the failed stratified S arm. Exact identical members, positions and class supervision. Other augmentation disabled.',
        scope='One-sequence seed 7 diagnostic. No new data/threshold/optimizer/architecture changes. Grayscale does not simulate all material or panel-contrast changes.',
        candidate_policy='No candidate selection; historical three-seed R/A and absolute gates remain required.',retention_tolerance=.05,
        inputs={str(q):file_sha256(q) for q in paths}))

def validate_ledger(ledger,p):
    rows=ledger['rows']
    if len(rows)!=600 or [x['position'] for x in rows]!=list(range(600)):raise ValueError('Incomplete augmentation ledger')
    if [x['member_id'] for x in rows]!=p['schedules'][KEY] or [x['grayscale'] for x in rows]!=p['gray_mask']:raise ValueError('Actual augmentation differs')
    for row in rows:
        if not row['labels_unchanged'] or not row['geometry_unchanged']:raise ValueError('Supervision changed')
        if not row['grayscale'] and row['before_sha256']!=row['after_sha256']:raise ValueError('Original exposure changed')
        if row['grayscale'] and not row['channels_equal']:raise ValueError('Not grayscale')

def train(p):
    import torch,ultralytics
    from torch.utils.data import DataLoader,Sampler
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    cp=OUT/KEY/'completion.json'
    if cp.exists():
        verify_tree(cp);r=read(cp)
        if r['status']!='complete' or r['protocol_identity']!=p['identity'] or r['optimizer_steps']!=100:raise ValueError('Stale completion')
        actual=read(r['exposure_path'])
        if actual['draws']!=p['schedules'][KEY] or actual['summary']!=p['exposures'][KEY]:raise ValueError('Stale exposure')
        validate_ledger(read(r['augmentation_path']),p);return r
    cell=OUT/KEY;cell.mkdir(parents=True,exist_ok=True)
    attempt=next((i for i in range(1,4) if not (cell/f'attempt-{i:03}').exists()),None)
    if attempt is None:raise ValueError('Attempt limit reached')
    name=f'attempt-{attempt:03}';run=cell/name;lookup={r['member_id']:r for r in p['pool_rows']};by_path={r['image_path']:r['member_id'] for r in lookup.values()};expected=p['schedules'][KEY];actual=[];audit=[]
    class Schedule(Sampler):
        def __init__(self,trainer,mapping):self.trainer,self.mapping=trainer,mapping
        def __len__(self):return 60
        def __iter__(self):
            start=self.trainer.epoch*60
            return iter(self.mapping[lookup[mid]['image_path']] for mid in expected[start:start+60])
    class Trainer(DetectionTrainer):
        step_count=0
        def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
            if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
            dataset=self.build_dataset(dataset_path,mode,batch_size);mapping={x:i for i,x in enumerate(dataset.im_files)}
            if set(mapping)!={lookup[m]['image_path'] for m in expected} or batch_size!=6:raise ValueError('Changed membership')
            return DataLoader(dataset,batch_size=6,sampler=Schedule(self,mapping),num_workers=0,collate_fn=dataset.collate_fn,generator=torch.Generator().manual_seed(7))
        def preprocess_batch(self,batch):
            batch=super().preprocess_batch(batch);start=len(actual);members=[by_path[x] for x in batch['im_file']];mask=p['gray_mask'][start:start+len(members)]
            before=batch['img'];labels={k:tensor_hash(batch[k]) for k in ('cls','bboxes','batch_idx')};after=grayscale(before,mask)
            labels_ok=all(tensor_hash(batch[k])==v for k,v in labels.items())
            for j,mid in enumerate(members):
                audit.append(dict(position=start+j,member_id=mid,grayscale=mask[j],before_sha256=tensor_hash(before[j]),after_sha256=tensor_hash(after[j]),
                    labels_unchanged=labels_ok,label_batch_hashes=labels,geometry_unchanged=before.shape==after.shape,
                    channels_equal=bool(torch.equal(after[j,0],after[j,1]) and torch.equal(after[j,1],after[j,2]))))
            actual.extend(members);batch['img']=after;return batch
        def optimizer_step(self):self.step_count+=1;return super().optimizer_step()
    try:
        model=YOLO(p['controls']['initial_weights'])
        model.train(trainer=Trainer,data=p['datasets'][KEY],epochs=10,imgsz=640,batch=6,nbs=6,device='cpu',workers=0,optimizer='AdamW',lr0=.001,lrf=1.,warmup_epochs=0,warmup_bias_lr=0,seed=7,deterministic=True,patience=0,amp=False,
            mosaic=0,close_mosaic=0,mixup=0,copy_paste=0,degrees=0,translate=0,scale=0,shear=0,perspective=0,flipud=0,fliplr=0,hsv_h=0,hsv_s=0,hsv_v=0,project=str(cell),name=name,plots=False,save=True,val=False)
        if actual!=expected or model.trainer.step_count!=100:raise ValueError('Actual exposures/steps mismatch')
        ledger=dict(rows=audit);validate_ledger(ledger,p)
        ap=run/'augmentation.json';save(ap,ledger);ep=run/'exposure.json';save(ep,dict(draws=actual,summary=exposures(p['pool_rows'],actual)))
        weight=run/'weights/last.pt'
        return save(cp,dict(status='complete',cell=KEY,protocol_identity=p['identity'],weights=str(weight),weights_sha256=file_sha256(weight),optimizer_steps=100,exposure_path=str(ep),augmentation_path=str(ap),torch_version=torch.__version__,ultralytics_version=ultralytics.__version__,
            inputs={str(q):file_sha256(q) for q in (OUT/'protocol.json',weight,ap,ep,run/'args.yaml',run/'results.csv')}))
    except BaseException:
        save(run/'failure.json',dict(status='failed',error=traceback.format_exc(),actual_draws=actual,augmentation_rows=audit));raise

def main():
    from ultralytics import YOLO
    p=prepare();cell=train(p);reviewed,rpath=checked_rows();paired,receipt_inputs=paired_truth(reviewed)
    npath=BASE/'hard-negative-isolated-v2/semantic-review.json';neg=read(npath)
    if neg['status']!='reviewed' or neg['accepted']!=48 or neg['held']:raise ValueError('Incomplete review')
    rp=REFERENCE/'evaluation-CF-100-7.json';verify_tree(rp);ref=read(rp);cp=OUT/KEY/'completion.json'
    sources={str(q):file_sha256(q) for q in (OUT/'protocol.json',cp,rpath,npath,rp)};sources.update(receipt_inputs)
    for row in reviewed+neg['frames']:
        if row['decision']!='accepted' or file_sha256(row['image_path'])!=row['image_sha256']:raise ValueError('Stale review')
        sources[row['image_path']]=row['image_sha256']
    ep=OUT/'evaluation.json'
    if ep.exists():
        verify_tree(ep);r=read(ep)
        if r['inputs']!=sources:raise ValueError('Stale evaluation')
    else:
        print('EVALUATE_GRAY_ENDPOINT',flush=True);model=YOLO(cell['weights']);digest=parameter_hash(model)
        rows=[score(row,t,predict(model,row['image_path'],.37),predict(model,row['image_path'],.001)) for row,t in paired];negatives=[]
        for row in neg['frames']:
            ps=predict(model,row['image_path'],.37);negatives.append(dict(view_id=row['view_id'],variant=row['variant'],image_sha256=row['image_sha256'],predictions=ps,frame_has_prediction=bool(ps)))
        r=save(ep,dict(status='complete',inputs=sources,rows=rows,negative_rows=negatives,parameter_sha256=digest,summary={v:summary([x for x in rows if x['variant']==v]) for v in VARIANTS},negative_summary=dict(frame_false_positive_rate=sum(x['frame_has_prediction'] for x in negatives)/48,unmatched_predictions=sum(len(x['predictions']) for x in negatives)),matching_conflicts=sum(x['matching_conflict'] for x in rows)))
    if len(r['rows'])!=48 or len(r['negative_rows'])!=48 or r['matching_conflicts']:raise ValueError('Incomplete/conflicting evaluation')
    gains={}
    for v in VARIANTS:
        a={x['pair_id']:x for x in r['rows'] if x['variant']==v};b={x['pair_id']:x for x in ref['rows'] if x['variant']==v}
        if set(a)!=set(b) or len(a)!=12:raise ValueError('Pair mismatch')
        gains[v]=dict(gained=[k for k in a if a[k]['planned_assigned_hit'] and not b[k]['planned_assigned_hit']],lost=[k for k in a if b[k]['planned_assigned_hit'] and not a[k]['planned_assigned_hit']])
    from scripts.vision.finalize_hard_negative_coverage_training import losses
    save(OUT/'completion.json',dict(status='grayscale_control_complete',selected_candidate=None,reference_summary=ref['summary'],grayscale_summary=r['summary'],reference_negative=ref['negative_summary'],grayscale_negative=r['negative_summary'],retention_vs_reference=retention(r['summary'],ref['summary']),paired_gains_losses=gains,
        optimizer_steps=100,image_exposures=600,grayscale_exposures=300,loss_curve=losses(Path(cell['exposure_path']).parent/'results.csv'),training_validation_role='training fit only',inputs={str(q):file_sha256(q) for q in (OUT/'protocol.json',ep,rp,cp)}))
    print('GRAYSCALE_COMPLETE',r['negative_summary'],flush=True)

if __name__=='__main__':main()
