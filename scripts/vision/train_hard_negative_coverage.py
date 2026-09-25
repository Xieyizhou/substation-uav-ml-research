"""Six independent fixed-budget old/expanded-negative development training cells."""
import argparse
import json
import random
import sys
import traceback
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.hard_negative_coverage import OUT,BASE,read,save,verify,file_sha256
from scripts.vision.exposure_protocol import exposures,cycle
from scripts.vision.hard_negative_coverage_admission import validate_decisions
from src.ml.artifacts import object_sha256
TRAIN=OUT/'training-v1'

def verify_tree(path,seen=None):
    seen=set() if seen is None else seen
    path=Path(path)
    if str(path) in seen:return
    seen.add(str(path));record=read(path)
    if 'identity' not in record:return
    verify(record)
    for child in record.get('inputs',{}):
        if Path(child).suffix=='.json':verify_tree(child,seen)


def expanded_draws(prior_rows,new_rows,seed,old_draws):
    rows={r['member_id']:r for r in prior_rows+new_rows}
    groups=defaultdict(list)
    for r in prior_rows+new_rows:
        if r['subset']=='hard_negative':groups[r['lineage_id']].append(r['member_id'])
    if len(groups)!=60 or any(len(v)!=2 for v in groups.values()):
        raise ValueError('Expected 60 complete negative pairs')
    rng=random.Random(f'coverage-v1:{seed}')
    keys=sorted(groups);rng.shuffle(keys)
    negative=[mid for group in keys[:54] for mid in sorted(groups[group])];rng.shuffle(negative)
    iterator=iter(negative)
    draws=[next(iterator) if rows[mid]['subset']=='hard_negative' else mid for mid in old_draws]
    if len(draws)!=600 or sum(rows[mid]['subset']=='hard_negative' for mid in draws)!=108:
        raise ValueError('Quota mismatch')
    return draws,keys[54:]


def prepare():
    path=TRAIN/'protocol.json'
    if path.exists():
        verify_tree(path);record=read(path);return record
    prior_path=BASE/'exposure-controlled-diagnosis-v1/protocol.json';prior=read(prior_path);verify(prior)
    inputs={str(prior_path):file_sha256(prior_path),str(Path(__file__)):file_sha256(Path(__file__)),
        str(ROOT/'docs/hard-negative-coverage-training-v1.md'):file_sha256(ROOT/'docs/hard-negative-coverage-training-v1.md')}
    # Resolve historical negative pair identity from the supplemental ledger.
    admission_path=BASE/'visual-bridge-supplement-v2/development-admission.json';admission=read(admission_path);verify(admission)
    inputs[str(admission_path)]=file_sha256(admission_path)
    source={r['member_id']:r for r in admission['entries']}
    rows=[]
    for r in prior['pool_rows']:
        r=dict(r)
        if r['subset']=='hard_negative':r['lineage_id']=source[r['member_id']]['derivation_group']
        rows.append(r)
    new=[]
    TRAIN.mkdir(parents=True,exist_ok=True)
    from PIL import Image
    for stage,n in (('final',96),):
        p=OUT/f'{stage}-admission.json';verify_tree(p);a=read(p)
        if a['status']!='accepted' or a['accepted']!=n:raise ValueError('Incomplete coverage admission')
        validate_decisions(a['frames'],a['decisions']);inputs[str(p)]=file_sha256(p)
        for r in a['frames']:
            image_path=TRAIN/'dataset/images'/f'{r["view_id"]}.png'
            label_path=TRAIN/'dataset/labels'/f'{r["view_id"]}.txt'
            image_path.parent.mkdir(parents=True,exist_ok=True);label_path.parent.mkdir(parents=True,exist_ok=True)
            with Image.open(r['image_path']) as image:image.convert('RGB').save(image_path)
            with label_path.open('x') as f:f.write('')
            new.append(dict(member_id=f'coverage:{r["view_id"]}',subset='hard_negative',image_path=str(image_path),
                label_path=str(label_path),image_sha256=file_sha256(image_path),label_sha256=file_sha256(label_path),
                source_image_path=r['image_path'],source_image_sha256=r['image_sha256'],lineage_id=r['derivation_group'],
                pair_id=r['pair_id'],correlation_group_id=r['correlation_group_id'],coverage_unit=r['coverage_unit'],class_instances={},data_role='development_training_only'))
    if len(new)!=96 or len({r['lineage_id'] for r in new})!=48:raise ValueError('Incomplete pairs')
    schedules={};exposure={};datasets={};omitted={}
    all_rows=rows+new;lookup={r['member_id']:r for r in all_rows}
    for r in all_rows:
        for kind in ('image','label'):
            p=r[f'{kind}_path'];digest=file_sha256(p)
            if digest!=r[f'{kind}_sha256']:raise ValueError('Input member stale')
            inputs[p]=digest
    for seed in (7,17,27):
        old=prior['schedules'][f'Y-100-{seed}'];expanded,missing=expanded_draws(rows,new,seed,old)
        omitted[str(seed)]=missing
        for arm,draws in (('O',old),('N',expanded)):
            key=f'{arm}-100-{seed}';schedules[key]=draws;exposure[key]=exposures(all_rows,draws)
            members=sorted(set(draws))
            listing=TRAIN/f'{key}.txt';listing.write_text('\n'.join(lookup[mid]['image_path'] for mid in members)+'\n')
            dataset=TRAIN/f'{key}.yaml';dataset.write_text(f'path: {TRAIN}\ntrain: {listing}\nval: {listing}\nnames: {json.dumps(prior["controls"].get("names",["transformer","switchgear","capacitor_bank","reactor"]))}\n')
            datasets[key]=str(dataset);inputs[str(listing)]=file_sha256(listing);inputs[str(dataset)]=file_sha256(dataset)
            if exposure[key]['by_subset']!={'base':216,'regular':156,'bridge_positive':120,'hard_negative':108}:raise ValueError('Quota changed')
        for a,b in zip(old,expanded):
            if lookup[a]['subset']!='hard_negative' and a!=b:raise ValueError('Positive exposure changed')
        if exposure[f'O-100-{seed}']['class_instance_exposure']!=exposure[f'N-100-{seed}']['class_instance_exposure']:
            raise ValueError('Class supervision changed')
    weight=prior['controls']['initial_weights'];inputs[weight]=file_sha256(weight)
    return save(path,dict(status='frozen',pool_rows=all_rows,schedules=schedules,exposures=exposure,datasets=datasets,
        controls={**prior['controls'],'initial_weights':weight},omitted_negative_pairs_by_seed=omitted,inputs=inputs,
        comparison='fixed 108 negative slots: old pool vs expanded pool; positives unchanged',expected_cells=6))


def checked_cell(path,protocol):
    verify_tree(TRAIN/'protocol.json');record=read(path);verify(record)
    actual=read(record['exposure_path']);verify(actual)
    key=record['cell']
    if record['status']!='complete' or record['protocol_identity']!=protocol['identity'] or actual['draws']!=protocol['schedules'][key] or actual['summary']!=protocol['exposures'][key] or record['optimizer_steps']!=100:
        raise ValueError('Invalid resumed training cell')
    return record


def train(key,protocol):
    import torch,ultralytics
    from torch.utils.data import DataLoader,Sampler
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    completion=TRAIN/key/'completion.json'
    if completion.exists():return checked_cell(completion,protocol)
    seed=int(key.split('-')[-1]);expected=protocol['schedules'][key];rows={r['member_id']:r for r in protocol['pool_rows']}
    path_members={r['image_path']:r['member_id'] for r in rows.values()};actual=[]
    cell=TRAIN/key;cell.mkdir(parents=True,exist_ok=True)
    attempt=next((i for i in range(1,4) if not (cell/f'attempt-{i:03}').exists()),None)
    if attempt is None:raise ValueError('Technical attempt budget exhausted')
    name=f'attempt-{attempt:03}';run=cell/name
    class Schedule(Sampler):
        def __init__(self,trainer,mapping):self.trainer,self.mapping=trainer,mapping
        def __len__(self):return 60
        def __iter__(self):
            start=self.trainer.epoch*60
            return iter(self.mapping[rows[mid]['image_path']] for mid in expected[start:start+60])
    class Trainer(DetectionTrainer):
        step_count=0
        def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
            if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
            dataset=self.build_dataset(dataset_path,mode,batch_size);mapping={p:i for i,p in enumerate(dataset.im_files)}
            if set(mapping)!={rows[mid]['image_path'] for mid in expected} or batch_size!=6:raise ValueError('Runtime membership changed')
            return DataLoader(dataset,batch_size=6,sampler=Schedule(self,mapping),num_workers=0,
                collate_fn=dataset.collate_fn,generator=torch.Generator().manual_seed(seed))
        def preprocess_batch(self,batch):
            actual.extend(path_members[p] for p in batch['im_file']);return super().preprocess_batch(batch)
        def optimizer_step(self):self.step_count+=1;return super().optimizer_step()
    try:
        verify_tree(TRAIN/'protocol.json')
        model=YOLO(protocol['controls']['initial_weights'])
        model.train(trainer=Trainer,data=protocol['datasets'][key],epochs=10,imgsz=640,batch=6,nbs=6,device='cpu',workers=0,
            optimizer='AdamW',lr0=.001,lrf=1.,warmup_epochs=0,warmup_bias_lr=0,seed=seed,deterministic=True,patience=0,
            amp=False,mosaic=0,close_mosaic=0,mixup=0,copy_paste=0,degrees=0,translate=0,scale=0,shear=0,perspective=0,
            flipud=0,fliplr=0,hsv_h=0,hsv_s=0,hsv_v=0,project=str(cell),name=name,plots=False,save=True,val=False)
        if actual!=expected or model.trainer.step_count!=100:raise ValueError('Actual exposure/steps mismatch')
        exposure_path=run/'exposure.json';save(exposure_path,dict(draws=actual,summary=exposures(protocol['pool_rows'],actual)))
        weight=run/'weights/last.pt'
        result=save(completion,dict(status='complete',cell=key,protocol_identity=protocol['identity'],weights=str(weight),
            weights_sha256=file_sha256(weight),exposure_path=str(exposure_path),optimizer_steps=100,
            torch_version=torch.__version__,ultralytics_version=ultralytics.__version__,
            inputs={str(p):file_sha256(p) for p in (weight,exposure_path,run/'results.csv',run/'args.yaml',TRAIN/'protocol.json')}))
        print('CELL_COMPLETE',key,flush=True);return result
    except BaseException:
        save(run/'failure.json',dict(status='failed',cell=key,error=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');a=p.parse_args();protocol=prepare()
    if not a.prepare_only:
        completed=[]
        for key in protocol['schedules']:
            completed.append(train(key,protocol))
            save(TRAIN/'progress.json',dict(status='training_complete' if len(completed)==6 else 'training_in_progress',
                completed_cells=[r['cell'] for r in completed],protocol_identity=protocol['identity'],
                inputs={str(TRAIN/r['cell']/'completion.json'):file_sha256(TRAIN/r['cell']/'completion.json') for r in completed}))
