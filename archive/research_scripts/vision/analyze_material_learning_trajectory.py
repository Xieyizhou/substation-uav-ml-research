"""Explicit offline raw/EMA trajectory inference. No training or checkpoint selection."""
import argparse
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from scripts.vision.train_material_learning_trajectory import OUT as TRAIN,KEYS,prior,complete,historical
from scripts.vision.material_trajectory_runtime import state_digest
from scripts.vision.structure_fit import scoring
from scripts.vision.evaluate_exposure_diagnosis import predict
from scripts.vision.infer_material_member_fit import forbidden

OUT=TRAIN/'analysis-v1'
MODES=('ema','raw')
DEV_STEPS=(0,150,300,450)


def classify(hits):
    if not hits:raise ValueError('Empty trajectory')
    if hits[-1]:return 'endpoint_hit_with_intermediate_loss' if any(not x for x in hits[1:-1]) and any(hits[:-1]) else 'endpoint_hit'
    if any(hits[:-1]):return 'previously_hit_endpoint_miss'
    return 'never_hit_at_observed_steps'


def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():p=prior.read(dest);prior.verify(p);return p
    fp=historical.OUT/'fit-transfer-diagnosis-v1/protocol.json';f=prior.read(fp);prior.verify(f)
    ep=historical.OUT/'evaluation/T-7.json';e=prior.read(ep);prior.verify(e)
    image_paths={}
    for path,digest in e['inputs'].items():
        if Path(path).suffix.lower() in ('.png','.jpg','.jpeg','.ppm'):
            image_paths.setdefault(digest,[]).append(path)
    dev=[]
    for index,row in enumerate(e['rows']+e['negative_rows']):
        paths=sorted(image_paths.get(row['image_sha256'],[]))
        if not paths:raise ValueError('Development RGB source missing')
        dev.append(dict(member_id='dev-'+str(index),image_path=paths[0],image_sha256=row['image_sha256'],
            image_path_aliases=paths,truth=row.get('truth',[]),condition=row['variant'],
            pair_id=row.get('pair_id'),view_id=row['view_id'],planned_truth_index=row.get('planned_truth_index'),
            identity_scope='Image hash plus full truth coordinates; no cross-image instance claim'))
    cells={};inputs={str(fp):prior.file_sha256(fp),str(ep):prior.file_sha256(ep),
                    str(Path(__file__).resolve()):prior.file_sha256(__file__)}
    for key in KEYS:
        complete(key);cp=TRAIN/'training'/key/'completion.json';c=prior.read(cp)
        sp=Path(c['snapshot_receipt']);s=prior.read(sp);prior.verify(s)
        cells[key]=s['checkpoints'];inputs.update(s['inputs'])
        inputs.update({str(cp):prior.file_sha256(cp),str(sp):prior.file_sha256(sp)})
    members=f['members']
    if len(members)!=84 or len(dev)!=96:raise ValueError('Incomplete datasets')
    inputs.update({m['image_path']:m['image_sha256'] for m in members+dev})
    for name in ('structure_fit.py','evaluate_exposure_diagnosis.py','material_trajectory_runtime.py'):
        path=Path(__file__).with_name(name);inputs[str(path)]=prior.file_sha256(path)
    OUT.mkdir(exist_ok=True)
    return prior.frozen(dest,dict(status='offline_trajectory_inference_frozen',cells=cells,members=members,
        development_members=dev,development_steps=list(DEV_STEPS),modes=list(MODES),environment=f['environment'],
        actual_exposures=f['actual_exposures'],inference=f['inference'],
        limits=['Raw and EMA kept separate; no intermediate checkpoint selection.',
                'Full-precision snapshots differ from half-serialized historical endpoint precision.',
                'Unobserved steps may contain additional transitions; origin RGB is not augmented training tensor.',
                'Same-source variants and repeated models do not increase independent scene count.'],inputs=inputs))


def validate(r,p,key,checkpoint,mode):
    prior.verify(r)
    if (r['cell'],r['step'],r['mode'],r['protocol_identity'])!=(key,checkpoint['step'],mode,p['identity']):
        raise ValueError('Wrong inference identity')
    members=p['members']+(p['development_members'] if checkpoint['step'] in DEV_STEPS else [])
    if len(r['rows'])!=len(members):raise ValueError('Missing member inference')
    for m,row in zip(members,r['rows'],strict=True):
        if row['member_id']!=m['member_id'] or row['image_sha256']!=m['image_sha256']:raise ValueError('Member drift')
        score=scoring(m['truth'],row['predictions'],row['low_predictions'])
        if any(row[k]!=v for k,v in score.items()):raise ValueError('Full matching drift')


def infer(p,key,c,mode):
    unit=OUT/key/f"{c['step']:03}-{mode}";dest=unit/'complete.json'
    if dest.exists():r=prior.read(dest);validate(r,p,key,c,mode);return
    unit.mkdir(parents=True,exist_ok=True)
    if any(prior.read(f)['status']=='semantic_stop' for f in unit.glob('attempt-*/failure.json')):raise ValueError('Semantic stop')
    n=len(list(unit.glob('attempt-*')))+1
    if n>3:raise ValueError('Attempt cap')
    attempt=unit/f'attempt-{n:03}';attempt.mkdir()
    try:
        import torch,ultralytics
        from ultralytics import YOLO
        if dict(torch=torch.__version__,ultralytics=ultralytics.__version__)!=p['environment']:raise ValueError('Environment drift')
        if prior.file_sha256(c['path'])!=c['sha256']:raise ValueError('Snapshot changed')
        torch.set_num_threads(4)
        with ExitStack() as stack:
            for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):
                stack.enter_context(patch.object(obj,name,forbidden))
            payload=torch.load(c['path'],map_location='cpu',weights_only=False)
            selected=payload['ema' if mode=='ema' else 'model']
            if state_digest(selected)!=c[mode+'_state_sha256']:raise ValueError('State identity mismatch')
            model=YOLO(c['path']);model.model=selected.float().eval();model.predictor=None
            rows=[]
            members=p['members']+(p['development_members'] if c['step'] in DEV_STEPS else [])
            with torch.inference_mode():
                for m in members:
                    rows.append(dict(member_id=m['member_id'],image_sha256=m['image_sha256'],
                        **scoring(m['truth'],predict(model,m['image_path'],.37),predict(model,m['image_path'],.001))))
        r=prior.frozen(attempt/'result.json',dict(status='complete',cell=key,step=c['step'],mode=mode,
            protocol_identity=p['identity'],rows=rows,optimizer_created=False,backward_executed=False,
            training_validation_executed=False,inputs={str(OUT/'protocol.json'):prior.file_sha256(OUT/'protocol.json'),c['path']:c['sha256']}))
        validate(r,p,key,c,mode)
        prior.frozen(dest,dict(**{k:v for k,v in r.items() if k not in ('identity','inputs')},
            inputs={**r['inputs'],str(attempt/'result.json'):prior.file_sha256(attempt/'result.json')}))
        print('COMPLETE',key,c['step'],mode,len(rows),flush=True)
    except BaseException as e:
        prior.frozen(attempt/'failure.json',dict(status='semantic_stop' if isinstance(e,ValueError) else 'technical_stop',
            error=traceback.format_exc(),child_processes_started=0));raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');a=ap.parse_args();p=freeze()
    if not a.infer:print('PREFLIGHT_ONLY');return
    # Chronological output per seed; every frozen checkpoint and both modes required.
    for key,cs in p['cells'].items():
        for c in cs:
            for mode in MODES:infer(p,key,c,mode)


if __name__=='__main__':main()
