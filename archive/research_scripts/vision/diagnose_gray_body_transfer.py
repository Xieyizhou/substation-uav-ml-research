"""Explicit existing-weight factorial diagnostic; never train or admit data."""
import argparse
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from scripts.vision.import_transfer_pilot_review import DEST as REVIEW, main as review, prior
from scripts.vision.freeze_condition_transfer_probe import OUT as DESIGN
from scripts.vision.audit_material_transfer_scope import CAND, resolve_truth
from scripts.vision.structure_fit import scoring
from scripts.vision.evaluate_exposure_diagnosis import predict
from scripts.vision.infer_material_member_fit import forbidden

from collections import Counter
from PIL import Image
from scripts.vision.train_gray_body_control import OUT as TRAIN, KEYS, contract, complete
from scripts.vision.infer_transfer_pilot_v2 import OUT as PILOT
from scripts.vision.infer_transfer_expansion import OUT as EXPANSION

OUT=TRAIN/'training-fit-transfer-diagnosis-v1'

def check_gray(row,member):
    if row['member_id']!=member['member_id']:raise ValueError('Wrong member')
    a,b=Image.open(row['image_path']).convert('RGB'),Image.open(member['image_path']).convert('RGB')
    if a.size!=b.size or a.tobytes()!=b.tobytes():raise ValueError('Export pixels differ')
    truth=[{k:v for k,v in t.items() if k!='object_id'} for t in row['truth']]
    if truth!=member['full_truth']['objects']:raise ValueError('Complete truth changed')
    if len([t for t in row['truth'] if t['object_id']==row['target']])!=1:raise ValueError('Ambiguous target')

def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():
        p=prior.read(dest);prior.verify(p);return p
    p,_,_=contract(KEYS[0])
    paths=[TRAIN/'protocol.json',Path(__file__).resolve()]
    sources=[]
    for directory in (PILOT,EXPANSION):
        path=directory/'protocol.json';q=prior.read(path);prior.verify(q)
        paths.append(path);sources.extend(q['members'])
    if len(sources)!=84 or len({r['member_id'] for r in sources})!=84:raise ValueError('Incomplete diagnostic matrix')
    if len({r['source_id'] for r in sources})!=12:raise ValueError('Source scope drift')
    pool={r['member_id']:r for r in p['pool_rows']}
    gray=[r for r in sources if r['condition']=='gray_target_body']
    if len(gray)!=12:raise ValueError('Missing gray training member')
    for row in gray:
        check_gray(row,pool[row['member_id']])
        paths.extend(Path(pool[row['member_id']][k+'_path']) for k in ('image','label'))
    models={};exposures={}
    for key in KEYS:
        complete(key)
        cp=TRAIN/'training'/key/'completion.json';c=prior.read(cp)
        xp=Path(c['exposure_path']);x=prior.read(xp);prior.verify(x)
        counts=Counter(x['actual'])
        if sum(counts[r['member_id']] for r in gray)!=30 or any(counts[r['member_id']]==0 for r in gray):raise ValueError('Gray exposure mismatch')
        models[key]=dict(weights=c['weights'],weights_sha256=c['weights_sha256'])
        exposures[key]=dict(counts)
        paths.extend([cp,xp,Path(c['weights'])])
    for row in sources:
        if prior.file_sha256(row['image_path'])!=row['image_sha256']:raise ValueError('Changed diagnostic image')
        paths.append(Path(row['image_path']))
    import torch,ultralytics
    OUT.mkdir(exist_ok=True)
    return prior.frozen(dest,dict(status='fit_transfer_diagnostic_frozen',members=sources,models=models,
        actual_exposures=exposures,inference=q['inference'],
        environment=dict(torch=torch.__version__,ultralytics=ultralytics.__version__),
        interpretation='12 actual exposed gray images and six same-source conditions. Fit diagnosis, not unseen-scene test; no new training.',
        inputs={str(x):prior.file_sha256(x) for x in paths}))

def validate(r,key,p):
    prior.verify(r)
    if r['status']!='complete' or r['model']!=key or r['protocol_identity']!=p['identity']:raise ValueError('Wrong result identity')
    if len(r['rows'])!=len(p['members']):raise ValueError('Incomplete predictions')
    for m,row in zip(p['members'],r['rows']):
        if row['member_id']!=m['member_id'] or row['truth']!=m['truth'] or row['image_sha256']!=m['image_sha256']:
            raise ValueError('Member or full truth drift')
        if scoring(m['truth'],row['predictions'],row['low_predictions']) != {k:row[k] for k in scoring(m['truth'],row['predictions'],row['low_predictions'])}:
            raise ValueError('Scoring drift')


def infer(key,p):
    prior.verify(p); dest=OUT/(key+'.json')
    if dest.exists():
        r=prior.read(dest);validate(r,key,p);return r
    folder=OUT/key;folder.mkdir(exist_ok=True)
    if any(prior.read(f)['status']=='semantic_stop' for f in folder.glob('attempt-*/failure.json')):raise ValueError('Semantic stop')
    n=len(list(folder.glob('attempt-*')))+1
    if n>3:raise ValueError('Attempt budget exhausted')
    attempt=folder/f'attempt-{n:03}';attempt.mkdir();model=None
    try:
        import torch,ultralytics
        from ultralytics import YOLO
        if dict(torch=torch.__version__,ultralytics=ultralytics.__version__)!=p['environment']:raise ValueError('Environment drift')
        torch.set_num_threads(4); rows=[]
        with ExitStack() as stack:
            for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):
                stack.enter_context(patch.object(obj,name,forbidden))
            model=YOLO(p['models'][key]['weights'])
            if list(model.names.values())!=['transformer','switchgear','capacitor_bank','reactor']:raise ValueError('Class drift')
            with torch.inference_mode():
                for m in p['members']:
                    rows.append(dict(member_id=m['member_id'],image_sha256=m['image_sha256'],
                        **scoring(m['truth'],predict(model,m['image_path'],.37),predict(model,m['image_path'],.001))))
        paths=[OUT/'protocol.json',Path(__file__).resolve()]
        r=prior.frozen(attempt/'result.json',dict(status='complete',model=key,protocol_identity=p['identity'],rows=rows,
            optimizer_created=False,backward_executed=False,training_validation_executed=False,
            inputs={str(x):prior.file_sha256(x) for x in paths}))
        validate(r,key,p)
        result=prior.frozen(dest,dict(**{k:v for k,v in r.items() if k not in ('identity','inputs')},
            inputs={**r['inputs'],str(attempt/'result.json'):prior.file_sha256(attempt/'result.json')}))
        print('COMPLETE',key,flush=True);return result
    except BaseException as exc:
        prior.frozen(attempt/'failure.json',dict(status='semantic_stop' if isinstance(exc,ValueError) else 'technical_stop',
            error=traceback.format_exc(),child_processes_started=0))
        raise
    finally:
        del model


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');a=ap.parse_args();p=freeze()
    if a.infer:
        for key in p['models']:infer(key,p)
    else:print('PREFLIGHT_ONLY',len(p['members']))


if __name__=='__main__':main()

