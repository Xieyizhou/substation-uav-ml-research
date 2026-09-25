"""Explicit inference only, seven existing weights, in-process bounded attempts."""
import argparse
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import traceback
from scripts.vision.material_member_fit import OUT,preflight,prior,scoring
from scripts.vision.review_material_member_fit import validate
from scripts.vision.evaluate_exposure_diagnosis import predict

def forbidden(*a,**k):raise RuntimeError('Training, optimizer, backward and training validation forbidden')

def validate_unit(r,key,p):
    prior.verify(r)
    if r['status']!='complete' or r['protocol_identity']!=p['identity'] or r['model']!=key:raise ValueError('Wrong unit identity')
    index={x['member_id']:x for x in r['rows']}
    if len(index)!=len(r['rows']) or set(index)!={x['member_id'] for x in p['members']}:raise ValueError('Missing/duplicate inference member')
    for m in p['members']:
        x=index[m['member_id']]
        if x['image_sha256']!=m['image_sha256'] or x['actual_exposures']!=p['models'][key]['counts'].get(m['member_id'],0):raise ValueError('Changed image/exposure')
        expected=scoring(m['truth'],x['predictions'],x['low_predictions'])
        if any(x[k]!=v for k,v in expected.items()):raise ValueError('Changed truth/matching')
    if any(r.get(k) is not False for k in ('optimizer_created','backward_executed','training_validation_executed')):raise ValueError('Forbidden runtime operation')

def run():
    p=preflight();rp=OUT/'review.json';validate(p,prior.read(rp))
    import torch,ultralytics
    from ultralytics import YOLO
    torch.set_num_threads(4);(OUT/'inference').mkdir(exist_ok=True)
    for key,spec in p['models'].items():
        dest=OUT/'inference'/f'{key}.json'
        if dest.exists():validate_unit(prior.read(dest),key,p);print('REUSE',key,flush=True);continue
        folder=OUT/'inference'/key;folder.mkdir(exist_ok=True);number=len(list(folder.glob('attempt-*')))+1
        if number>3:raise ValueError('Technical attempt budget exhausted')
        attempt=folder/f'attempt-{number:03}';attempt.mkdir();model=None
        try:
            preflight();validate(p,prior.read(rp));results=[]
            if prior.file_sha256(spec['weights'])!=spec['weights_sha256']:raise ValueError('Changed weight')
            with ExitStack() as stack:
                for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):
                    stack.enter_context(patch.object(obj,name,forbidden))
                model=YOLO(spec['weights'])
                if list(model.names.values())!=['transformer','switchgear','capacitor_bank','reactor']:raise ValueError('Class mapping changed')
                with torch.inference_mode():
                    for m in p['members']:
                        results.append(dict(member_id=m['member_id'],image_sha256=m['image_sha256'],actual_exposures=spec['counts'].get(m['member_id'],0),
                            **scoring(m['truth'],predict(model,m['image_path'],.37),predict(model,m['image_path'],.001))))
            paths=[OUT/'protocol.json',rp,Path(__file__).resolve(),Path(spec['weights'])]
            r=prior.frozen(attempt/'result.json',dict(status='complete',model=key,protocol_identity=p['identity'],rows=results,
                optimizer_created=False,backward_executed=False,training_validation_executed=False,
                environment=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,device='cpu',threads=4),
                inputs={str(x):prior.file_sha256(x) for x in paths}))
            validate_unit(r,key,p)
            prior.frozen(dest,dict(**{k:v for k,v in r.items() if k not in ('identity','inputs')},inputs={**r['inputs'],str(attempt/'result.json'):prior.file_sha256(attempt/'result.json')}))
            print('COMPLETE',key,len(results),'frames',flush=True)
        except BaseException as exc:
            prior.frozen(attempt/'failure.json',dict(status='semantic_stop' if isinstance(exc,ValueError) else 'technical_or_cancelled_stop',error=traceback.format_exc(),child_processes_started=0))
            raise
        finally:
            del model

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');a=ap.parse_args()
    if a.infer:run()
    else:preflight();print('PREFLIGHT_ONLY_NO_INFERENCE_NO_TRAINING')
