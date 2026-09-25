"""Explicit scale stress inference; native files, labels and official scores unchanged."""
import argparse,traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from scripts.vision.material_transfer_controls import OUT,prior
from scripts.vision.material_member_fit import scoring
from scripts.vision.infer_material_member_fit import forbidden

def validate(r,p,key):
    prior.verify(r)
    if r['protocol_identity']!=p['identity'] or r['model']!=key or r['status']!='complete':raise ValueError('Wrong unit')
    expected={(x['id'],s) for x in p['rows'] for s in p['scales']}
    if len(r['rows'])!=len(expected) or {(x['id'],x['imgsz']) for x in r['rows']}!=expected:raise ValueError('Missing/duplicate scales')
    source={x['id']:x for x in p['rows']}
    for x in r['rows']:
        row=source[x['id']]
        if x['image_sha256']!=row['image_sha256']:raise ValueError('Changed image')
        if any(x[k]!=v for k,v in scoring(row['truth'],x['predictions'],x['low_predictions']).items()):raise ValueError('Changed scoring')

def run():
    p=prior.read(OUT/'protocol.json');prior.verify(p)
    import torch,ultralytics
    from ultralytics import YOLO
    torch.set_num_threads(4);(OUT/'inference').mkdir(exist_ok=True)
    for key,spec in p['models'].items():
        dest=OUT/'inference'/f'{key}.json'
        if dest.exists():validate(prior.read(dest),p,key);print('REUSE',key,flush=True);continue
        folder=OUT/'inference'/key;folder.mkdir(exist_ok=True)
        if any(prior.read(f).get('status')=='semantic_stop' for f in folder.glob('attempt-*/failure.json')):raise ValueError('Unresolved semantic failure')
        n=len(list(folder.glob('attempt-*')))+1
        if n>3:raise ValueError('Attempt limit')
        attempt=folder/f'attempt-{n:03}';attempt.mkdir();model=None
        try:
            prior.verify(p);results=[]
            if prior.file_sha256(spec['weights'])!=spec['weights_sha256']:raise ValueError('Changed weights')
            with ExitStack() as stack:
                for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):stack.enter_context(patch.object(obj,name,forbidden))
                model=YOLO(spec['weights'])
                if list(model.names.values())!=['transformer','switchgear','capacitor_bank','reactor']:raise ValueError('Changed class mapping')
                with torch.inference_mode():
                    for row in p['rows']:
                        for size in p['scales']:
                            preds=[]
                            for conf in p['thresholds']:
                                with Image.open(row['image_path']) as im:
                                    z=model.predict(im.convert('RGB'),imgsz=size,rect=False,conf=conf,iou=.7,agnostic_nms=False,max_det=300,device='cpu',verbose=False)[0]
                                preds.append([dict(bbox_xyxy=b,class_name=model.names[int(c)],confidence=q) for b,c,q in zip(z.boxes.xyxy.tolist(),z.boxes.cls.tolist(),z.boxes.conf.tolist())])
                            results.append(dict(id=row['id'],imgsz=size,image_sha256=row['image_sha256'],**scoring(row['truth'],*preds)))
            paths=[OUT/'protocol.json',Path(__file__).resolve(),Path(spec['weights']),prior.ROOT/'scripts/vision/structure_fit.py']
            r=prior.frozen(attempt/'result.json',dict(status='complete',model=key,protocol_identity=p['identity'],rows=results,
                training_started=False,optimizer_created=False,backward_executed=False,validation_run=False,
                environment=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,device='cpu',threads=4),
                inputs={str(x):prior.file_sha256(x) for x in paths}))
            validate(r,p,key);prior.frozen(dest,dict(**{k:v for k,v in r.items() if k not in ('identity','inputs')},inputs={**r['inputs'],str(attempt/'result.json'):prior.file_sha256(attempt/'result.json')}))
            print('COMPLETE',key,'52 full-frame scale records',flush=True)
        except BaseException as ex:
            prior.frozen(attempt/'failure.json',dict(status='semantic_stop' if isinstance(ex,ValueError) else 'technical_or_cancelled_stop',error=traceback.format_exc(),child_processes_started=0));raise
        finally:del model

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');a=ap.parse_args()
    if a.infer:run()
    else:prior.verify(prior.read(OUT/'protocol.json'));print('PREFLIGHT_ONLY_NO_INFERENCE')
