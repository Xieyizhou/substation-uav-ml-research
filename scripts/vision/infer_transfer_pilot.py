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

OUT=REVIEW.parent/'inference-v1'


def freeze():
    review()
    dest=OUT/'protocol.json'
    if dest.exists():
        p=prior.read(dest); prior.verify(p); return p
    pp=DESIGN/'protocol.json'; p=prior.read(pp); prior.verify(p)
    ep=REVIEW/'evidence.json'; e=prior.read(ep); prior.verify(e)
    cp=CAND/'reviewed-completion.json'; c=prior.read(cp); prior.verify(c)
    sources={s['source_pose_id']:s for s in p['sources']}; rows=[]
    paths=[pp,ep,cp,REVIEW/'label-review.json',Path(__file__).resolve()]
    for page in e['pages']:
        events=[x for x in e['events'] if x['source_id']==page['source_id'] and x['condition']==page['condition']]
        source=sources[page['source_id']]
        rows.append(dict(member_id=page['source_id']+'-'+page['condition'],source_id=page['source_id'],
            condition=page['condition'],target=source['target_object_id'],image_path=events[0]['image_path'],
            image_sha256=events[0]['image_sha256'],truth=[dict(x['truth'],object_id=x['object_id']) for x in events]))
    for m in c['members']:
        if m.get('source_pose_id') in p['pilot_source_ids'] and m['variant'] in ('warm','cool'):
            rows.append(dict(member_id=m['member_id'],source_id=m['source_pose_id'],condition=m['variant'],
                target=m['planned_object_id'],image_path=m['image_path'],image_sha256=m['image_sha256'],
                truth=resolve_truth(m['full_truth'],m['instance_mapping'])))
    if len(rows)!=28 or len({m['member_id'] for m in rows})!=28: raise ValueError('Pilot membership incomplete')
    for m in rows:
        if prior.file_sha256(m['image_path'])!=m['image_sha256']:raise ValueError('Changed image')
        paths.append(Path(m['image_path']))
    for s in p['models'].values():paths.append(Path(s['weights']))
    import torch,ultralytics
    paths += [prior.ROOT/'scripts/vision'/f for f in ('structure_fit.py','evaluate_exposure_diagnosis.py','exposure_metrics.py','evaluate_paired_visual_factors.py')]
    OUT.mkdir(exist_ok=True)
    return prior.frozen(dest,dict(status='diagnostic_inference_frozen',members=rows,models=p['models'],
        inference=p['inference'],environment=dict(torch=torch.__version__,ultralytics=ultralytics.__version__),
        training_admitted=False,promotable=False,training_started=False,
        interpretation='Four same-source poses; variants and seeds do not increase independent sample count. Not training admission.',
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
