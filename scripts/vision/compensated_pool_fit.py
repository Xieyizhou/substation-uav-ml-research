"""Explicit existing-weight native-image fit diagnosis; every entry forbids training."""
import argparse
import traceback
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from scripts.vision import train_compensated_material as trained
from scripts.vision import train_frozen_multiscale as reference
from scripts.vision.build_compensated_error_review import DEST, prior
from scripts.vision.record_compensated_error_review import validate as validate_review
from scripts.vision.structure_fit import truth_for, scoring
from scripts.vision.infer_material_member_fit import forbidden, validate_unit
from scripts.vision.evaluate_exposure_diagnosis import predict

OUT = DEST/'pool-fit-v1'
KEYS = tuple(f'{a}-{s}' for s in (7,17,27) for a in ('R','V','VM'))


def check(p):
    prior.verify(p)
    e = prior.read(DEST/'evidence.json'); r = prior.read(DEST/'review.json')
    prior.verify(e); prior.verify(r); validate_review(e,r['decisions'])
    ids = {m['member_id'] for m in p['members']}
    if len(ids) != len(p['members']) or len(ids) != 277:
        raise ValueError('Missing or duplicate pool member')
    for m in p['members']:
        if truth_for(m) != m['truth']:
            raise ValueError('Complete label changed')
    for spec in p['models'].values():
        if set(spec['counts'])-ids or sum(spec['counts'].values()) != 2700:
            raise ValueError('Actual exposure outside pool or incomplete')


def freeze():
    dest = OUT/'protocol.json'
    if dest.exists():
        p = prior.read(dest); check(p); return p
    p,_,_ = trained.contract('V-7')
    paths = [trained.OUT/'protocol.json', trained.OUT.parent/'dataset-completion.json',
             DEST/'evidence.json', DEST/'review.json', Path(__file__).resolve()]
    members = []
    for row in p['pool_rows']:
        m = dict(row)
        for f in ('image','label'):
            m[f+'_path'] = str(Path(m[f+'_path']).resolve())
            if prior.file_sha256(m[f+'_path']) != m[f+'_sha256']:
                raise ValueError('Changed pool bytes')
            paths.append(Path(m[f+'_path']))
        m['truth'] = truth_for(m)
        if Counter(t['class_name'] for t in m['truth']) != Counter(m['class_instances']):
            raise ValueError('Incomplete class supervision')
        m['new_compensated_member'] = 'full_truth' in m
        members.append(m)
    models = {}
    for key in KEYS:
        arm,seed = key.split('-')
        mod,k = (reference,f'fixed-{seed}') if arm=='R' else (trained,key)
        mod.complete(k)
        cp = mod.OUT/'training'/k/'completion.json'; c = prior.read(cp)
        xp = Path(c['exposure_path']); x = prior.read(xp); prior.verify(x)
        models[key] = dict(weights=c['weights'],weights_sha256=c['weights_sha256'],counts=dict(Counter(x['actual'])))
        paths += [cp,xp,Path(c['weights'])]
    import torch, ultralytics
    paths += [prior.ROOT/'scripts/vision'/f for f in (
        'structure_fit.py','exposure_metrics.py','evaluate_paired_visual_factors.py',
        'evaluate_exposure_diagnosis.py','infer_material_member_fit.py','record_compensated_error_review.py')]
    OUT.mkdir(exist_ok=True)
    result = prior.frozen(dest,dict(status='frozen_inference_only',members=members,models=models,
        training_allowed=False, training_started=False,
        inference_protocol=dict(device='cpu',imgsz=640,confidences=[.37,.001],iou=.7,
                                agnostic_nms=False,max_det=300,matching_iou=.5),
        environment=dict(torch=torch.__version__,ultralytics=ultralytics.__version__),
        interpretation='Native unaugmented pool images; only actually exposed members count as training fit. Zero-exposure members separately reported.',
        review_scope='New37 full-label review reused; historical whole pool not re-admitted. Development content gaps retained.',
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    check(result); return result


def infer(key,p):
    check(p); spec=p['models'][key]; dest=OUT/'inference'/f'{key}.json'
    if dest.exists():
        r=prior.read(dest);validate_unit(r,key,p);return r
    folder=OUT/'inference'/key;folder.mkdir(parents=True,exist_ok=True)
    if any(prior.read(f)['status']=='semantic_stop' for f in folder.glob('attempt-*/failure.json')):
        raise ValueError('Prior semantic failure requires independent correction')
    n=len(list(folder.glob('attempt-*')))+1
    if n>3:raise ValueError('Technical attempt budget exhausted')
    attempt=folder/f'attempt-{n:03}';attempt.mkdir();model=None
    try:
        import torch,ultralytics
        from ultralytics import YOLO
        if dict(torch=torch.__version__,ultralytics=ultralytics.__version__)!=p['environment']:
            raise ValueError('Environment drift')
        torch.set_num_threads(4);rows=[]
        with ExitStack() as stack:
            for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):
                stack.enter_context(patch.object(obj,name,forbidden))
            model=YOLO(spec['weights'])
            if list(model.names.values())!=['transformer','switchgear','capacitor_bank','reactor']:
                raise ValueError('Class order mismatch')
            with torch.inference_mode():
                for i,m in enumerate(p['members']):
                    rows.append(dict(member_id=m['member_id'],image_sha256=m['image_sha256'],
                        actual_exposures=spec['counts'].get(m['member_id'],0),
                        **scoring(m['truth'],predict(model,m['image_path'],.37),predict(model,m['image_path'],.001))))
                    if (i+1)%50==0:print(key,i+1,'/',len(p['members']),flush=True)
        paths=[OUT/'protocol.json',Path(__file__).resolve(),Path(spec['weights'])]
        r=prior.frozen(attempt/'result.json',dict(status='complete',model=key,protocol_identity=p['identity'],rows=rows,
            optimizer_created=False,backward_executed=False,training_validation_executed=False,
            inputs={str(x):prior.file_sha256(x) for x in paths}))
        validate_unit(r,key,p)
        r=prior.frozen(dest,dict(**{k:v for k,v in r.items() if k not in ('identity','inputs')},
            inputs={**r['inputs'],str(attempt/'result.json'):prior.file_sha256(attempt/'result.json')}))
        print('COMPLETE',key,flush=True);return r
    except BaseException as exc:
        prior.frozen(attempt/'failure.json',dict(status='semantic_stop' if isinstance(exc,ValueError) else 'technical_or_cancelled_stop',
            error=traceback.format_exc(),child_processes_started=0))
        raise
    finally:
        del model


def finish(p):
    check(p);members={m['member_id']:m for m in p['members']}
    common=set.intersection(*[{m for m,c in spec['counts'].items() if c>0} for spec in p['models'].values()])
    paths=[OUT/'protocol.json',Path(__file__).resolve()];groups={};instances=[]
    for key in KEYS:
        path=OUT/'inference'/f'{key}.json';r=prior.read(path);validate_unit(r,key,p);paths.append(path)
        g=defaultdict(Counter)
        for row in r['rows']:
            m=members[row['member_id']];exposed=row['actual_exposures']>0
            groups_for_row=['exposed' if exposed else 'unexposed']
            if m['member_id'] in common:groups_for_row.append('common_exposed')
            if m['new_compensated_member']:groups_for_row.append('new37_exposed' if exposed else 'new37_unexposed')
            hits={x['truth_index'] for x in row['matches']}
            for grp in groups_for_row:
                g[grp,'all']['images']+=1
                g[grp,'all']['unmatched_predictions']+=row['unmatched_prediction_count']
                if not row['truth']:
                    g[grp,'all']['negative_images']+=1
                    g[grp,'all']['negative_fp_images']+=bool(row['predictions'])
            for i,t in enumerate(row['truth']):
                miss=next((x for x in row['misses'] if x['truth_index']==i),None)
                instances.append(dict(model=key,member_id=m['member_id'],truth=t,hit=i in hits,miss=miss,
                    actual_exposures=row['actual_exposures'],variant=m.get('variant','unknown'),
                    lineage_id=m['lineage_id'],new_compensated_member=m['new_compensated_member']))
                for grp in groups_for_row:
                    for cls in ('all',t['class_name']):
                        g[grp,cls]['truth']+=1;g[grp,cls]['hit']+=i in hits
                        if miss:g[grp,cls][miss['reason']]+=1
        groups[key]=[dict(group=grp,category=cls,**counts) for (grp,cls),counts in g.items()]
    return prior.frozen(OUT/'summary.json',dict(status='native_pool_fit_complete',groups=groups,instances=instances,
        common_exposed_members=sorted(common),training_fit_not_generalization=True,
        unknown_labels_not_removed=True,selected_candidate=None,
        limits=['Native RGB only, not every actual brightness tensor','Historical supervision risks not re-certified',
                'Same layout/assets; material variants are not independent scenes','No optimizer or new training'],
        inputs={str(x):prior.file_sha256(x) for x in paths}))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');a=ap.parse_args();p=freeze()
    if a.infer:
        for key in KEYS:infer(key,p)
        print(finish(p)['status'],flush=True)
    else:print('PREFLIGHT_ONLY_NO_INFERENCE_NO_TRAINING')


if __name__=='__main__':main()
