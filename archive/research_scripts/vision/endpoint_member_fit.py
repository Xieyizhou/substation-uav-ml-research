"""Existing nine weights on an already reviewed four-pose training subset."""
import argparse
from collections import Counter,defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from scripts.vision import material_member_fit as reviewed
from scripts.vision.review_material_member_fit import validate as review_validate
from scripts.vision.infer_material_member_fit import validate_unit,forbidden
from scripts.vision.structure_fit import truth_for,scoring
from scripts.vision.evaluate_exposure_diagnosis import predict
from scripts.vision import scale_endpoint_control as control
from scripts.vision import train_frozen_multiscale as fixed

prior=control.prior
OUT=control.OUT/'member-fit-v1'
KEYS=tuple(f'{arm}-{seed}' for seed in (7,17,27) for arm in ('fixed','small','large'))

def reviewed_input():
    p=reviewed.preflight();r=prior.read(reviewed.OUT/'review.json');review_validate(p,r)
    return p,r

def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():p=prior.read(dest);prior.verify(p);reviewed_input();return p
    source,rev=reviewed_input();models={};paths=[reviewed.OUT/'protocol.json',reviewed.OUT/'review.json',Path(__file__).resolve()]
    _,train,_=fixed.contract('fixed-7')
    for key in KEYS:
        if key.startswith('fixed-'):
            fixed.complete(key);cp=fixed.OUT/'training'/key/'completion.json';c=prior.read(cp)
        else:
            c=control.complete(key);cp=control.OUT/'training'/key/'completion.json'
        xp=Path(c['exposure_path']);x=prior.read(xp);prior.verify(x)
        counts=dict(Counter(x['actual']))
        models[key]=dict(weights=c['weights'],weights_sha256=c['weights_sha256'],counts=counts,role='actual_exposure_training_fit_only')
        paths += [cp,xp,Path(c['weights'])]
        for m in source['members']:
            if counts.get(m['member_id'],0)<=0:raise ValueError('Selected member not actually exposed')
    for seed in (7,17,27):
        if not models[f'fixed-{seed}']['counts']==models[f'small-{seed}']['counts']==models[f'large-{seed}']['counts']:raise ValueError('Unequal exposure')
    for m in source['members']:
        if truth_for(m)!=m['truth']:raise ValueError('Changed labels')
        paths += [Path(m['image_path']),Path(m['label_path'])]
    census_path=reviewed.PRIOR/'coverage-census.json';census=prior.read(census_path);prior.verify(census);paths.append(census_path)
    ids={r['member_id'] for r in train['pool_rows']};census_rows=[r for r in census['members'] if r['member_id'] in ids]
    sources=Counter(r['source_variant'] for r in census_rows if r['instances'])
    paths += [prior.ROOT/'scripts/vision'/n for n in ('structure_fit.py','exposure_metrics.py','evaluate_exposure_diagnosis.py','infer_material_member_fit.py')]
    OUT.mkdir(exist_ok=True)
    return prior.frozen(dest,dict(status='frozen_diagnostic_only',members=source['members'],events=source['events'],models=models,
        inference_protocol=source['inference_protocol'],independent_pose_groups=4,training_allowed=False,
        review_reuse='Existing explicit 42-label content review, original protocol identity checked; no new approval generated',
        selected_source_variants=dict(Counter(m['source_variant'] for m in source['members'])),
        current_pool_positive_source_variants=dict(sources),native_unaugmented_only=True,
        environment=train['environment'],inputs={str(p):prior.file_sha256(p) for p in paths}))

def check(p):
    prior.verify(p);reviewed_input()
    for m in p['members']:
        if truth_for(m)!=m['truth']:raise ValueError('Changed truth')

def infer(key,p):
    check(p);spec=p['models'][key];dest=OUT/'inference'/f'{key}.json'
    if dest.exists():r=prior.read(dest);validate_unit(r,key,p);return r
    attempt=control.attempt_folder(OUT/'inference'/key)
    try:
        import torch,ultralytics
        from ultralytics import YOLO
        torch.set_num_threads(4)
        if dict(torch=torch.__version__,ultralytics=ultralytics.__version__)!=p['environment']:raise ValueError('Environment drift')
        with ExitStack() as stack:
            for obj,name in ((torch.optim.Optimizer,'__init__'),(torch.Tensor,'backward'),(torch.autograd,'backward'),(YOLO,'train'),(YOLO,'val')):stack.enter_context(patch.object(obj,name,forbidden))
            model=YOLO(spec['weights']);rows=[]
            if list(model.names.values())!=['transformer','switchgear','capacitor_bank','reactor']:raise ValueError('Class identity mismatch')
            with torch.inference_mode():
                for m in p['members']:
                    rows.append(dict(member_id=m['member_id'],image_sha256=m['image_sha256'],actual_exposures=spec['counts'][m['member_id']],
                        **scoring(m['truth'],predict(model,m['image_path'],.37),predict(model,m['image_path'],.001))))
        paths=[OUT/'protocol.json',Path(__file__).resolve(),Path(spec['weights'])]
        r=prior.frozen(dest,dict(status='complete',model=key,protocol_identity=p['identity'],rows=rows,
            optimizer_created=False,backward_executed=False,training_validation_executed=False,
            inputs={str(x):prior.file_sha256(x) for x in paths}))
        validate_unit(r,key,p);return r
    except BaseException as ex:control.failure(attempt,ex);raise

def finish(p):
    paths=[OUT/'protocol.json',Path(__file__).resolve()];groups={};detail=[]
    _,review=reviewed_input();decisions={d['event_id']:d for d in review['decisions']}
    events={(x['member_id'],x['truth']['annotation_id']):x for x in p['events']}
    members={m['member_id']:m for m in p['members']}
    for key in KEYS:
        path=OUT/'inference'/f'{key}.json';r=prior.read(path);validate_unit(r,key,p);paths.append(path);g=defaultdict(Counter)
        for x in r['rows']:
            m=members[x['member_id']];hits={z['truth_index'] for z in x['matches']}
            for i,t in enumerate(x['truth']):
                e=events[(x['member_id'],t['annotation_id'])];d=decisions[e['event_id']]
                miss=next((z for z in x['misses'] if z['truth_index']==i),None)
                detail.append(dict(model=key,member_id=m['member_id'],event_id=e['event_id'],object_id=e['object_id'],lineage_id=m['lineage_id'],
                    source_variant=m['source_variant'],content=d['content'],actual_exposures=x['actual_exposures'],truth=t,hit=i in hits,miss=miss))
                for v in ('all',m['source_variant']):
                    for c in ('all',t['class_name']):
                        g[v,c]['truth']+=1;g[v,c]['hit']+=i in hits
                        if miss:g[v,c][miss['reason']]+=1
        groups[key]=[dict(variant=v,category=c,**counts) for (v,c),counts in g.items()]
    dev=control.OUT/'evaluation/summary.json';r=prior.read(dev);prior.verify(r);paths.append(dev)
    result=prior.frozen(OUT/'summary.json',dict(status='native_training_fit_complete_lighting_tensor_evidence_not_measured',groups=groups,instance_results=detail,
        development_aggregate={a:r['aggregate'][a] for a in ('fixed','small','large')},selected_candidate=None,
        independent_pose_groups=4,training_fit_not_generalization=True,
        limits=['12 selected images/42 labels, not whole-pool fit','Native images only; no certification of every actual brightness tensor','No independent positive lighting variant in selected subset','Same assets/layout; no unseen-scene claim'],
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    for k,gs in groups.items():print(k,[x for x in gs if x['category']=='all'],flush=True)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');a=ap.parse_args();p=freeze();check(p)
    if a.infer:
        for k in KEYS:infer(k,p);print('INFERRED',k,flush=True)
        finish(p)
    else:print('PREFLIGHT_ONLY_NO_INFERENCE_NO_TRAINING')

if __name__=='__main__':main()
