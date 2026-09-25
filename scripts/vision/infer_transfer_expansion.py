"""Existing-weight inference on the remaining eight frozen diagnostic sources."""
import argparse
from pathlib import Path
from scripts.vision.infer_transfer_pilot_v2 import freeze as pilot_protocol,base
from scripts.vision.import_transfer_expansion_review import main as review,DEST as REVIEW

OUT=REVIEW.parent/'inference-v1'


def freeze():
    prior=base.prior;review();pilot=pilot_protocol()
    dest=OUT/'protocol.json'
    if dest.exists():p=prior.read(dest);prior.verify(p);return p
    ep=REVIEW/'evidence.json';e=prior.read(ep);prior.verify(e)
    dp=base.DESIGN/'protocol.json';design=prior.read(dp);prior.verify(design)
    cp=base.CAND/'reviewed-completion.json';c=prior.read(cp);prior.verify(c)
    sources={s['source_pose_id']:s for s in design['sources']};members=[]
    ids=set(sources)-set(design['pilot_source_ids'])
    for page in e['pages']:
        sid=page['source_id'];condition=page['condition']
        events=[x for x in e['events'] if x['source_id']==sid and x['condition']==condition]
        if not events or sid not in ids:raise ValueError('Unexpected expansion frame')
        members.append(dict(member_id=sid+'-'+condition,source_id=sid,condition=condition,
            target=sources[sid]['target_object_id'],image_path=events[0]['image_path'],image_sha256=events[0]['image_sha256'],
            truth=[dict(x['truth'],object_id=x['object_id']) for x in events]))
    for m in c['members']:
        if m.get('source_pose_id') in ids and m['variant'] in ('warm','cool'):
            resolved=base.resolve_truth(m['full_truth'],m['instance_mapping'])
            members.append(dict(member_id=m['member_id'],source_id=m['source_pose_id'],condition=m['variant'],
                target=m['planned_object_id'],image_path=m['image_path'],image_sha256=m['image_sha256'],
                truth=[dict(t,object_id=r['object_id']) for t,r in zip(m['full_truth']['objects'],resolved,strict=True)]))
    if len(members)!=56 or len({m['member_id'] for m in members})!=56:raise ValueError('Incomplete membership')
    for m in members:
        if prior.file_sha256(m['image_path'])!=m['image_sha256']:raise ValueError('Stale image')
        if any(not t.get('class_name') or not t.get('object_id') for t in m['truth']):raise ValueError('Incomplete truth')
        base.scoring(m['truth'],[],[])
    paths=[ep,dp,cp,REVIEW/'label-review.json',Path(__file__).resolve()]+[Path(m['image_path']) for m in members]
    OUT.mkdir(exist_ok=True)
    return prior.frozen(dest,dict(status='diagnostic_expansion_inference_frozen',members=members,
        models=pilot['models'],inference=pilot['inference'],environment=pilot['environment'],
        training_started=False,training_admitted=False,promotable=False,
        inputs={**pilot['inputs'],**{str(x):prior.file_sha256(x) for x in paths}}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--infer',action='store_true');a=ap.parse_args();p=freeze();base.OUT=OUT
    if a.infer:
        for key in p['models']:base.infer(key,p)
    else:print('PREFLIGHT_ONLY',len(p['members']))
