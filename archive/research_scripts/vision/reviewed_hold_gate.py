"""Fail-closed quality and count validation shared by continuation/finalization."""
from collections import Counter
from pathlib import Path
from scripts.vision.reviewed_hold_control import OUT as QUALITY,prior
from scripts.vision.train_reviewed_hold import OUT,KEYS,REFERENCE


def validate_review(e,q):
    expected={(f['member_id'],l['runtime_label']):(f,l) for f in e['events'] for l in f['labels']}
    ds=q['decisions'];keys=[(d['member_id'],d['runtime_label']) for d in ds]
    if len(keys)!=len(set(keys)) or set(keys)!=set(expected):raise ValueError('Missing or duplicate review')
    for d in ds:
        f,l=expected[(d['member_id'],d['runtime_label'])]
        if d['evidence_identity']!=e['identity'] or d['page_sha256']!=f['page_sha256'] or d['label']!=l:raise ValueError('Stale review')
        if d['status']!='identifiable_content_with_recorded_limits' or not d['reason'] or d['review_nature']!='AI-assisted' or not d['reviewed_at']:raise ValueError('Unresolved quality decision')
    if q['new_risk_members']:raise ValueError('Risk requires bounded resolve, not training')


def validate_sequences(p,reference):
    idx={r['member_id']:r for r in p['pool_rows']}
    for key in KEYS:
        new=p['schedules'][key];old=reference['schedules'][key]
        if len(new)!=2700 or len(old)!=2700:raise ValueError('Exposure count mismatch')
        if p['brightness_factors'][key]!=reference['brightness_factors'][key]:raise ValueError('Brightness positions changed')
        if set(new)&set(p['held_members']):raise ValueError('Held frame exposed')
        for a,b in zip(old,new):
            if idx[a]['subset']!=idx[b]['subset'] or (idx[a]['subset']=='hard_negative' and a!=b):raise ValueError('Subset/negative position changed')
        for cls in p['names'].values() if isinstance(p['names'],dict) else p['names']:
            if sum(idx[m]['class_instances'].get(cls,0) for m in old)!=sum(idx[m]['class_instances'].get(cls,0) for m in new):raise ValueError('Class exposure changed')


def check():
    ep=QUALITY/'review-evidence.json';qp=QUALITY/'quality-review.json';pp=OUT/'protocol.json';rp=REFERENCE/'protocol.json'
    for x in (ep,qp,pp,rp):prior.verify(prior.read(x))
    validate_review(prior.read(ep),prior.read(qp));p=prior.read(pp);validate_sequences(p,prior.read(rp))
    return p


if __name__=='__main__':check();print('QUALITY_AND_SEQUENCE_GATES_PASSED')
