"""Append explicit newly reviewed content risks to an independent quarantine."""
from collections import Counter
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior
from scripts.vision.record_order_fit_remaining_review import validate


def extend(members, previous, decisions):
    idx={m['member_id']:m for m in members}
    old={m['member_id']:m for m in previous}
    if len(idx)!=len(members) or len(old)!=len(previous) or set(idx)!=set(old):
        raise ValueError('Isolation member coverage changed')
    if len({d['member_id'] for d in decisions})!=len(decisions):
        raise ValueError('Duplicate reviewed member')
    if any(d['member_id'] not in idx for d in decisions):
        raise ValueError('Unknown reviewed member')
    bad={d['member_id'] for d in decisions if any(x['status']=='insufficient_content' for x in d['label_observations'])}
    lineages={idx[mid]['lineage_id'] for mid in bad}
    rows=[]
    for mid,m in idx.items():
        r=dict(old[mid]);r['reasons']=list(r['reasons'])
        if mid in bad:
            r['reasons'].append('remaining_review_explicit_insufficient_content')
            r['status']='quarantined'
        elif m['lineage_id'] in lineages:
            r['reasons'].append('registered_same_lineage_as_remaining_review_content_risk')
            r['status']='quarantined'
        r['training_eligible']=False
        rows.append(r)
    return rows


def run():
    paths=[OUT/'protocol.json',OUT/'quality-isolation.json',OUT/'remaining-review/evidence.json',
           OUT/'remaining-review/review.json',Path(__file__).resolve()]
    p,old,e,r=[prior.read(x) for x in paths[:4]]
    for record in (p,old,e,r):prior.verify(record)
    validate(e,r['decisions'])
    rows=extend(p['members'],old['members'],r['decisions'])
    dest=OUT/'quality-isolation-v2.json'
    if dest.exists():
        result=prior.read(dest);prior.verify(result);return result
    active=set().union(*(set(v) for v in p['actual_exposures'].values()))
    residual={x['member_id'] for x in rows if x['status']=='requires_complete_quality_and_role_validation'}&active
    return prior.frozen(dest,dict(status='expanded_whole_image_quarantine_not_dataset_admission',members=rows,
        counts=dict(Counter(x['status'] for x in rows)),newly_identified_content_risk_members=r['insufficient_content_members'],
        active_remaining_candidates=len(residual),
        remaining_class_instances=dict(Counter(c for m in p['members'] if m['member_id'] in residual for c,n in m['class_instances'].items() for _ in range(n))),
        policy='Keep all historical bytes and labels; quarantine whole frames and registered same-lineage members. Limited views are recorded, not automatically rejected or approved. Source/full-frame/role gates remain required.',
        inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':
    r=run();print(r['counts'],r['active_remaining_candidates'],r['remaining_class_instances'])
