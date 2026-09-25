"""New-version whole-image quarantine; preserve historical bytes and supervision."""
from collections import Counter
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT,freeze,prior


def classify(members,review,historical_held):
    bad=set(review['unresolved_content_members']);pending=set(review['limited_only_members'])
    idx={m['member_id']:m for m in members}
    if len(idx)!=len(members) or not bad|pending<=set(idx):raise ValueError('Unknown or duplicate member')
    bad_lineages={idx[m]['lineage_id'] for m in bad}
    rows=[]
    for m in members:
        mid=m['member_id'];reasons=[]
        if mid in historical_held:reasons.append('historical_hold_not_restored')
        if mid in bad:reasons.append('explicit_current_insufficient_content_whole_image')
        elif m['lineage_id'] in bad_lineages:reasons.append('registered_same_lineage_as_content_risk')
        status='quarantined' if reasons else 'pending_limited_content_eligibility' if mid in pending else 'requires_complete_quality_and_role_validation'
        rows.append(dict(member_id=mid,status=status,reasons=reasons,lineage_id=m['lineage_id'],
            image_sha256=m['image_sha256'],label_sha256=m['label_sha256'],
            historical_files_unchanged=True,labels_removed=False,training_eligible=False))
    return rows


def run():
    p=freeze();rp=OUT/'risk-reconciliation/review.json';r=prior.read(rp);prior.verify(r)
    dp=OUT.parent/'design.json';d=prior.read(dp);prior.verify(d)
    rows=classify(p['members'],r,set(d['held_members']))
    eligible_ids={m['member_id'] for m in rows if m['status']=='requires_complete_quality_and_role_validation'}
    active=set().union(*(set(v) for v in p['actual_exposures'].values()))
    residual=[m for m in p['members'] if m['member_id'] in eligible_ids&active]
    dest=OUT/'quality-isolation.json'
    if dest.exists():v=prior.read(dest);prior.verify(v);return v
    return prior.frozen(dest,dict(status='whole_image_quarantine_frozen_remaining_members_not_approved',members=rows,
        current_insufficient_members=len(r['unresolved_content_members']),
        counts=dict(Counter(x['status'] for x in rows)),
        active_remaining_candidates=len(residual),
        remaining_class_instances=dict(Counter(c for m in residual for c,n in m['class_instances'].items() for _ in range(n))),
        remaining_subsets=dict(Counter(m['subset'] for m in residual)),
        policy='Quarantine whole images and registered lineage links. No single-label removal, empty labels, ignore regions, or historical sequence modification. Remaining membership is not automatic approval.',
        lineage_limit='Member-only lineage placeholders do not establish scene independence; further source trace remains required.',
        inputs={str(x):prior.file_sha256(x) for x in [OUT/'protocol.json',rp,dp,Path(__file__).resolve()]}))


if __name__=='__main__':
    r=run();print(r['status'],r['counts'],r['active_remaining_candidates'],r['remaining_class_instances'])
