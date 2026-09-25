"""Single closeout inventory: inherit only explicit quality decisions, never infer them."""
from collections import Counter, defaultdict
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior


def inventory():
    names=['protocol','quality-isolation-v4','cohort-review-links','early-positive-review-links',
           'negative-review-links','low-light-review-links','authored-quality-dispositions-01','authored-quality-dispositions-02']
    paths=[OUT/(name+'.json') for name in names];data=[prior.read(p) for p in paths]
    for r in data:prior.verify(r)
    pool,quarantine,cohort,early,negative,low,first,second=data
    supported=defaultdict(list)
    for i,record,allowed in [(2,cohort,{'existing_explicit_bounded_review_validated'}),
                             (3,early,{'existing_bounded_review_validated'}),
                             (4,negative,{'existing_negative_review_and_empty_full_labels_verified'}),
                             (5,low,{'existing_low_light_explicit_review_validated'})]:
        for m in record['members']:
            if m['status'] in allowed:supported[m['member_id']].append(str(paths[i]))
    for i,record in [(6,first),(7,second)]:
        for d in record['decisions']:
            if d['quality_status']=='supported_for_bounded_research_with_recorded_limits':
                supported[d['member_id']].append(str(paths[i]))
    q={m['member_id']:m for m in quarantine['members']};rows=[]
    for m in pool['members']:
        mid=m['member_id']
        stage='quarantined' if q[mid]['status']=='quarantined' else ('explicit_quality_evidence_present_role_gate_pending' if supported[mid] else 'final_quality_disposition_required')
        rows.append(dict(member_id=mid,subset=m['subset'],variant=m['variant'],stage=stage,
            class_instances=m['class_instances'],quality_decision_sources=supported[mid],
            preserved_quarantine_reasons=q[mid]['reasons'],training_eligible=False))
    dest=OUT/'closeout-inventory-v1.json'
    if dest.exists():
        r=prior.read(dest);prior.verify(r);return r
    paths.extend([Path(__file__).resolve(),prior.ROOT/'docs/plans/order_fit_dataset_closeout.md'])
    return prior.frozen(dest,dict(status='closeout_inventory_not_dataset_ready',members=rows,
        counts=dict(Counter(m['stage'] for m in rows)),
        inputs={str(p):prior.file_sha256(p) for p in paths}))


if __name__=='__main__':print(inventory()['counts'])
