"""Preserve adverse historical decisions for explicit supersession audit."""
from collections import Counter
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT,freeze,prior

ADVERSE={'insufficient_or_uncertain','occluded_target_content_unresolved','pending_target_evidence'}


def run():
    p=freeze();ip=OUT/'review-index.json';index=prior.read(ip);prior.verify(index)
    active=set().union(*(set(v) for v in p['actual_exposures'].values()))
    members={m['member_id']:m for m in p['members']};risks=[]
    for row in index['members']:
        hits=[v for v in row['potential_review_links'] if v['status'] in ADVERSE]
        if not hits:continue
        mid=row['member_id'];m=members[mid]
        risks.append(dict(member_id=mid,currently_exposed=mid in active,
            image_sha256=m['image_sha256'],label_sha256=m['label_sha256'],lineage_id=m['lineage_id'],
            class_instances=m['class_instances'],adverse_records=hits,
            status='supersession_or_isolation_required_not_newly_certified_label_error',
            actual_exposures={k:v.get(mid,0) for k,v in p['actual_exposures'].items()}))
    dest=OUT/'risk-history.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='historical_risk_inventory_requires_target_level_resolution',members=risks,
        exposed_members=sum(r['currently_exposed'] for r in risks),
        exposed_by_subset=dict(Counter(members[r['member_id']]['subset'] for r in risks if r['currently_exposed'])),
        rules=['A newer training receipt does not supersede an adverse quality decision.',
            'Check current full-label identity and explicit later per-target evidence before declaring unresolved risk.',
            'Do not remove a label or silently discard a whole image based on this discovery index.',
            'Preserve named unresolved gaps in future training-set eligibility; fit scores cannot resolve them.'],
        inputs={str(x):prior.file_sha256(x) for x in [OUT/'protocol.json',ip,Path(__file__).resolve()]}))


if __name__=='__main__':
    r=run();print(r['status'],r['exposed_members'],r['exposed_by_subset'])
