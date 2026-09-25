"""Pool-wide closure for explicit parent links, source paths and exact pixels."""
from collections import defaultdict
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior


def run():
    paths=[OUT/'protocol.json',OUT/'quality-isolation-v4.json',OUT/'development-exact-overlap-v1.json']
    p,q,pixels=[prior.read(path) for path in paths]
    for r in (p,q,pixels):prior.verify(r)
    idx={m['member_id']:m for m in p['members']};edges=defaultdict(set);groups=defaultdict(list)
    for mid,m in idx.items():
        parent=m.get('source_member_id')
        if parent in idx and parent!=mid:edges[mid].add(parent);edges[parent].add(mid)
        groups[('registered_lineage',m['lineage_id'])].append(mid)
        for key in ['source_image','source_image_path']:
            if m.get(key):groups[('source_path',str(Path(m[key]).resolve()))].append(mid)
    for row in pixels['members']:groups[('exact_pixel',row['pixel_sha256'])].append(row['member_id'])
    relations=[]
    for (kind,value),members in groups.items():
        members=sorted(set(members))
        if len(members)<2:continue
        relations.append(dict(kind=kind,value=value,members=members))
        for mid in members:edges[mid].update(set(members)-{mid})
    original={m['member_id'] for m in q['members'] if m['status']=='quarantined'}
    closure=set(original);pending=list(original)
    while pending:
        mid=pending.pop()
        for other in edges[mid]-closure:closure.add(other);pending.append(other)
    dest=OUT/'risk-derivation-closure-v1.json'
    if dest.exists():
        r=prior.read(dest);prior.verify(r);return r
    paths.append(Path(__file__).resolve())
    return prior.frozen(dest,dict(status='additional_related_members_found' if closure-original else 'no_unisolated_explicit_risk_derivative_found',
        pool_members_checked=len(idx),relations=relations,original_quarantined=sorted(original),
        additionally_affected=sorted(closure-original),
        limits='Explicit pool metadata and exact pixels only; does not infer new pose equivalence or independent assets. Common source paths conservatively group rendered derivatives.',
        dataset_ready=False,inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__=='__main__':
    r=run();print(r['status'],len(r['relations']),len(r['additionally_affected']))
