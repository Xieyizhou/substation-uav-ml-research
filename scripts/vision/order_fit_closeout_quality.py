"""Batch closeout quality evidence; no inference from file presence or fit scores."""
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior

# Explicit reviewer selection after reading all source/variant label reasons.
SELECTED_POSES=('G01','G02','G03','G04','S01','S02','S03','S04','S05','S06','S07','S08')
SELECTED_VARIANTS=('original','warm','cool')


def reuse_source_variant_quality():
    paths=[OUT/'protocol.json',OUT/'early-positive-review-links.json',OUT/'quality-isolation-v4.json']
    p,links,q=[prior.read(path) for path in paths]
    for record in (p,links,q):prior.verify(record)
    pool={m['member_id']:m for m in p['members']};byid={m['member_id']:m for m in links['members']}
    held={m['member_id'] for m in q['members'] if m['status']=='quarantined'}
    rows=[]
    for pose in SELECTED_POSES:
        for variant in SELECTED_VARIANTS:
            mid=pose+'-'+variant;m=pool[mid];link=byid[mid]
            if mid in held:raise ValueError('Cannot restore held member')
            decisions=link['decisions']
            if len(decisions)!=len(m['truth']):raise ValueError('Incomplete full-label quality review')
            wanted='source_content_review_passed' if variant=='original' else 'variant_content_review_passed'
            objects=set()
            for item in decisions:
                d=item['decision'];rp=Path(item['review_path']);r=prior.read(rp);prior.verify(r);paths.append(rp)
                if d not in r['decisions']:raise ValueError('Decision absent from signed source')
                if d.get('status',d.get('decision'))!=wanted or not d.get('reason'):raise ValueError('Not a quality pass')
                if item['object_id'] in objects:raise ValueError('Duplicate instance review')
                objects.add(item['object_id'])
            for key in ('image','label'):
                path=Path(m[key+'_path'])
                if prior.file_sha256(path)!=m[key+'_sha256']:raise ValueError('Current member drift')
                paths.append(path)
            rows.append(dict(member_id=mid,quality_status='existing_explicit_quality_pass_reused_with_limits',
                source_decisions=decisions,image_sha256=m['image_sha256'],label_sha256=m['label_sha256'],
                training_eligible=False,remaining_gates=['full_scene_source_role_confirmation','export_and_loader_preflight']))
    paths.append(Path(__file__).resolve())
    dest=OUT/'closeout-source-variant-quality.json'
    if dest.exists():
        r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='36_existing_quality_decisions_reused_not_dataset_admission',members=rows,
        interpretation='No new visual approval manufactured. Original/warm/cool explicit passes retained; gray diagnostic reviews excluded from this reuse.',
        inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__=='__main__':print(reuse_source_variant_quality()['status'])
