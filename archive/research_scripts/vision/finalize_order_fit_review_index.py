"""Complete evidence links while retaining every unresolved admission gate."""
from collections import Counter
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior


def run():
    paths=[OUT/'review-coverage-matrix-v2.json',OUT/'low-light-review-links.json',OUT/'remaining-variant-review-links.json',OUT/'gray-full-label-correspondence.json']
    old,low,variants,gray=[prior.read(p) for p in paths]
    for r in (old,low,variants,gray):prior.verify(r)
    added={}
    for name,record in [('low-light-review-links',low),('remaining-variant-review-links',variants)]:
        for m in record['members']:
            if m['member_id'] in added:raise ValueError('Duplicate new member link')
            added[m['member_id']]=(name,m['status'])
    rows=[]
    for source in old['members']:
        row=dict(source);row['review_sources']=list(row['review_sources'])
        if row['member_id'] in added:
            name,status=added.pop(row['member_id'])
            row['review_sources'].append(name);row['explicit_review_link_found']=True
            row['linked_review_status']=status
            if row['next_gate']=='missing_explicit_review_link':row['next_gate']='review_source_full_frame_and_role_disposition'
        rows.append(row)
    if added:raise ValueError('Unknown linked members')
    if any(not m['explicit_review_link_found'] and m['quarantine_status']!='quarantined' for m in rows):
        raise ValueError('Unresolved review-link coverage')
    dest=OUT/'review-coverage-matrix-v3.json'
    if dest.exists():
        r=prior.read(dest);prior.verify(r);return r
    paths.append(Path(__file__).resolve())
    return prior.frozen(dest,dict(status='nonquarantined_review_links_complete_eligibility_pending',members=rows,
        counts=dict(Counter(r['next_gate'] for r in rows)),dataset_ready=False,
        scope='Diagnostic review links are not quality approvals; all final eligibility flags remain false.',
        inputs={str(p):prior.file_sha256(p) for p in paths}))


if __name__=='__main__':print(run()['counts'])
