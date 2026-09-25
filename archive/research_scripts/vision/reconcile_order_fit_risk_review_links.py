"""Add omitted risk-review links, preserving all pending and hold decisions."""
from collections import Counter, defaultdict
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior


def run():
    paths = [OUT/'protocol.json', OUT/'review-coverage-matrix-v1.json', OUT/'risk-reconciliation/review.json']
    p, old, review = [prior.read(path) for path in paths]
    for record in (p, old, review): prior.verify(record)
    members = {m['member_id']: m for m in p['members']}; grouped = defaultdict(list)
    for d in review['decisions']: grouped[d['member_id']].append(d)
    for mid, decisions in grouped.items():
        m = members[mid]
        if sorted(d['label_index'] for d in decisions) != list(range(len(m['truth']))):
            raise ValueError('Incomplete or duplicate risk review labels')
        for d in decisions:
            if d['truth'] != m['truth'][d['label_index']] or any(d[k+'_sha256'] != m[k+'_sha256'] for k in ['image', 'label']):
                raise ValueError('Risk review identity differs')
    rows = []
    for original in old['members']:
        row = dict(original); row['review_sources'] = list(row['review_sources'])
        if row['member_id'] in grouped:
            row['review_sources'].append('risk-reconciliation/review')
            row['explicit_review_link_found'] = True
            if row['next_gate'] == 'missing_explicit_review_link':
                row['next_gate'] = 'limited_content_eligibility_decision_required'
        rows.append(row)
    paths.append(Path(__file__).resolve())
    dest = OUT/'review-coverage-matrix-v2.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    return prior.frozen(dest, dict(status='omitted_risk_links_reconciled_not_admission', members=rows,
        counts=dict(Counter(r['next_gate'] for r in rows)),
        active_counts=dict(Counter(r['next_gate'] for r in rows if r['currently_exposed'])),
        erratum='v1 omitted the risk-reconciliation review source; four exposed members were reviewed but pending limited-content eligibility, not unreviewed.',
        inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__ == '__main__':
    r = run(); print(r['counts']); print(r['active_counts'])
