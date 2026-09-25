"""Join explicit review coverage without synthesizing approval decisions."""
from collections import Counter, defaultdict
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior


def run():
    names = ['cohort-review-links', 'early-positive-review-links', 'legacy-review-links', 'negative-review-links']
    paths = [OUT/'protocol.json', OUT/'quality-isolation-v3.json', OUT/'remaining-review/review.json']
    paths += [OUT/(name+'.json') for name in names]
    records = [prior.read(p) for p in paths]
    for record in records: prior.verify(record)
    p, q, remaining = records[:3]
    links = defaultdict(list)
    for name, record in zip(names, records[3:]):
        for member in record['members']:
            links[member['member_id']].append(name)
    for decision in remaining['decisions']:
        links[decision['member_id']].append('remaining-review')
    quarantine = {m['member_id']: m for m in q['members']}
    active = set().union(*(set(v) for v in p['actual_exposures'].values()))
    rows = []
    for member in p['members']:
        mid = member['member_id']; status = quarantine[mid]['status']
        rows.append(dict(member_id=mid, subset=member['subset'], currently_exposed=mid in active,
            quarantine_status=status, review_sources=links[mid],
            explicit_review_link_found=bool(links[mid]),
            full_admission_decision_present=False,
            next_gate='preserve_hold' if status == 'quarantined' else
                ('review_source_full_frame_and_role_disposition' if links[mid] else 'missing_explicit_review_link'),
            training_eligible=False))
    dest = OUT/'review-coverage-matrix-v1.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    paths.append(Path(__file__).resolve())
    return prior.frozen(dest, dict(status='coverage_inventory_not_training_admission', members=rows,
        counts=dict(Counter(r['next_gate'] for r in rows)),
        active_counts=dict(Counter(r['next_gate'] for r in rows if r['currently_exposed'])),
        limitation='Links show existing explicit evidence, not final eligibility. No selection by model fit.',
        inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__ == '__main__':
    r = run(); print(r['counts']); print(r['active_counts'])
