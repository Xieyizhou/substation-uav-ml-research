"""Append named source-coverage holds without altering historical supervision."""
from collections import Counter
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior


def isolate(members, previous, decisions):
    index = {m['member_id']: m for m in members}
    old = {m['member_id']: m for m in previous}
    if len(index) != len(members) or len(old) != len(previous) or set(index) != set(old):
        raise ValueError('Member coverage mismatch')
    keys = [(d['member_id'], d['object_id']) for d in decisions]
    if len(set(keys)) != len(keys) or any(mid not in index for mid, _ in keys):
        raise ValueError('Duplicate or unknown decision')
    risks = {d['member_id'] for d in decisions if d['status'] == 'hold_pending_named_target_coverage_risk'}
    lineages = {index[mid]['lineage_id'] for mid in risks}
    rows = []
    for mid, member in index.items():
        row = dict(old[mid]); row['reasons'] = list(row['reasons'])
        if mid in risks or member['lineage_id'] in lineages:
            row['status'] = 'quarantined'
            row['reasons'].append('named_target_coverage_risk' if mid in risks else 'registered_same_lineage_target_coverage_risk')
        row['training_eligible'] = False
        rows.append(row)
    unresolved = sorted(mid for mid in risks if index[mid].get('lineage_resolution') == 'member_only_no_independence_claim')
    return rows, unresolved


def run():
    paths = [OUT/'protocol.json', OUT/'quality-isolation-v2.json',
             OUT/'source-spatial-context/spatial-review-02.json', Path(__file__).resolve()]
    p, old, review = [prior.read(path) for path in paths[:3]]
    for record in (p, old, review): prior.verify(record)
    rows, unresolved = isolate(p['members'], old['members'], review['decisions'])
    dest = OUT/'quality-isolation-v3.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    return prior.frozen(dest, dict(status='source_coverage_hold_added_not_dataset_admission',
        members=rows, counts=dict(Counter(r['status'] for r in rows)),
        unresolved_risk_lineage_members=unresolved,
        limitation='Member-only lineage does not prove source isolation; resolve before dataset admission.',
        dataset_ready=False, training_started=False, history_modified=False,
        inputs={str(path): prior.file_sha256(path) for path in paths}))


if __name__ == '__main__':
    print(run()['counts'])
