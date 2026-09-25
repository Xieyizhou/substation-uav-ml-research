"""Freeze the complete reviewed candidate dataset; never starts an optimizer."""
from collections import Counter
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT,prior


def run():
    dest=OUT/'reviewed-dataset-v1/manifest.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    names=['protocol','closeout-inventory-v2','source-screen-completion','appearance-full-mask-coverage',
           'lighting-full-mask-coverage','remaining-fullframe-closure','legacy-fullframe-review-v1/review',
           'cohort-review-links','early-positive-review-links','negative-review-links','low-light-review-links',
           'candidate-loader-v2/attempt-01/completion','development-exact-overlap-v1','reference-fingerprint-overlap-v1',
           'resolved-legacy-lineages','risk-derivation-closure-v1','quality-hold-exposure-impact']
    paths=[OUT/(name+'.json') for name in names];records=[prior.read(path) for path in paths]
    for r in records:prior.verify(r)
    data=dict(zip(names,records));q=data['closeout-inventory-v2'];p=data['protocol']
    selected={m['member_id'] for m in q['members'] if m['stage']=='quality_supported_source_role_gate_pending'}
    if data['source-screen-completion']['status']!='all_candidate_source_screens_verified':raise ValueError('Incomplete source screen')
    if data['development-exact-overlap-v1']['matches'] or data['reference-fingerprint-overlap-v1']['matches']:raise ValueError('Role fingerprint conflict')
    coverage={}
    def bind(mid,path):
        if mid not in selected:return
        if mid in coverage:raise ValueError('Overlapping full-frame approval scopes')
        coverage[mid]=str(path)
    for name in ('appearance-full-mask-coverage','lighting-full-mask-coverage'):
        for m in data[name]['members']:bind(m['member_id'],OUT/(name+'.json'))
    for name in ('remaining-fullframe-closure','legacy-fullframe-review-v1/review'):
        for d in data[name]['decisions']:bind(d['member_id'],OUT/(name+'.json'))
    for name in ('cohort-review-links','negative-review-links','low-light-review-links'):
        for m in data[name]['members']:bind(m['member_id'],OUT/(name+'.json'))
    for m in data['early-positive-review-links']['members']:
        if m['status']=='existing_bounded_review_validated':bind(m['member_id'],OUT/'early-positive-review-links.json')
    if set(coverage)!=selected:raise ValueError('Full-frame evidence missing: '+str(sorted(selected-set(coverage))))
    exported=data['candidate-loader-v2/attempt-01/completion']
    if exported['optimizer_created'] or exported['backward_executed'] or exported['validation_run']:raise ValueError('Invalid preflight')
    if {m['member_id'] for m in exported['members']}!=selected:raise ValueError('Staged member mismatch')
    originals={m['member_id']:m for m in p['members']};source_groups={m['member_id']:m for m in data['resolved-legacy-lineages']['members']}
    rows=[];counts=Counter()
    for m in exported['members']:
        original=originals[m['member_id']]
        for key in ('image','label'):
            path=Path(m[key+'_path'])
            if prior.file_sha256(path)!=original[key+'_sha256']:raise ValueError('Staged bytes stale')
            paths.append(path)
        source=source_groups.get(m['member_id'])
        row=dict(m,full_frame_evidence=coverage[m['member_id']],
            data_role='bounded_development_training_dataset',source_independence_claim=False,
            resolved_historical_source=source,training_admitted=False,promotable=False)
        rows.append(row);counts.update(m['class_instances'])
    if len(rows)!=376:raise ValueError('Unexpected reviewed cohort size')
    dest.parent.mkdir(exist_ok=True);paths.append(Path(__file__).resolve())
    return prior.frozen(dest,dict(status='reviewed_dataset_frozen_training_schedule_not_ready',members=rows,
        excluded=[m for m in q['members'] if m['stage']=='whole_frame_held'],
        member_count=len(rows),subset_counts=dict(Counter(m['subset'] for m in rows)),full_label_instances=dict(counts),
        dataset_quality_and_source_checks_complete=True,actual_single_pass_loader_verified=True,
        training_schedule_ready=False,training_started=False,
        retained_limits=['AI-assisted bounded quality, not pixel visibility certification for all images.',
            'Shared layout/assets and derived variants; no independent-scene or generalization claim.',
            '53 whole frames remain excluded, with original files and labels untouched.',
            'Historical documentation gaps remain post-hoc limitations, not rewritten collection approvals.',
            'Old exposure schedules cannot be reused: exact counts, brightness and full-sequence loader checks remain required.'],
        inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__=='__main__':print(run()['status'])
