"""Freeze a bounded impact inventory only; no replay, relabeling or training."""
from collections import Counter
from pathlib import Path

from scripts.vision.brightness_lr_retention import OUT as RETENTION
from scripts.vision.run_dual_box_diagnosis import SOURCE, ROOT, read, verify, frozen, file_sha256
from scripts.vision.run_depth_clip_test import OUT as DEPTH
from scripts.vision.test_body_material_applicability import baseline_verify

OUT = ROOT / 'data/research/ml_training_recovery_v1/annotation-revision-design-v1'


def inventory(pool, frames, schedules):
    ids = [r['member_id'] for r in pool]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate pool member')
    for draws in schedules.values():
        if set(draws) - set(ids):
            raise ValueError('Schedule contains unknown member')
    counts = {k: Counter(v) for k, v in schedules.items()}
    result = []
    for frame in frames:
        if frame['review_ids'] not in (['T027'], ['T036']):
            continue
        if ids.count(frame['member_id']) != 1:
            raise ValueError('Source member missing')
        matches = [r for r in pool if r['member_id'] == frame['member_id']
                   or r['lineage_id'] == frame['lineage_id']]
        result.append(dict(review_id=frame['review_ids'][0],
            source_member_id=frame['member_id'], source_lineage_id=frame['lineage_id'],
            members=[dict(r, planned_exposures={k: v[r['member_id']] for k, v in counts.items()})
                     for r in matches],
            closure_status='bounded_metadata_matches_not_global_source_closure',
            actual_exposures_verified=False,
            gaps=['Each member requires separate source and full-label review.',
                  'Member-only lineage must not be counted as an independent scene.']))
    if {r['review_id'] for r in result} != {'T027', 'T036'} or len(result) != 2:
        raise ValueError('Expected two unique held source frames')
    return result


def main():
    dest = OUT / 'design-receipt.json'
    if dest.exists():
        verify(read(dest))
        print('VALID_DESIGN_REUSED; NOT_TRAINING_READY')
        return
    paths = [RETENTION / 'protocol.json', SOURCE / 'protocol.json',
             SOURCE / 'review-completion.json', SOURCE / 'semantic-review.json',
             DEPTH / 'completion.json', Path(__file__),
             ROOT / 'docs/plans/annotation_revision_design_v1.md',
             ROOT / 'tests/test_annotation_revision_design.py']
    for path in paths[:5]:
        verify(read(path))
    p = read(paths[0])
    rows = inventory(p['pool_rows'], read(paths[1])['frames'], p['schedules'])
    for group in rows:
        for r in group['members']:
            for field in ('image', 'label'):
                path = Path(r[field + '_path'])
                if file_sha256(path) != r[field + '_sha256']:
                    raise ValueError('Changed member ' + str(path))
                paths.append(path)
    baseline = baseline_verify(ROOT / 'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified'] != 40:
        raise ValueError('Baseline integrity failed')
    OUT.mkdir(parents=True, exist_ok=True)
    frozen(dest, dict(status='revision_design_frozen_quality_and_source_closure_pending',
        scope='Current 236-member frozen pool and two held pilot source identities only.',
        affected_source_groups=rows, baseline=baseline,
        training_ready=False, training_started=False, production_ready=False,
        historical_labels_changed=False, revised_training_pool_created=False,
        review_decisions_generated=False,
        next_step='Complete per-member source closure and extreme-truncation review; no automatic label revision.',
        inputs={str(path): file_sha256(path) for path in paths}))
    print('DESIGN_FROZEN; NO_LABEL_WRITES_NO_TRAINING')


if __name__ == '__main__':
    main()
