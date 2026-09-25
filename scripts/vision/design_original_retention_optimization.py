"""Prospective matched-budget continuation design; deliberately no training entry."""
from collections import Counter
from pathlib import Path
from scripts.vision.run_visibility_repair_training_v2 import OUT as SOURCE, read, save, file_sha256

OUT = SOURCE / 'original-retention-optimization-design-v1'


def build(original, lookup, variants):
    if len(original) != 1800:
        raise ValueError('Expected frozen 300-step sequence')
    extra = list(original[:900])
    reference = list(original) + extra
    treatment = list(reference)
    counts = Counter()
    for i, mid in enumerate(extra, 1800):
        row = lookup[mid]
        if row['subset'] != 'bridge_positive':
            continue
        lineage = row['lineage_id']
        variant = ('neutral_bridge', 'background_bridge')[counts[lineage] % 2]
        counts[lineage] += 1
        replacement = variants[(lineage, variant)]
        if lookup[replacement]['class_instances'] != row['class_instances']:
            raise ValueError('Full class supervision changed')
        treatment[i] = replacement
    if reference[:1800] != treatment[:1800] or reference[:1800] != original:
        raise ValueError('Original exposure prefix changed')
    return reference, treatment


def main():
    dest = OUT / 'design.json'
    if dest.exists():
        raise ValueError('Preserve existing design; do not overwrite')
    p = read(SOURCE / 'protocol.json')
    lookup = {r['member_id']: r for r in p['pool_rows']}
    from scripts.vision.design_visibility_repair_contrast import SOURCE as VISIBILITY
    trace_path = VISIBILITY.parent / 'trace.json'
    trace = read(trace_path)
    variants = {(r['lineage_id'], f['variant']): f['member_id']
                for f in trace['frames'] if (r := lookup.get(f['member_id'])) is not None}
    schedules = {}; ledgers = {}
    for seed in (7, 17, 27):
        original = p['schedules'][f'original_only-300-{seed}']
        reference, treatment = build(original, lookup, variants)
        for arm, draws in [('retained_reference', reference), ('retained_appearance', treatment)]:
            key = f'{arm}-450-{seed}'
            schedules[key] = draws
            classes = Counter()
            for mid in draws:
                classes.update(lookup[mid]['class_instances'])
            ledgers[key] = {'subsets': dict(Counter(lookup[m]['subset'] for m in draws)),
                            'class_instances': dict(classes), 'members': dict(Counter(draws)),
                            'changed_slots': sum(a != b for a, b in zip(reference, draws))}
        assert ledgers[f'retained_reference-450-{seed}']['class_instances'] == ledgers[f'retained_appearance-450-{seed}']['class_instances']
    paths = [SOURCE / 'protocol.json', trace_path, Path(__file__)]
    paths += [SOURCE / f'evaluation-{arm}-300-{seed}.json'
              for arm in ('original_only', 'three_variant') for seed in (7, 17, 27)]
    save(dest, dict(status='draft_blocked_on_previous_review_and_full_preflight', schedules=schedules,
        ledgers=ledgers, optimizer_steps=450, image_exposures=2700,
        interpretation='Matched 450-step budget; identical original 300-step prefix, then original versus appearance bridge exposures. Tests ordered continuation, not interleaved augmentation or pure extra data.',
        training_started=False, required_gates=['Complete previous visual error review and historical retention checks',
        'Deep-verify all inputs; verify exact paired full label coordinates and lineage identities',
        'Freeze evaluation and 450-step matched reference policy; validate actual sampler and optimizer controls',
        'Use all seeds and terminal endpoints only; no sealed scene evaluation'],
        inputs={str(path): file_sha256(path) for path in paths}))
    print(dest)


if __name__ == '__main__':
    main()
