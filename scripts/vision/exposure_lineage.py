"""Resolve source pose groups without changing the frozen training schedule."""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.vision.exposure_protocol import BASE, OUT, prepare, read, save, file_sha256


def main():
    protocol = prepare()
    trusted_path = BASE/'trusted-training-base-v1/frozen-ledger.json'
    candidates_path = BASE/'visual-augmentation-240-v2/frozen-intake-ledger.json'
    admission_path = BASE/'visual-bridge-supplement-v2/development-admission.json'
    trusted = {f'base:{r["view_id"]}': r for r in read(trusted_path)['entries']}
    candidates = {f'candidate:{r["candidate_id"]}': r for r in read(candidates_path)['entries']}
    admission = {r['member_id']: r for r in read(admission_path)['entries']}
    rows = []
    for member in protocol['pool_rows']:
        mid = member['member_id']
        if mid in trusted:
            source = trusted[mid]
            if source['review_decision'] != 'accepted' or source['training_image_sha256'] != member['image_sha256']:
                raise ValueError('Trusted source changed')
            group = f'{source["map_id"]}:view:{source["view_id"]}'
            metadata = {'map_id': source['map_id'], 'equipment_instance_id': source['expected_object_id'],
                        'source_image_path': source['source_image_path'], 'view_id': source['view_id']}
        elif mid in candidates:
            source = candidates[mid]
            if source['review_decision'] != 'accepted':
                raise ValueError('Regular source unreviewed')
            group = source['derivation_group']
            metadata = {k: source[k] for k in ('map_layout_id', 'asset_family_ids', 'equipment_instance_id', 'recording_group', 'pose_id')}
        else:
            source = admission[mid]
            if not source['development_training_eligible']:
                raise ValueError('Bridge source ineligible')
            group = source['derivation_group']
            metadata = {k: source[k] for k in ('map_id', 'pair_id', 'view_id')}
        rows.append({'member_id': mid, 'subset': member['subset'], 'source_pose_group': group, **metadata})
    groups = {r['member_id']: r['source_pose_group'] for r in rows}
    exposures = {key: dict(Counter(groups[mid] for mid in draws)) for key, draws in protocol['schedules'].items()}
    return save(OUT/'source-lineage.json', {'status': 'resolved', 'members': rows, 'exposure_by_source_group': exposures,
        'pose_groups_by_subset': {s: len({r['source_pose_group'] for r in rows if r['subset'] == s})
                                 for s in ('base', 'regular', 'bridge_positive', 'hard_negative')},
        'inputs': {str(p): file_sha256(p) for p in (OUT/'protocol.json', trusted_path, candidates_path, admission_path, Path(__file__))},
        'limits': ['Pose/source groups are not independent map layouts or asset families. Same scene context remains correlated.']})


if __name__ == '__main__':
    print(main()['pose_groups_by_subset'])
