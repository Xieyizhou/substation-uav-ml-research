#!/usr/bin/env python3
"""Create a new replay membership excluding protected exact image matches."""
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def partition(rows, protected):
    kept, excluded, seen = [], [], set()
    for row in rows:
        digest = row.get('image_sha256') or row['payload_sha256']
        reason = 'protected_exact_overlap' if digest in protected else 'within_replay_exact_duplicate' if digest in seen else None
        if reason:
            excluded.append({**row, 'exclusion_reason': reason})
        else:
            kept.append(row)
            seen.add(digest)
    return kept, excluded


def main():
    view = ROOT / 'data/research/visual_yolo_v2'
    source = view / 'identity/train_membership.jsonl'
    rows = read_rows(source)
    protected, references, matches = set(), [], []
    dev_hashes = {r['payload_sha256'] for r in rows}
    for name in ('v2', 'blind', 'qualification'):
        folder = ROOT / f'data/research/canonical_views_v1/protected-{name}-image-index-v1'
        if name == 'qualification':
            folder = ROOT / 'data/research/canonical_views_v1/protected-qualification-index-v1'
        index = folder / 'images.jsonl'
        receipt = json.loads((folder / 'receipt.json').read_text())
        digest = file_sha256(index)
        if receipt['index_sha256'] != digest:
            raise ValueError(f'Changed protected index: {index}')
        members = read_rows(index)
        if len(members) != receipt.get('row_count', receipt.get('members_count')):
            raise ValueError('Protected index count mismatch')
        references.append({'path': str(index), 'sha256': digest, 'count': len(members)})
        protected.update(r['image_sha256'] for r in members)
        for row in members:
            if row['image_sha256'] in dev_hashes:
                path = Path(receipt['source_view']) / row['image_relative_path']
                if file_sha256(path) != row['image_sha256']:
                    raise ValueError(f'Protected match bytes changed: {path}')
                matches.append({**row, 'verified_image_path': str(path)})
    kept, excluded = partition(rows, protected)
    for row in excluded:
        if file_sha256(view / row['image_relative_path']) != row['payload_sha256']:
            raise ValueError('Excluded development bytes changed')
    assert not {r['payload_sha256'] for r in kept} & protected
    output = ROOT / 'data/research/ml_training_recovery_v1/replay-exact-clean-v1'
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, values in [('train_membership.jsonl', kept), ('excluded.jsonl', excluded), ('verified-protected-matches.jsonl', matches)]:
        path = output / name
        path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in values))
        files[name] = {'path': str(path), 'sha256': file_sha256(path), 'count': len(values)}
    report = {'schema_version': 1, 'status': 'derived_replay_exact_overlap_removed',
              'source_membership': str(source), 'source_membership_sha256': file_sha256(source),
              'image_root': str(view), 'source_count': len(rows), 'retained_count': len(kept),
              'excluded_count': len(excluded), 'exclusion_counts': dict(Counter(r['exclusion_reason'] for r in excluded)),
              'protected_matching_images_rehashed': len(matches), 'development_excluded_images_rehashed': len(excluded),
              'remaining_protected_exact_overlap': 0, 'files': files, 'references': references,
              'training_admitted': False, 'near_duplicate_checked': False,
              'limits': ['Derived membership only; existing training CLI still uses its configured dataset.',
                         'Raw file SHA256 cannot exclude identical pixels with different image encoding.',
                         'Prior model training and historical blind results are not retroactively repaired.',
                         'Near duplicates, recording isolation and final quotas remain pending.',
                         'Qualification index covers discovered v2_1 qualification only; repository-wide coverage is unproven.']}
    report['identity'] = object_sha256(report)
    write_json(output / 'report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
