#!/usr/bin/env python3
"""Recheck retained hard frames against hashed collection truth; never approve labels."""

import argparse
from bisect import bisect_left
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256
from src.vision.canonical.plan import read_record, write_record


def compare_objects(selected, source):
    old = {row['annotation_id']: row for row in source}
    new = {row['annotation_id']: row for row in selected}
    return {
        'removed_annotation_ids': sorted(old.keys() - new.keys()),
        'added_annotation_ids': sorted(new.keys() - old.keys()),
        'changed_annotation_ids': sorted(key for key in old.keys() & new.keys() if old[key] != new[key]),
        'duplicate_annotation_ids': len(old) != len(source) or len(new) != len(selected),
    }


def audit(manifest_path, output):
    manifest = read_record(manifest_path)
    cache, rows, counts = {}, [], Counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for selected in manifest['selected']:
        collection = Path(selected['collection'])
        if str(collection) not in cache:
            receipt = read_record(collection/'collection-receipt.json')
            truth_path = collection/receipt['truth']['relative_path']
            if file_sha256(truth_path) != receipt['truth']['sha256']:
                raise ValueError(f'Changed source truth: {truth_path}')
            truth = sorted((json.loads(line) for line in truth_path.read_text().splitlines()),
                           key=lambda row: row['simulation_timestamp'])
            truth = [row for row in truth if row['validation_status'] == 'valid']
            cache[str(collection)] = (receipt, truth, [row['simulation_timestamp'] for row in truth])
        receipt, truth, times = cache[str(collection)]
        if receipt['identity'] != selected['collection_identity']:
            raise ValueError('Collection identity mismatch')
        image = collection/selected['rgb_path']
        if file_sha256(image) != selected['image_sha256']:
            raise ValueError(f'Changed image: {image}')
        index = bisect_left(times, selected['rgb_timestamp'])
        nearest = min(truth[max(0, index-1):index+1],
                      key=lambda row: abs(row['simulation_timestamp']-selected['rgb_timestamp']), default=None)
        skew = abs(nearest['simulation_timestamp']-selected['rgb_timestamp'])*1000 if nearest else None
        diff = compare_objects(selected['objects'], nearest['objects']) if nearest else {}
        reasons = []
        if nearest is None or skew > 33.334 + 1e-9:
            reasons.append('missing_or_stale_source_truth')
        else:
            reasons.extend(key for key, value in diff.items() if value)
        if not selected['objects']:
            reasons.append('no_target_requires_semantic_review')
        else:
            reasons.append('target_completeness_requires_semantic_review')
        counts.update(reasons)
        rows.append({
            'frame_id': selected['frame_id'], 'map_id': selected['map_id'],
            'image_path': str(image), 'image_sha256': selected['image_sha256'],
            'collection_identity': receipt['identity'], 'source_truth_file_sha256': receipt['truth']['sha256'],
            'source_truth_identity': object_sha256(nearest) if nearest else None,
            'selected_objects': selected['objects'], 'source_objects': nearest['objects'] if nearest else [],
            'skew_ms': skew, **diff, 'reasons': reasons,
            'status': 'quarantined_pending_review', 'training_admitted': False,
        })
    queue = output/'review-queue.jsonl'
    queue.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n' for row in rows))
    return write_record(output/'report.json', {
        'schema_version': 1, 'status': 'structural_audit_complete_semantic_review_pending',
        'source_manifest': str(manifest_path), 'source_manifest_sha256': file_sha256(manifest_path),
        'source_manifest_identity': manifest['identity'], 'frames_checked': len(rows),
        'collections_checked': len(cache), 'reason_frame_counts': dict(counts),
        'images_rehashed': len(rows), 'source_truth_files_rehashed': len(cache),
        'review_queue': str(queue), 'review_queue_sha256': file_sha256(queue),
        'training_admitted': False,
        'limits': ['Nearest valid source truth is not proof of semantic completeness.',
                   'No automatic class assignment, bbox removal or negative-frame approval.'],
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest, args.output), ensure_ascii=False, indent=2))
