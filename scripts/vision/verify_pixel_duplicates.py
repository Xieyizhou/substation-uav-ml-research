#!/usr/bin/env python3
"""Verify decoded RGB equality against the supplied exclusion inventories.

Protected images are used solely to compute exclusion fingerprints. No labels,
predictions, visual sheets or evaluation metrics are read from protected data.
"""
import hashlib
import io
import json
import struct
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json

BASE = ROOT / 'data/research/ml_training_recovery_v1'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rgb_digest(payload):
    with Image.open(io.BytesIO(payload)) as image:
        # Fail rather than discard alpha, precision or color-management data.
        if image.mode != 'RGB' or image.info.get('icc_profile'):
            raise ValueError(f'Unsupported source image mode/profile: {image.mode}')
        size = image.size
        digest = hashlib.sha256(b'rgb8-row-major-v1\0' + struct.pack('>II', *size) + image.tobytes()).hexdigest()
    return digest, size


def fingerprint(item):
    digest, path = item
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f'Changed image bytes: {path}')
    pixel, size = rgb_digest(payload)
    return {'image_sha256': digest, 'pixel_sha256': pixel, 'size': size, 'verified_path': path}


def main():
    out = BASE / 'pixel-dedup-v1'
    out.mkdir(exist_ok=True)
    inputs = {}
    def bind(path):
        inputs[str(path)] = file_sha256(path)
        return path
    near_report = json.loads(bind(BASE / 'near-dedup-v1/report.json').read_text())
    for spec in near_report['files'].values():
        if file_sha256(Path(spec['path'])) != spec['sha256']:
            raise ValueError('Changed near-screen output')
    candidates = rows(bind(BASE / 'near-dedup-v1/decisions.jsonl'))
    replay = rows(bind(BASE / 'replay-exact-clean-v1/train_membership.jsonl'))
    replay_near = {r['sample_id'] for r in rows(bind(BASE / 'near-dedup-v1/replay-protected-near-hits.jsonl'))}
    protected = []
    for name in ('protected-v2-image-index-v1', 'protected-blind-image-index-v1', 'protected-qualification-index-v1'):
        folder = ROOT / 'data/research/canonical_views_v1' / name
        receipt = json.loads(bind(folder / 'receipt.json').read_text())
        index = bind(folder / 'images.jsonl')
        if file_sha256(index) != receipt['index_sha256']:
            raise ValueError('Protected index mismatch')
        source_paths = {}
        if name == 'protected-qualification-index-v1':
            for source in receipt['sources']:
                if source.get('collection_identity') is None:
                    continue
                collection = ROOT / 'data/research/visual_hard_examples_v2_1/qualification' / source['run_id'] / 'collection'
                capture = json.loads(bind(collection / 'collection-receipt.json').read_text())
                if capture['identity'] != source['collection_identity']:
                    raise ValueError('Qualification collection changed')
                for member in capture['members']:
                    source_paths[member['image_sha256']] = str(collection / member['rgb_path'])
        for row in rows(index):
            path = source_paths[row['image_sha256']] if source_paths else str(Path(receipt['source_view']) / row['image_relative_path'])
            protected.append({**row, 'path': path, 'role': name})
    sources = {}
    for row in protected:
        sources.setdefault(row['image_sha256'], row['path'])
    view = ROOT / 'data/research/visual_yolo_v2'
    for row in replay:
        sources.setdefault(row['payload_sha256'], str(view / row['image_relative_path']))
    for row in candidates:
        sources.setdefault(row['image_sha256'], row['image_path'])
    print(f'Computing RGB fingerprints for {len(sources)} unique file hashes', flush=True)
    fingerprints = {}
    fp_path = out / 'pixel-fingerprints.jsonl'
    with fp_path.open('w') as stream, ThreadPoolExecutor(max_workers=8) as executor:
        for i, result in enumerate(executor.map(fingerprint, sources.items()), 1):
            fingerprints[result['image_sha256']] = result['pixel_sha256']
            stream.write(json.dumps(result, sort_keys=True) + '\n')
            if i % 4000 == 0:
                stream.flush()
                print(f'Pixel fingerprints verified: {i}/{len(sources)}', flush=True)
    prot_index, replay_index = defaultdict(list), defaultdict(list)
    for row in protected:
        prot_index[fingerprints[row['image_sha256']]].append(row)
    for row in replay:
        replay_index[fingerprints[row['payload_sha256']]].append(row)
    decisions, seen = [], {}
    for row in candidates:
        pixel = fingerprints[row['image_sha256']]
        hits = prot_index.get(pixel, [])
        replay_hits = replay_index.get(pixel, [])
        if hits:
            status = 'exclude_protected_pixel_equal'
        elif replay_hits:
            status = 'duplicate_replay_pixel_equal'
        elif pixel in seen:
            status = 'duplicate_reviewed_pixel_equal'
        elif row['near_matches']:
            status = 'pixel_distinct_near_similarity_unresolved'
        else:
            status = 'no_pixel_duplicate_or_prior_near_hit_in_supplied_scope'
        decisions.append({**row, 'pixel_sha256': pixel, 'pixel_status': status,
                          'protected_equal_count': len(hits), 'replay_equal_count': len(replay_hits),
                          'protected_equal_example': hits[0] if hits else None,
                          'replay_equal_example': replay_hits[0] if replay_hits else None,
                          'same_pixel_reviewed_representative': seen.get(pixel), 'training_admitted': False})
        seen.setdefault(pixel, row['frame_id'])
    replay_decisions, clean_replay, replay_seen = [], [], set()
    for row in replay:
        pixel = fingerprints[row['payload_sha256']]
        hits = prot_index.get(pixel, [])
        status = 'exclude_protected_pixel_equal' if hits else 'duplicate_within_replay_pixel_equal' if pixel in replay_seen else 'pixel_distinct_near_similarity_unresolved' if row['sample_id'] in replay_near else 'no_pixel_duplicate_or_prior_near_hit_in_supplied_scope'
        replay_decisions.append({**row, 'pixel_sha256': pixel, 'pixel_status': status,
                                 'protected_equal_count': len(hits), 'protected_equal_example': hits[0] if hits else None})
        if not hits and pixel not in replay_seen:
            clean_replay.append(row)
            replay_seen.add(pixel)
    files = {'pixel-fingerprints.jsonl': {'path': str(fp_path), 'sha256': file_sha256(fp_path), 'count': len(fingerprints)}}
    for name, data in [('reviewed-decisions.jsonl', decisions), ('replay-decisions.jsonl', replay_decisions), ('replay-pixel-distinct-membership.jsonl', clean_replay)]:
        path = out / name
        path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in data))
        files[name] = {'path': str(path), 'sha256': file_sha256(path), 'count': len(data)}
    report = {'schema_version': 1, 'status': 'pixel_equality_check_complete', 'inputs': inputs, 'files': files,
              'algorithm': 'RGB8 original dimensions, no resize, crop, alpha removal, color transform or lossy re-encoding; versioned SHA256 of dimensions and decoded row-major bytes',
              'protected_index_rows': len(protected), 'verified_unique_file_hashes': len(fingerprints),
              'reviewed_status_counts': dict(Counter(r['pixel_status'] for r in decisions)),
              'replay_status_counts': dict(Counter(r['pixel_status'] for r in replay_decisions)),
              'derived_replay_count': len(clean_replay), 'training_admitted': False,
              'limits': ['Unequal pixels do not rule out near duplicates or scene/recording leakage.',
                         'Prior similarity quarantines remain in force pending further review.',
                         'Protected inventories have the previously documented scope limits.',
                         'One representative file was rehashed per declared file SHA256, not every alias path.',
                         'Derived membership does not change any existing training entrypoint.']}
    report['identity'] = object_sha256(report)
    write_json(out / 'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
