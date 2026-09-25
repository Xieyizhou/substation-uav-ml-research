#!/usr/bin/env python3
"""Conservative dHash screening; similarity hits are quarantined, not relabelled."""
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.training.hard_example_curator import dhash64

BASE = ROOT / 'data/research/ml_training_recovery_v1'
MASKS = [0] + [1 << i for i in range(64)] + [(1 << i) | (1 << j) for i in range(64) for j in range(i)]


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def neighbors(value, index):
    for mask in MASKS:
        for row in index.get(value ^ mask, ()):
            yield mask.bit_count(), row


def main():
    out = BASE / 'near-dedup-v1'
    out.mkdir(exist_ok=True)
    inputs = {}
    def bind(path):
        inputs[str(path)] = file_sha256(path)
        return path
    review_path = bind(BASE / 'semantic-review-final.json')
    review = json.loads(review_path.read_text())
    supplied = review.pop('review_identity')
    if object_sha256(review) != supplied:
        raise ValueError('Review identity mismatch')
    reviewed = [r for r in review['decisions'] if r['decision'] == 'accepted']
    manifest_path = bind(ROOT / 'data/research/visual_hard_examples_v2_12/curated-run18-run19-full-strict-v1/curated-manifest.json')
    original = {r['image_sha256']: r for r in json.loads(manifest_path.read_text())['selected']}
    clean_path = bind(BASE / 'replay-exact-clean-v1/train_membership.jsonl')
    replay = read(clean_path)
    view = ROOT / 'data/research/visual_yolo_v2'
    def fingerprint(row):
        path = view / row['image_relative_path']
        if file_sha256(path) != row['payload_sha256']:
            raise ValueError(f'Changed replay image: {path}')
        return {'image_sha256': row['payload_sha256'], 'perceptual_hash': dhash64(path),
                'sample_id': row['sample_id'], 'recording_id': row['recording_id'], 'role': 'development_replay'}
    cache = out / 'replay-fingerprints.jsonl'
    with ThreadPoolExecutor(max_workers=8) as executor:
        refs = []
        for i, row in enumerate(executor.map(fingerprint, replay), 1):
            refs.append(row)
            if i % 2000 == 0:
                print(f'Replay image hashes verified and dHash computed: {i}/{len(replay)}', flush=True)
    cache.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in refs))
    protected = []
    for name in ('protected-v2-image-index-v1', 'protected-blind-image-index-v1', 'protected-qualification-index-v1'):
        folder = ROOT / 'data/research/canonical_views_v1' / name
        receipt = json.loads(bind(folder / 'receipt.json').read_text())
        path = bind(folder / 'images.jsonl')
        if inputs[str(path)] != receipt['index_sha256']:
            raise ValueError('Protected index changed')
        protected.extend({**r, 'role': name} for r in read(path))
    indexes = {}
    for role, rows in [('protected', protected), ('replay', refs)]:
        index = defaultdict(list)
        for r in rows:
            index[int(r['perceptual_hash'], 16)].append(r)
        indexes[role] = index
    retained, decisions, accepted_index = [], [], defaultdict(list)
    protected_recordings = {r.get('recording_id') for r in protected} - {None}
    for row in sorted(reviewed, key=lambda r: (r['image_sha256'], r['frame_id'])):
        if file_sha256(row['image_path']) != row['image_sha256']:
            raise ValueError('Reviewed image changed')
        source = original[row['image_sha256']]
        value = int(dhash64(row['image_path']), 16)
        result = {**row, 'seed': source['seed'], 'split': source['split'],
                  'perceptual_hash': f'{value:016x}', 'training_admitted': False}
        if source['split'] != 'development':
            raise ValueError('Non-development candidate')
        matches = []
        for role in ('protected', 'replay'):
            hits = list(neighbors(value, indexes[role]))
            if hits:
                distance, ref = min(hits, key=lambda item: (item[0], item[1]['image_sha256']))
                matches.append({'role': role, 'count': len(hits), 'minimum_hamming_distance': distance,
                                'example': ref})
        within = list(neighbors(value, accepted_index))
        if within:
            distance, ref = min(within, key=lambda item: item[0])
            matches.append({'role': 'retained_reviewed', 'count': len(within),
                            'minimum_hamming_distance': distance, 'example': ref})
        result['near_matches'] = matches
        result['status'] = 'quarantine_similarity_review' if matches else 'retained_pending_other_gates'
        decisions.append(result)
        if not matches:
            retained.append(result)
            accepted_index[value].append({'image_sha256': row['image_sha256'], 'frame_id': row['frame_id']})
    # The replay pool is itself screened against protected fingerprints.
    replay_hits = []
    for row in refs:
        hits = list(neighbors(int(row['perceptual_hash'], 16), indexes['protected']))
        if hits:
            distance, match = min(hits, key=lambda pair: pair[0])
            replay_hits.append({**row, 'match_count': len(hits), 'minimum_hamming_distance': distance,
                                'example': match, 'status': 'quarantine_similarity_review'})
    files = {}
    for name, rows in [('decisions.jsonl', decisions), ('retained.jsonl', retained), ('replay-protected-near-hits.jsonl', replay_hits)]:
        path = out / name
        path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
        files[name] = {'path': str(path), 'sha256': file_sha256(path), 'count': len(rows)}
    coverage = Counter(c for row in retained for c in row.get('taxonomy_confirmed', []))
    coverage['no_target'] = sum(r.get('no_target_confirmed') is True for r in retained)
    report = {'inputs': inputs, 'files': files, 'reviewed_count': len(reviewed), 'retained_count': len(retained),
              'quarantined_count': len(decisions)-len(retained), 'coverage_frame_counts': dict(coverage),
              'quota_deficits_to_600': {c: max(0,600-coverage[c]) for c in ('transformer','switchgear','capacitor_bank','reactor','no_target')},
              'maps': dict(Counter(r['map_id'] for r in retained)),
              'collection_groups': len({r['collection_identity'] for r in retained}),
              'seed_groups': len({r['seed'] for r in retained}),
              'replay_protected_near_hit_frames': len(replay_hits),
              'algorithm': 'Pillow grayscale resize 9x8 dHash64; Hamming <=2; direct retained-representative comparison, no transitive chaining',
              'training_admitted': False,
              'limits': ['Similarity hits are candidates for review, not confirmed duplicate images.',
                         'Historical hard pools other than the reviewed source are not fully covered.',
                         'Qualification inventory completeness remains unproven.',
                         'Cross-encoding pixel equality and full recording/seed/view lineage isolation remain pending.',
                         'Replay fingerprints are exclusion evidence only; no held-out labels or metrics read.']}
    report['identity'] = object_sha256(report)
    write_json(out / 'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
