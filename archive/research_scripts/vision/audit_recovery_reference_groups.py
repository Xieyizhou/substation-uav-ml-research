#!/usr/bin/env python3
"""Extend recovery exclusion evidence with discovered protected captures.

Only image fingerprints and source identifiers are used from protected data.
Diagnostic images are partial/derived exclusion evidence, not complete captures.
No training view or model is changed by this audit.
"""
import hashlib
import io
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.audit_reviewed_near_duplicates import neighbors
from scripts.vision.build_v2_1_training_view import CLASSES, _select_replay
from scripts.vision.verify_pixel_duplicates import rgb_digest

BASE = ROOT / 'data/research/ml_training_recovery_v1'
CLEAR = 'no_pixel_duplicate_or_prior_near_hit_in_supplied_scope'
PROTECTED_SPLITS = {'validation', 'full_validation', 'blind', 'heldout_test', 'test', 'qualification'}


def discover(roots, patterns=()):
    command = ['rg', '--files', '--hidden', '--no-ignore', *map(str, roots)]
    for pattern in patterns:
        command.extend(['-g', pattern])
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode not in (0, 1):
        raise ValueError(result.stderr)
    return sorted(result.stdout.splitlines())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def check_identity(value):
    if object_sha256({k: v for k, v in value.items() if k != 'identity'}) != value['identity']:
        raise ValueError('Artifact identity mismatch')


def recording_seed(recording_id):
    match = re.fullmatch(r'visual-v2-.+-(\d+)-r\d+', recording_id)
    if not match:
        raise ValueError(f'Cannot derive source seed from recording: {recording_id}')
    return int(match[1])


def fingerprint(row):
    payload = Path(row['image_path']).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if row.get('image_sha256') and digest != row['image_sha256']:
        raise ValueError(f'Changed image: {row["image_path"]}')
    pixel, size = rgb_digest(payload)
    if row.get('pixel_sha256') and pixel != row['pixel_sha256']:
        raise ValueError(f'Changed RGB fingerprint: {row["image_path"]}')
    with Image.open(io.BytesIO(payload)) as image:
        pixels = list(image.convert('L').resize((9, 8)).getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return {**row, 'image_sha256': digest, 'pixel_sha256': pixel,
            'perceptual_hash': f'{value:016x}', 'size': size}


def make_indexes(rows):
    indexes = {name: defaultdict(list) for name in ('image_sha256', 'pixel_sha256', 'perceptual_hash')}
    for row in rows:
        for name, index in indexes.items():
            key = int(row[name], 16) if name == 'perceptual_hash' else row[name]
            index[key].append(row)
    return indexes


def match_summary(row, indexes):
    result = {}
    for name in ('image_sha256', 'pixel_sha256'):
        hits = indexes[name].get(row[name], [])
        result[name + '_matches'] = len(hits)
        if hits:
            result[name + '_example'] = hits[0]['image_path']
    hits = list(neighbors(int(row['perceptual_hash'], 16), indexes['perceptual_hash']))
    result['near_matches'] = len(hits)
    if hits:
        distance, example = min(hits, key=lambda pair: (pair[0], pair[1]['image_path']))
        result.update(minimum_hamming_distance=distance, near_example=example['image_path'])
    return result


def main():
    inputs, outputs = {}, {}
    out = BASE / 'reference-group-audit-v1'
    out.mkdir(exist_ok=True)

    def bind(path):
        path = Path(path)
        inputs[str(path)] = file_sha256(path)
        return path

    def save(name, value, jsonl=False):
        path = out / name
        if jsonl:
            path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in value))
        else:
            write_json(path, value)
        outputs[name] = {'sha256': file_sha256(path), 'path': str(path)}

    for source in (__file__, ROOT / 'scripts/vision/verify_pixel_duplicates.py',
                   ROOT / 'scripts/vision/audit_reviewed_near_duplicates.py',
                   ROOT / 'scripts/vision/build_v2_1_training_view.py', ROOT / 'src/ml/artifacts.py'):
        bind(source)
    candidate_path = bind(BASE / 'near-review-final-v1/candidate-manifest.json')
    manifest = json.loads(candidate_path.read_text())
    check_identity(manifest)
    candidates = manifest['selected']
    prior_report = json.loads(bind(BASE / 'near-review-final-v1/report.json').read_text())
    check_identity(prior_report)
    if prior_report['output_files'][str(candidate_path)] != file_sha256(candidate_path):
        raise ValueError('Candidate manifest changed')
    pixel_report = json.loads(bind(BASE / 'pixel-dedup-v1/report.json').read_text())
    check_identity(pixel_report)
    replay_path = bind(BASE / 'pixel-dedup-v1/replay-decisions.jsonl')
    if file_sha256(replay_path) != pixel_report['files']['replay-decisions.jsonl']['sha256']:
        raise ValueError('Replay decisions changed')
    replay = [r for r in read_rows(replay_path) if r['pixel_status'] == CLEAR]

    old_protected = []
    for name in ('protected-v2-image-index-v1', 'protected-blind-image-index-v1', 'protected-qualification-index-v1'):
        folder = ROOT / 'data/research/canonical_views_v1' / name
        receipt = json.loads(bind(folder / 'receipt.json').read_text())
        path = bind(folder / 'images.jsonl')
        if file_sha256(path) != receipt['index_sha256']:
            raise ValueError('Protected index changed')
        for source in (folder / 'receipt.json', path):
            if file_sha256(source) != pixel_report['inputs'][str(source)]:
                raise ValueError('Protected reference differs from the prior exclusion scope')
        old_protected.extend(read_rows(path))
    protected_recordings = {r['recording_id'] for r in old_protected}
    protected_seeds = {int(r['seed']) if 'seed' in r else recording_seed(r['recording_id']) for r in old_protected}
    old_hashes = {r['image_sha256'] for r in old_protected}

    # Explicit no-ignore discovery is essential: raw captures are gitignored.
    paths = discover(['data/research', 'outputs/research'], ['collection-receipt.json', 'run-receipt.json'])
    collections, inventory, run_inventory, supplemental = {}, [], [], []
    for relative in paths:
        path = bind(ROOT / relative)
        if path.name != 'collection-receipt.json':
            run = json.loads(path.read_text())
            is_protected = run.get('split') in PROTECTED_SPLITS or '/qualification/' in relative
            run_inventory.append({'path': str(path), 'sha256': file_sha256(path),
                                  'declared_split': run.get('split'), 'seed': run.get('seed'),
                                  'collection_identity': run.get('collection_identity'), 'protected': is_protected})
            if is_protected:
                protected_recordings.add(path.parent.name)
                if run.get('seed') is not None:
                    protected_seeds.add(int(run['seed']))
            continue
        capture = json.loads(path.read_text())
        collections[str(path.parent)] = capture
        is_protected = capture.get('split') in PROTECTED_SPLITS or '/qualification/' in relative
        record = {'receipt_path': str(path), 'receipt_sha256': file_sha256(path),
                  'collection_identity': capture['identity'], 'declared_split': capture.get('split'),
                  'map_id': capture.get('map_id'), 'seed': capture.get('seed'),
                  'member_count': len(capture.get('members', [])), 'protected': is_protected}
        inventory.append(record)
        if not is_protected:
            continue
        check_identity(capture)
        if capture['frame_count'] != len(capture['members']):
            raise ValueError('Protected collection member count mismatch')
        protected_seeds.add(int(capture['seed']))
        protected_recordings.add(path.parent.parent.name)
        for member in capture['members']:
            supplemental.append({'image_path': str(path.parent / member['rgb_path']),
                                 'image_sha256': member['image_sha256'], 'frame_id': member['frame_id'],
                                 'collection_identity': capture['identity'], 'recording_id': path.parent.parent.name,
                                 'seed': capture['seed'], 'map_id': capture['map_id'],
                                 'role': 'qualification_collection' if '/qualification/' in relative else 'historical_validation_collection',
                                 'previous_index_has_file_hash': member['image_sha256'] in old_hashes})

    config_inventory, qualification_roots = [], set()
    for relative in discover(['config/perception'], ['*.json']):
        config = json.loads((ROOT / relative).read_text())
        if not isinstance(config, dict) or not (config.get('qualification') is True or 'qualification' in Path(relative).name):
            continue
        path = bind(ROOT / relative)
        root = config.get('output_root')
        config_inventory.append({'path': str(path), 'sha256': file_sha256(path), 'output_root': root})
        if root:
            qualification_roots.add(root)
            for map_row in config.get('maps', []):
                for run_ids in map_row.get('runs', {}).values():
                    protected_recordings.update(run_ids)

    diagnostic_inventory = []
    for relative in sorted(qualification_roots):
        root = ROOT / relative
        files = discover([root]) if root.exists() else []
        images = [p for p in files if Path(p).suffix.lower() in {'.jpg', '.jpeg', '.ppm', '.png'}]
        diagnostic_inventory.append({'root': relative, 'exists': root.exists(), 'available_image_count': len(images),
                                     'image_paths': images, 'complete_recording_coverage': False})
        for path in images:
            run_id = Path(path).parent.parent.name.removesuffix('-diagnostics')
            protected_recordings.add(run_id)
            supplemental.append({'image_path': path, 'recording_id': run_id,
                                 'role': 'qualification_partial_diagnostic',
                                 'raw_capture_equivalence_verified': False})

    protected_collections = {r['collection_identity'] for r in inventory if r['protected']}
    group_rows, development = [], []
    for row in candidates:
        capture = collections.get(str(Path(row['collection'])))
        if capture is None or capture['identity'] != row['collection_identity']:
            raise ValueError('Missing or changed candidate source collection')
        check_identity(capture)
        source = [r for r in capture['members'] if r['frame_id'] == row['frame_id']]
        if len(source) != 1 or any(source[0][k] != row[k] for k in ('image_sha256', 'rgb_path', 'seed', 'map_id', 'split')):
            raise ValueError('Candidate source member mismatch')
        if row['split'] != 'development' or any(capture[k] != row[k] for k in ('seed', 'map_id', 'split')):
            raise ValueError('Candidate split/group mismatch')
        run_id = Path(row['collection']).parent.name
        reasons = []
        if row['collection_identity'] in protected_collections:
            reasons.append('protected_collection_identity')
        if row['seed'] in protected_seeds:
            reasons.append('protected_seed')
        if run_id in protected_recordings:
            reasons.append('protected_recording_id')
        group_rows.append({'kind': 'candidate', 'id': row['frame_id'], 'recording_id': run_id,
                           'seed': row['seed'], 'collection_identity': row['collection_identity'], 'exclusion_reasons': reasons})
        development.append({'kind': 'candidate', 'id': row['frame_id'],
                            'image_path': str(Path(row['collection']) / row['rgb_path']),
                            'image_sha256': row['image_sha256'], 'pixel_sha256': row['pixel_sha256'],
                            'group_exclusion_reasons': reasons})
    for row in replay:
        seed = recording_seed(row['recording_id'])
        reasons = []
        if seed in protected_seeds:
            reasons.append('protected_seed')
        if row['recording_id'] in protected_recordings:
            reasons.append('protected_recording_id')
        group_rows.append({'kind': 'replay', 'id': row['sample_id'], 'recording_id': row['recording_id'],
                           'seed': seed, 'exclusion_reasons': reasons})
        development.append({'kind': 'replay', 'id': row['sample_id'],
                            'image_path': str(ROOT / 'data/research/visual_yolo_v2' / row['image_relative_path']),
                            'image_sha256': row['payload_sha256'], 'pixel_sha256': row['pixel_sha256'],
                            'group_exclusion_reasons': reasons})
    save('source-inventory.json', {'discovery': 'rg --files --hidden --no-ignore; no symlink directory traversal',
                                 'receipt_paths': paths, 'collections': inventory, 'runs': run_inventory,
                                 'qualification_configs': config_inventory, 'qualification_roots': diagnostic_inventory})
    save('source-groups.jsonl', group_rows, True)

    print(f'Verifying {len(supplemental)} supplemental reference paths and {len(development)} development paths', flush=True)
    all_fingerprints = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i, result in enumerate(executor.map(fingerprint, supplemental + development), 1):
            all_fingerprints.append(result)
            if i % 1000 == 0:
                print(f'Image fingerprints verified: {i}/{len(supplemental) + len(development)}', flush=True)
    references = all_fingerprints[:len(supplemental)]
    save('supplemental-protected-fingerprints.jsonl', references, True)
    indexes = make_indexes(references)
    decisions = []
    for row in all_fingerprints[len(supplemental):]:
        matches = match_summary(row, indexes)
        excluded = bool(row['group_exclusion_reasons'] or matches['image_sha256_matches'] or matches['pixel_sha256_matches'] or matches['near_matches'])
        decisions.append({**row, **matches, 'current_run_decision': 'exclude_supplemental_protection' if excluded else 'candidate_pending_other_gates',
                          'training_admitted': False})
    save('decisions.jsonl', decisions, True)
    retained = {(r['kind'], r['id']) for r in decisions if r['current_run_decision'] == 'candidate_pending_other_gates'}
    selected = [r for r in candidates if ('candidate', r['frame_id']) in retained]
    selected_replay = [r for r in replay if ('replay', r['sample_id']) in retained]
    new_manifest = {'schema_version': 1, 'selected': selected, 'selected_count': len(selected),
                    'source_manifest_sha256': file_sha256(candidate_path), 'training_admitted': False,
                    'status': 'candidate_only_not_training_admission',
                    'audit_decisions_sha256': outputs['decisions.jsonl']['sha256']}
    new_manifest['identity'] = object_sha256(new_manifest)
    save('candidate-manifest.json', new_manifest)
    save('replay-candidate-membership.jsonl', selected_replay, True)
    try:
        quota_selection = _select_replay(selected_replay, 1000)
        feasibility = {'feasible': True, 'selected_unique_count': len({r['sample_id'] for _, r in quota_selection}),
                       'assigned_class_counts': dict(Counter(c for c, _ in quota_selection))}
        save('replay-quota-selection.jsonl', [{'assigned_class': c, **r, 'training_admitted': False} for c, r in quota_selection], True)
    except ValueError as error:
        feasibility = {'feasible': False, 'reason': str(error)}
    coverage = Counter(c for r in selected for c in {o['class_name'] for o in r['objects']})
    coverage['no_target'] = sum(not r['objects'] for r in selected)
    report = {'schema_version': 1, 'status': 'complete_for_declared_discovery_scope', 'training_admitted': False,
              'inputs': inputs, 'files': outputs, 'receipt_count': len(paths), 'collection_count': len(inventory),
              'protected_collection_count': sum(r['protected'] for r in inventory),
              'protected_run_receipts_without_collection_identity': sum(r['protected'] and not r['collection_identity'] for r in run_inventory),
              'supplemental_role_counts': dict(Counter(r['role'] for r in references)),
              'collection_members_absent_from_old_file_hash_indexes': sum(r.get('previous_index_has_file_hash') is False for r in references),
              'qualification_output_roots': len(diagnostic_inventory),
              'qualification_roots_without_images': sum(not r['available_image_count'] for r in diagnostic_inventory),
              'protected_seeds': sorted(protected_seeds),
              'candidate_source_groups': len({r['collection_identity'] for r in candidates}),
              'replay_recording_groups': len({r['recording_id'] for r in replay}),
              'group_overlap_frames': {kind: sum(r['kind'] == kind and bool(r['exclusion_reasons']) for r in group_rows) for kind in ('candidate', 'replay')},
              'diagnostic_reference_frames_without_verified_seed_lineage': sum(r['role'] == 'qualification_partial_diagnostic' for r in references),
              'input_candidates': len(candidates), 'retained_candidates': len(selected),
              'input_replay': len(replay), 'retained_replay': len(selected_replay),
              'decisions_by_kind': {kind: dict(Counter(r['current_run_decision'] for r in decisions if r['kind'] == kind)) for kind in ('candidate', 'replay')},
              'coverage_frame_counts': dict(coverage),
              'quota_deficits_to_600': {c: max(0, 600 - coverage[c]) for c in (*CLASSES, 'no_target')},
              'replay_quota_feasibility': feasibility,
              'limits': ['All earlier exclusions remain in effect. New similarity hits are policy exclusions, not confirmed duplicates.',
                         'Collection discovery covers data/research and outputs/research receipt files, including ignored files; other directories, orphan images and external archives are not proven complete.',
                         'Qualification image discovery covers output roots of configs explicitly marked or named qualification in config/perception.',
                         'Diagnostic images provide partial exclusion evidence; their full raw recording coverage, historical transformation provenance and source seed lineage are unverified.',
                         'Group checks establish no known recording/collection/seed reuse only when the overlap count is zero. Different seeds do not establish independent world geometry or camera trajectories.',
                         'Protected labels, predictions and metrics are not used; protected images are not visually opened.',
                         'Existing replay class metadata is used for quota feasibility only; full label audit, canonical increments and final joint deduplication remain pending.',
                         'The final training dataset and model have not been changed.']}
    report['identity'] = object_sha256(report)
    write_json(out / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('inputs', 'files')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
