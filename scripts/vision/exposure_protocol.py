"""Immutable inputs and paired, exact exposure schedules for development diagnosis."""
import json
import random
from collections import Counter
from pathlib import Path

from src.ml.artifacts import file_sha256, object_sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'data/research/ml_training_recovery_v1'
OUT = BASE / 'exposure-controlled-diagnosis-v1'
PRIOR = BASE / 'visual-bridge-training-v1'
SEEDS = (7, 17, 27)
NAMES = ('transformer', 'switchgear', 'capacitor_bank', 'reactor')
QUOTAS = {'R': {'base': 330, 'regular': 270},
          'X': {'base': 216, 'regular': 156, 'bridge_positive': 156, 'hard_negative': 72},
          'Y': {'base': 216, 'regular': 156, 'bridge_positive': 120, 'hard_negative': 108}}
FLAGS = {'training_admitted': False, 'promotable': False, 'unseen_scene_status': 'sealed_not_evaluated'}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, record):
    record = {**record, **FLAGS}
    record.pop('identity', None)
    record['identity'] = object_sha256(record)
    write_json(path, record)
    return record


def verify(record):
    if object_sha256({k: v for k, v in record.items() if k != 'identity'}) != record['identity']:
        raise ValueError('Record identity changed')
    for path, digest in record.get('inputs', {}).items():
        if file_sha256(path) != digest:
            raise ValueError(f'Input bytes changed: {path}')


def cycle(ids, count, rng):
    result = []
    while len(result) < count:
        block = sorted(ids)
        rng.shuffle(block)
        result.extend(block)
    return result[:count]


def make_schedules(rows, seed):
    subsets = {s: [r['member_id'] for r in rows if r['subset'] == s] for s in QUOTAS['X']}
    rng = random.Random(seed)
    streams = {s: cycle(ids, 1800, random.Random(f'{seed}:{s}')) for s, ids in subsets.items()}
    result = {a: [] for a in QUOTAS}
    offsets = Counter()
    for _ in range(3):
        positions = [s for s, n in QUOTAS['X'].items() for _ in range(n)]
        rng.shuffle(positions)
        # Swap the final 36 appearance slots, preserving shared stream prefixes.
        appearance = [i for i, s in enumerate(positions) if s == 'bridge_positive']
        swaps = set(appearance[-36:])
        shared = {}
        for i, subset in enumerate(positions):
            if i not in swaps:
                shared[i] = streams[subset][offsets[subset]]
                offsets[subset] += 1
        for arm in ('X', 'Y'):
            extra_subset = 'bridge_positive' if arm == 'X' else 'hard_negative'
            counts = Counter(result[arm] + list(shared.values()))
            extras = []
            for _ in swaps:
                ids = sorted(subsets[extra_subset])
                rng.shuffle(ids)
                mid = min(ids, key=lambda mid: counts[mid])
                counts[mid] += 1
                extras.append(mid)
            extra_iter = iter(extras)
            result[arm].extend(shared[i] if i not in swaps else next(extra_iter) for i in range(600))
        reference = [s for s, n in QUOTAS['R'].items() for _ in range(n)]
        rng.shuffle(reference)
        for subset in reference:
            result['R'].append(streams[subset][offsets['R:' + subset]])
            offsets['R:' + subset] += 1
    return result


def exposures(rows, draws):
    by_id = {r['member_id']: r for r in rows}
    classes, subsets, groups = Counter(), Counter(), Counter()
    for mid in draws:
        row = by_id[mid]
        subsets[row['subset']] += 1
        groups[row['lineage_id']] += 1
        classes.update(row['class_instances'])
    return {'draws': len(draws), 'unique_frames': len(set(draws)), 'by_subset': dict(subsets),
            'by_member': dict(Counter(draws)), 'class_instance_exposure': dict(classes),
            'by_lineage': dict(groups), 'lineage_count': len(groups)}


def prepare():
    path = OUT / 'protocol.json'
    if path.exists():
        record = read(path)
        verify(record)
        return record
    prior = read(PRIOR / 'protocol.json')
    admission_path = BASE / 'visual-bridge-supplement-v2/development-admission.json'
    admission = read(admission_path)
    if admission['status'] != 'eligible_for_frozen_development_training':
        raise ValueError('Development admission missing')
    sources = {r['member_id']: r for r in admission['entries']}
    rows = []
    inputs = {}
    for original in prior['pool_rows']:
        row = dict(original)
        for kind in ('image', 'label'):
            p = row[f'{kind}_path']
            if file_sha256(p) != row[f'{kind}_sha256']:
                raise ValueError(f'Frozen member changed: {p}')
            inputs[p] = row[f'{kind}_sha256']
        row['lineage_id'] = sources.get(row['member_id'], {}).get('derivation_group', row['member_id'])
        row['lineage_resolution'] = 'source_derivation_group' if row['member_id'] in sources else 'member_only_no_independence_claim'
        row['class_instances'] = dict(Counter(NAMES[int(line.split()[0])] for line in Path(row['label_path']).read_text().splitlines()))
        rows.append(row)
    if Counter(r['subset'] for r in rows) != {'base': 66, 'regular': 48, 'bridge_positive': 48, 'hard_negative': 24}:
        raise ValueError('Unexpected pool membership')
    OUT.mkdir(parents=True, exist_ok=True)
    schedules, expected, datasets = {}, {}, {}
    for arm in QUOTAS:
        members = [r for r in rows if r['subset'] in QUOTAS[arm]]
        listing = OUT / f'{arm}.txt'
        listing.write_text('\n'.join(r['image_path'] for r in members) + '\n')
        dataset = OUT / f'{arm}.yaml'
        dataset.write_text(f'path: {OUT}\ntrain: {listing}\nval: {listing}\nnames: {json.dumps(NAMES)}\n')
        datasets[arm] = str(dataset)
        inputs[str(dataset)] = file_sha256(dataset)
        inputs[str(listing)] = file_sha256(listing)
    for seed in SEEDS:
        all_draws = make_schedules(rows, seed)
        for arm, draws in all_draws.items():
            for steps in (100, 300):
                key = f'{arm}-{steps}-{seed}'
                schedules[key] = draws[:steps * 6]
                expected[key] = exposures(rows, schedules[key])
    files = [PRIOR / 'protocol.json', PRIOR / 'development-evaluation.json', admission_path,
             BASE / 'paired-visual-factors-v1/semantic-review.json',
             BASE / 'hard-negative-isolated-v2/semantic-review.json', Path(__file__),
             Path(prior['controls']['initial_weights'])]
    for source in files:
        inputs[str(source)] = file_sha256(source)
    return save(path, {'status': 'frozen_before_training', 'schema_version': 1, 'inputs': inputs,
        'pool_rows': rows, 'quotas_per_600': QUOTAS, 'schedules': schedules, 'exposures': expected,
        'datasets': datasets, 'seeds': SEEDS, 'steps': [100, 300],
        'controls': {**prior['controls'], 'epochs': 'steps/10', 'optimizer_steps': [100, 300],
                     'final_validation_role': 'training_fit_only', 'selection': 'terminal_weights_only'},
        'acceptance_policy': prior['acceptance_policy']['required_all'],
        'retention': {'variants': ['original', 'lighting'], 'tolerance': .05,
                      'metrics': ['instance_recall', 'per_class_instance_recall'],
                      'references': ['same_budget_R', 'historical_A']},
        'selection_order': ['Y-100', 'X-100', 'Y-300', 'X-300'],
        'diagnostic_confidence': .001, 'formal_confidence': .37,
        'limits': ['Only added source groups have resolved pose lineage; common member IDs do not prove independent scenes.',
                   'X/Y exchange appearance exposure for negative exposure at fixed budget.']})
