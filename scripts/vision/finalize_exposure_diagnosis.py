"""Validate all evidence and apply pre-frozen development/retention gates."""
import sys
import statistics
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.vision.exposure_protocol import OUT, PRIOR, NAMES, SEEDS, prepare, read, save, verify, file_sha256
from scripts.vision.run_exposure_diagnosis import checked_cell

VARIANTS = ('original', 'material', 'background', 'lighting')


def stats(values):
    valid = [v for v in values if v is not None]
    return {'mean': statistics.mean(valid) if valid else None,
            'min_seed': min(valid) if valid else None, 'max_seed': max(valid) if valid else None,
            'stdev': statistics.stdev(valid) if len(valid) > 1 else None,
            'values': values, 'defined_seed_count': len(valid)}


def aggregate(records):
    result = {}
    for variant in VARIANTS:
        result[variant] = {metric: stats([r['summary'][variant][metric] for r in records]) for metric in
                           ('planned_instance_hit_rate', 'instance_recall', 'matched_precision', 'unmatched_predictions')}
        result[variant]['per_class'] = {name: {metric: stats([r['summary'][variant]['per_class'][name][metric] for r in records])
            for metric in ('instance_recall', 'matched_precision', 'unmatched_predictions')} for name in NAMES}
    result['no_target'] = {metric: stats([r['negative_summary'][metric] for r in records])
                           for metric in ('frame_false_positive_rate', 'unmatched_predictions')}
    return result


def policy_checks(candidate, reference, historical, policy):
    checks = []
    for rule in policy['acceptance_policy']:
        variant, metric, summary = rule['metric'].split('.')
        actual = candidate[variant][metric][summary]
        passed = actual + 1e-12 >= rule['value'] if rule['op'] == '>=' else actual <= rule['value'] + 1e-12
        checks.append({**rule, 'actual': actual, 'passed': passed})
    for ref_name, ref in [('same_budget_R', reference), ('historical_A', historical)]:
        for variant in policy['retention']['variants']:
            for category in (None, *NAMES):
                a = candidate[variant] if category is None else candidate[variant]['per_class'][category]
                b = ref[variant] if category is None else ref[variant]['per_class'][category]
                delta = a['instance_recall']['mean'] - b['instance_recall']['mean']
                checks.append({'metric': f'{variant}.{category or "all"}.instance_recall', 'reference': ref_name,
                    'actual_delta': delta, 'minimum_delta': -policy['retention']['tolerance'],
                    'passed': delta >= -policy['retention']['tolerance'] - 1e-12})
    return {'passed': all(c['passed'] for c in checks), 'checks': checks}


def verify_review(review, manifest):
    verify(review)
    verify(manifest)
    expected = {p['prediction_id']: (frame, p) for frame in manifest['frames'] for p in frame['predictions']}
    decisions = review['decisions']
    if len(decisions) != len(expected) or {d['prediction_id'] for d in decisions} != set(expected):
        raise ValueError('Review decisions missing or duplicated')
    for d in decisions:
        frame, prediction = expected[d['prediction_id']]
        if d.get('variant') != frame.get('variant') or d['view_id'] != frame['view_id']:
            raise ValueError('Review light variant or view identity mismatch')
        if (d['decision'] != 'reviewed' or d['content_category'] == 'unknown' or not d['reason'] or
            d['review_nature'] != 'AI-assisted' or not d['reviewed_at']):
            raise ValueError('Unresolved visual review')
        for key in ('image', 'evidence'):
            if d[f'{key}_sha256'] != frame[f'{key}_sha256'] or file_sha256(frame[f'{key}_path']) != d[f'{key}_sha256']:
                raise ValueError('Stale visual evidence')
        if d['bbox_xyxy'] != prediction['bbox_xyxy'] or d['confidence'] != prediction['confidence']:
            raise ValueError('Reviewed prediction changed')


def finalize():
    protocol = prepare()
    lineage_path = OUT/'source-lineage.json'
    verify(read(lineage_path))
    verification_path = OUT/'verification.json'
    verification = read(verification_path)
    verify(verification)
    if verification['status'] != 'passed':
        raise ValueError('Targeted regressions or baseline verification failed')
    inputs = {str(OUT/'protocol.json'): file_sha256(OUT/'protocol.json'), str(Path(__file__)): file_sha256(Path(__file__))}
    inputs[str(lineage_path)] = file_sha256(lineage_path)
    inputs[str(verification_path)] = file_sha256(verification_path)
    progress_path = OUT/'training-progress.json'
    progress = read(progress_path)
    verify(progress)
    if progress['status'] != 'complete' or set(progress['completed_cells']) != set(protocol['schedules']):
        raise ValueError('Training grid progress incomplete')
    inputs[str(progress_path)] = file_sha256(progress_path)
    records = {}
    training = {}
    for key in protocol['schedules']:
        completion_path = OUT / key / 'completion.json'
        training[key] = checked_cell(completion_path, protocol)
        inputs[str(completion_path)] = file_sha256(completion_path)
    for key in list(protocol['schedules']) + [f'historical-{a}-{s}' for a in ('A', 'D') for s in SEEDS]:
        path = OUT / 'evaluation' / f'{key}.json'
        record = read(path)
        verify(record)
        if record['status'] != 'complete' or len(record['rows']) != 48 or len(record['negative_rows']) != 48:
            raise ValueError('Evaluation incomplete')
        if record['matching_conflicts']:
            raise ValueError('Unresolved planned-target matching conflict')
        records[key] = record
        inputs[str(path)] = file_sha256(path)
    review_path, manifest_path = OUT/'visual-review-v2.json', OUT/'review-manifest-v2.json'
    verify_review(read(review_path), read(manifest_path))
    inputs.update({str(p): file_sha256(p) for p in (review_path, manifest_path)})
    groups = {f'{a}-{steps}': aggregate([records[f'{a}-{steps}-{s}'] for s in SEEDS]) for a in 'RXY' for steps in (100, 300)}
    historical = aggregate([records[f'historical-A-{s}'] for s in SEEDS])
    old = read(PRIOR/'development-evaluation.json')
    # Recomputed historical reference must retain the previously frozen metrics.
    for a in ('A', 'D'):
        for seed in SEEDS:
            old_result, new_result = old['results'][f'{a}_{seed}'], records[f'historical-{a}-{seed}']
            for variant in VARIANTS:
                for metric in ('planned_instance_hit_rate', 'instance_recall', 'matched_precision', 'unmatched_predictions'):
                    old_value = old_result['paired_summary'][variant][metric]
                    new_value = new_result['summary'][variant][metric]
                    if old_value != new_value and (old_value is None or new_value is None or abs(old_value - new_value) > 1e-12):
                        raise ValueError('Historical reproduction differs')
    policy = {family: policy_checks(groups[family], groups[f'R-{family.split("-")[1]}'], historical, protocol)
              for family in protocol['selection_order']}
    selected = next((family for family in protocol['selection_order'] if policy[family]['passed']), None)
    loss_prefixes = {}
    for arm in 'RXY':
        for seed in SEEDS:
            losses = []
            for steps in (100, 300):
                path = Path(training[f'{arm}-{steps}-{seed}']['weights']).parents[1]/'results.csv'
                with path.open() as stream:
                    losses.append(list(csv.DictReader(stream)))
            keys = ('train/box_loss', 'train/cls_loss', 'train/dfl_loss')
            loss_prefixes[f'{arm}-{seed}'] = {
                'first_100_step_losses_identical': all(losses[0][i][k] == losses[1][i][k] for i in range(10) for k in keys),
                'terminal_losses': {str(steps): {k: float(losses[j][-1][k]) for k in keys} for j, steps in enumerate((100, 300))}}
    deltas = {}
    for left, right in [('X-100', 'Y-100'), ('X-300', 'Y-300'), ('R-100', 'R-300'), ('X-100', 'X-300'), ('Y-100', 'Y-300')]:
        deltas[f'{right}_minus_{left}'] = {v: {m: (groups[right][v][m]['mean']-groups[left][v][m]['mean']
            if groups[right][v][m]['mean'] is not None and groups[left][v][m]['mean'] is not None else None)
            for m in ('planned_instance_hit_rate', 'instance_recall', 'matched_precision')} for v in VARIANTS}
        deltas[f'{right}_minus_{left}']['no_target_fpr'] = groups[right]['no_target']['frame_false_positive_rate']['mean']-groups[left]['no_target']['frame_false_positive_rate']['mean']
    return save(OUT/'completion.json', {'status': 'development_candidate_selected' if selected else 'development_complete_no_candidate',
        'selected_family': selected, 'training_cells': 18, 'evaluated_weights': 24,
        'protocol_identity': protocol['identity'], 'inputs': inputs, 'aggregate': groups,
        'historical_A': historical, 'policy_results': policy, 'contrasts': deltas,
        'training_loss_comparison': loss_prefixes,
        'protected_label_accessed': False, 'threshold_tuned': False})


if __name__ == '__main__':
    result = finalize()
    print(result['status'], result['selected_family'], result['identity'])
