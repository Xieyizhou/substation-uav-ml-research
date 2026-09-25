"""Endpoint-only 100/300-step comparison with an exactly repeated effective sequence."""
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.vision.run_exposure_diagnosis as runner
from scripts.vision.validate_canonical_batch_fix import OUT as REFERENCE
from scripts.vision.run_order_diagnosis import read, save, file_sha256, verify_tree, exposures, predict, checked_rows, paired_truth, score, summary, VARIANTS
from scripts.vision.exposure_protocol import BASE, NAMES
from scripts.vision.run_fixed_sequence_diagnosis import parameter_hash

OUT = REFERENCE.parent / 'fixed-budget-diagnosis-v1'
KEY = 'CB-300-7'
FIELDS = ('train/box_loss', 'train/cls_loss', 'train/dfl_loss')


def repeat_sequence(draws):
    if len(draws) != 600 or any(not isinstance(x, str) or not x for x in draws):
        raise ValueError('Require one complete frozen 100-step sequence')
    return list(draws) * 3


def loss_curve(path, epochs):
    with Path(path).open() as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    if [int(r['epoch']) for r in rows] != list(range(1, epochs + 1)):
        raise ValueError('Incomplete epoch record')
    curve = [[float(r[n]) for n in FIELDS] for r in rows]
    if not all(math.isfinite(x) for row in curve for x in row):
        raise ValueError('Nonfinite loss')
    return curve


def retention(a, b):
    checks = []
    for v in ('original', 'lighting'):
        for name in (None, *NAMES):
            x = a[v] if name is None else a[v]['per_class'][name]
            y = b[v] if name is None else b[v]['per_class'][name]
            delta = x['instance_recall'] - y['instance_recall']
            checks.append(dict(variant=v, category=name or 'all', delta=delta, passed=delta >= -.05 - 1e-12))
    return checks


def prepare():
    path = OUT / 'protocol.json'
    if path.exists():
        verify_tree(path)
        return read(path)
    verify_tree(REFERENCE / 'completion.json')
    ref = read(REFERENCE / 'protocol.json')
    draws = repeat_sequence(ref['schedules']['CF-100-7'])
    paths = [REFERENCE / 'completion.json', Path(__file__), ROOT / 'scripts/vision/run_exposure_diagnosis.py', ROOT / 'tests/test_fixed_budget_diagnosis.py']
    return save(path, dict(status='frozen', schedules={KEY: draws}, pool_rows=ref['pool_rows'],
        exposures={KEY: exposures(ref['pool_rows'], draws)}, datasets={'CB': ref['datasets']['CF-100-7']}, controls=ref['controls'],
        reference='CF-100-7: latest verified canonical control, not selected by quality',
        design='Repeat the exact effective 600 draws three times; independent v2.11 initialization; endpoint only; seed 7; all optimizer and augmentation settings unchanged.',
        interpretation='One sequence budget diagnostic, not a three-seed candidate family. No new data, threshold search, architecture change or sealed-scene evaluation.',
        retention_tolerance=.05, max_attempts=3, selected_candidate=None,
        inputs={str(q): file_sha256(q) for q in paths}))


def main():
    from ultralytics import YOLO
    p = prepare()
    runner.OUT = OUT
    cp = OUT / KEY / 'completion.json'
    if not cp.exists() and len(list((OUT / KEY).glob('attempt-*'))) >= p['max_attempts']:
        raise ValueError('Attempt limit reached')
    cell = runner.train(KEY, p)
    reference_cell = read(REFERENCE / 'CF-100-7/completion.json')
    short = loss_curve(Path(reference_cell['exposure_path']).parent / 'results.csv', 10)
    long = loss_curve(Path(cell['exposure_path']).parent / 'results.csv', 30)
    if long[:10] != short:
        raise ValueError('First 100 steps do not reproduce recorded reference losses')
    reviewed, rpath = checked_rows()
    paired, receipt_inputs = paired_truth(reviewed)
    npath = BASE / 'hard-negative-isolated-v2/semantic-review.json'
    neg = read(npath)
    if neg['status'] != 'reviewed' or neg['accepted'] != 48 or neg['held']:
        raise ValueError('Incomplete negative review')
    rp = REFERENCE / 'evaluation-CF-100-7.json'
    verify_tree(rp)
    ref = read(rp)
    sources = {str(q): file_sha256(q) for q in (OUT / 'protocol.json', cp, rpath, npath, rp)}
    sources.update(receipt_inputs)
    for row in reviewed + neg['frames']:
        if row['decision'] != 'accepted' or file_sha256(row['image_path']) != row['image_sha256']:
            raise ValueError('Stale review')
        sources[row['image_path']] = row['image_sha256']
    ep = OUT / 'evaluation.json'
    if ep.exists():
        verify_tree(ep)
        result = read(ep)
        if result['inputs'] != sources:
            raise ValueError('Stale evaluation')
    else:
        print('EVALUATE_300_ENDPOINT', flush=True)
        model = YOLO(cell['weights'])
        digest = parameter_hash(model)
        rows = [score(row, truth, predict(model, row['image_path'], .37), predict(model, row['image_path'], .001)) for row, truth in paired]
        negatives = []
        for row in neg['frames']:
            preds = predict(model, row['image_path'], .37)
            negatives.append(dict(view_id=row['view_id'], variant=row['variant'], image_sha256=row['image_sha256'], predictions=preds, frame_has_prediction=bool(preds)))
        result = save(ep, dict(status='complete', inputs=sources, rows=rows, negative_rows=negatives, parameter_sha256=digest,
            summary={v: summary([r for r in rows if r['variant'] == v]) for v in VARIANTS},
            negative_summary=dict(frame_false_positive_rate=sum(r['frame_has_prediction'] for r in negatives) / 48,
                unmatched_predictions=sum(len(r['predictions']) for r in negatives)),
            matching_conflicts=sum(r['matching_conflict'] for r in rows)))
    if len(result['rows']) != 48 or len(result['negative_rows']) != 48 or result['matching_conflicts']:
        raise ValueError('Incomplete evaluation or unresolved matching conflict')
    gains = {}
    for v in VARIANTS:
        a = {r['pair_id']: r for r in result['rows'] if r['variant'] == v}
        b = {r['pair_id']: r for r in ref['rows'] if r['variant'] == v}
        if set(a) != set(b) or len(a) != 12:
            raise ValueError('Incomplete pair identity')
        gains[v] = dict(gained=[k for k in a if a[k]['planned_assigned_hit'] and not b[k]['planned_assigned_hit']],
                        lost=[k for k in a if b[k]['planned_assigned_hit'] and not a[k]['planned_assigned_hit']])
    verify_tree(OUT / 'protocol.json')
    save(OUT / 'completion.json', dict(status='budget_diagnostic_complete', selected_candidate=None,
        prefix_losses_equal=True, prefix_state_equality='not measured; exact exposure and recorded epoch loss prefix verified',
        optimizer_steps=300, image_exposures=1800, reference_optimizer_steps=100, reference_reused=True,
        loss_curve=long, training_member_validation_role='training fit diagnostic only',
        summary_100=ref['summary'], summary_300=result['summary'], negative_100=ref['negative_summary'], negative_300=result['negative_summary'],
        paired_gains_losses=gains, retention_vs_100=retention(result['summary'], ref['summary']),
        candidate_status='Not eligible: single-sequence diagnostic does not replace three-seed, same-budget R and historical A gates.',
        inputs={str(q): file_sha256(q) for q in (OUT / 'protocol.json', ep, rp, cp)}))
    print('BUDGET_DIAGNOSTIC_COMPLETE', result['negative_summary'], result['summary'], flush=True)


if __name__ == '__main__':
    main()
