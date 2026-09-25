"""Frozen six-endpoint development evaluation; never opens sealed scene data."""
import sys
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.vision.train_hard_negative_coverage import TRAIN, OUT, BASE, prepare, checked_cell, verify_tree
from scripts.vision.exposure_protocol import read, save, verify, file_sha256, SEEDS, NAMES
from scripts.vision.evaluate_exposure_diagnosis import predict
from scripts.vision.evaluate_paired_visual_factors import checked_rows
from scripts.vision.evaluate_visual_augmentation_abcd import paired_truth
from scripts.vision.exposure_metrics import score, summary
from scripts.vision.finalize_exposure_diagnosis import aggregate, policy_checks, VARIANTS

EVAL = OUT / 'evaluation-v1'
OLD = BASE / 'exposure-controlled-diagnosis-v1'
KEYS = [f'{arm}-100-{seed}' for arm in ('O', 'N') for seed in SEEDS]


def validate(record, key, paired, negatives, inputs):
    verify(record)
    if record['status'] != 'complete' or record['cell'] != key or record['inputs'] != inputs:
        raise ValueError('Incomplete or stale evaluation')
    expected = {(r['view_id'], r['variant'], r['image_sha256']) for r, _ in paired}
    actual = {(r['view_id'], r['variant'], r['image_sha256']) for r in record['rows']}
    neg_expected = {(r['view_id'], r['variant'], r['image_sha256']) for r in negatives}
    neg_actual = {(r['view_id'], r['variant'], r['image_sha256']) for r in record['negative_rows']}
    if len(record['rows']) != 48 or actual != expected or len(record['negative_rows']) != 48 or neg_actual != neg_expected:
        raise ValueError('Evaluation membership missing or duplicated')
    if record['matching_conflicts'] or any(r['matching_conflict'] for r in record['rows']):
        raise ValueError('Unresolved target matching conflict')


def main():
    from ultralytics import YOLO
    training = prepare()
    verify_tree(OUT / 'training-completion.json')
    cells = {key: checked_cell(TRAIN / key / 'completion.json', training) for key in KEYS}
    reviewed, review_path = checked_rows()
    paired, receipt_inputs = paired_truth(reviewed)
    neg_path = BASE / 'hard-negative-isolated-v2/semantic-review.json'
    negative = read(neg_path)
    if negative['status'] != 'reviewed' or negative['accepted'] != 48 or negative['held']:
        raise ValueError('Negative review incomplete')
    inputs = dict(receipt_inputs)
    sources = [Path(__file__), OUT / 'training-completion.json', TRAIN / 'protocol.json', review_path, neg_path,
               OLD / 'protocol.json']
    sources += [ROOT / 'scripts/vision' / f'{name}.py' for name in
                ('exposure_metrics', 'evaluate_exposure_diagnosis', 'evaluate_paired_visual_factors',
                 'evaluate_visual_augmentation_abcd', 'finalize_exposure_diagnosis')]
    for key in KEYS:
        sources += [TRAIN / key / 'completion.json', Path(cells[key]['weights'])]
    historical = {}
    for family in ('R-100', 'historical-A', 'Y-100'):
        historical[family] = []
        for seed in SEEDS:
            path = OLD / 'evaluation' / f'{family}-{seed}.json'
            verify_tree(path)
            r = read(path)
            if r['status'] != 'complete' or len(r['rows']) != 48 or len(r['negative_rows']) != 48 or r['matching_conflicts']:
                raise ValueError('Historical reference incomplete')
            historical[family].append(r)
            sources.append(path)
    for row in reviewed + negative['frames']:
        if row['decision'] != 'accepted' or file_sha256(row['image_path']) != row['image_sha256']:
            raise ValueError('Review or image stale')
        inputs[row['image_path']] = row['image_sha256']
    inputs.update({str(p): file_sha256(p) for p in sources})
    old_policy = read(OLD / 'protocol.json')
    protocol_data = dict(status='frozen', inputs=inputs, cells=KEYS,
        acceptance_policy=old_policy['acceptance_policy'], retention=old_policy['retention'],
        inference=dict(device='cpu', imgsz=640, confidence=.37, diagnostic_confidence=.001,
                       nms_iou=.7, agnostic_nms=False, max_det=300, matching_iou=.5),
        selection='Only N family with all three seeds may qualify; no checkpoint or seed selection.',
        scope='Viewed development sets only: 12 paired poses/48 images and 48 negative images; seeds are repeated measurements.')
    protocol_path = EVAL / 'protocol.json'
    if protocol_path.exists():
        protocol = read(protocol_path)
        verify_tree(protocol_path)
        if any(protocol[k] != v for k, v in protocol_data.items()):
            raise ValueError('Frozen evaluation protocol changed')
    else:
        protocol = save(protocol_path, protocol_data)
    source_inputs = {str(protocol_path): file_sha256(protocol_path)}
    records = {}
    for key in KEYS:
        path = EVAL / f'{key}.json'
        if path.exists():
            record = read(path)
        else:
            print(f'EVALUATE {key}', flush=True)
            model = YOLO(cells[key]['weights'])
            rows = [score(r, truth, predict(model, r['image_path'], .37), predict(model, r['image_path'], .001)) for r, truth in paired]
            negs = []
            for r in negative['frames']:
                formal = predict(model, r['image_path'], .37)
                low = predict(model, r['image_path'], .001)
                negs.append(dict(view_id=r['view_id'], variant=r['variant'], image_sha256=r['image_sha256'],
                    predictions=formal, low_predictions=low, frame_has_prediction=bool(formal)))
            deltas = []
            for pair_id in sorted({r['pair_id'] for r in rows}):
                group = {r['variant']: r for r in rows if r['pair_id'] == pair_id}
                for v in VARIANTS[1:]:
                    a, b = group['original'], group[v]
                    deltas.append(dict(pair_id=pair_id, variant=v,
                        planned_hit_delta=int(b['planned_instance_hit'])-int(a['planned_instance_hit']),
                        matched_instance_delta=len(b['matches'])-len(a['matches']),
                        unmatched_prediction_delta=b['unmatched_prediction_count']-a['unmatched_prediction_count']))
            record = save(path, dict(status='complete', cell=key, inputs=source_inputs, rows=rows,
                negative_rows=negs, paired_deltas=deltas,
                summary={v: summary([r for r in rows if r['variant'] == v]) for v in VARIANTS},
                negative_summary=dict(frame_false_positive_rate=sum(r['frame_has_prediction'] for r in negs)/48,
                                      unmatched_predictions=sum(len(r['predictions']) for r in negs)),
                matching_conflicts=sum(r['matching_conflict'] for r in rows)))
            del model
        validate(record, key, paired, negative['frames'], source_inputs)
        records[key] = record
        print(f'COMPLETE {key} FPR={record["negative_summary"]["frame_false_positive_rate"]:.6f}', flush=True)
    groups = {arm: aggregate([records[f'{arm}-100-{s}'] for s in SEEDS]) for arm in ('O', 'N')}
    reference, history = aggregate(historical['R-100']), aggregate(historical['historical-A'])
    gates = {arm: policy_checks(groups[arm], reference, history, protocol) for arm in groups}
    reproduction = {str(seed): all(records[f'O-100-{seed}'][k] == historical['Y-100'][i][k]
        for k in ('summary', 'negative_summary', 'rows', 'negative_rows')) for i, seed in enumerate(SEEDS)}
    gains = []
    for seed in SEEDS:
        old = {(r['pair_id'], r['variant']): r for r in records[f'O-100-{seed}']['rows']}
        for r in records[f'N-100-{seed}']['rows']:
            a = old[r['pair_id'], r['variant']]
            gains.append(dict(seed=seed, pair_id=r['pair_id'], variant=r['variant'], category=r['category'],
                planned_hit_delta=int(r['planned_instance_hit'])-int(a['planned_instance_hit']),
                matched_instance_delta=len(r['matches'])-len(a['matches'])))
    suites = ['tests.test_hard_negative_coverage_evaluation', 'tests.test_hard_negative_coverage',
              'tests.test_exposure_diagnosis', 'tests.test_paired_visual_factors', 'tests.test_visual_bridge_training']
    test = subprocess.run([sys.executable, '-m', 'unittest', *suites], cwd=ROOT, capture_output=True, text=True)
    baseline = subprocess.run([sys.executable, 'scripts/vision/verify_experiment_baseline.py'], cwd=ROOT, capture_output=True, text=True)
    data = json.loads(baseline.stdout) if baseline.returncode == 0 else {}
    passed = test.returncode == baseline.returncode == 0 and data.get('integrity_passed') and data.get('pinned_files_verified') == 40
    verification_path = EVAL / 'verification.json'
    save(verification_path, dict(status='passed' if passed else 'failed', test_output=test.stdout+test.stderr,
        baseline=data, inputs={str(ROOT/(s.replace('.', '/')+'.py')): file_sha256(ROOT/(s.replace('.', '/')+'.py')) for s in suites}))
    if not passed:
        raise ValueError('Regression or baseline verification failed')
    verify_tree(protocol_path)
    completion = save(EVAL / 'completion.json', dict(status='development_complete_candidate' if gates['N']['passed'] else 'development_complete_no_candidate',
        selected_candidate='N-100-all-three-seeds' if gates['N']['passed'] else None,
        groups=groups, same_budget_R=reference, historical_A=history, gates=gates,
        old_reference_prediction_reproduced=reproduction, between_family_paired_deltas=gains,
        inputs={str(p): file_sha256(p) for p in [protocol_path, verification_path]+[EVAL/f'{k}.json' for k in KEYS]}))
    print(json.dumps({'status': completion['status'], 'gates': gates, 'groups': groups, 'reproduction': reproduction}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
