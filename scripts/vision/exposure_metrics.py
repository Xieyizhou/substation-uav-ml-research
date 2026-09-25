"""One-to-one scoring and explicitly bounded, low-confidence error diagnosis."""
from collections import Counter
from scripts.vision.analyze_recovery_paired_calibration import iou
from scripts.vision.evaluate_paired_visual_factors import match, summarize
from scripts.vision.exposure_protocol import NAMES


def missed_reason(truth, low_predictions):
    same = [p for p in low_predictions if p['class_name'] == truth['class_name']]
    if any(iou(p['bbox_xyxy'], truth['bbox_xyxy']) >= .5 and p['confidence'] < .37 for p in same):
        return 'low_confidence_same_class'
    if any(p['class_name'] != truth['class_name'] and iou(p['bbox_xyxy'], truth['bbox_xyxy']) >= .5 for p in low_predictions):
        return 'wrong_class'
    best = max((iou(p['bbox_xyxy'], truth['bbox_xyxy']) for p in same), default=0.)
    if .1 <= best < .5:
        return 'localization'
    return 'no_qualifying_retained_prediction'


def score(row, truth, predictions, low):
    matches, used_p, used_t = match(predictions, truth)
    targets = [i for i, t in enumerate(truth) if t['class_name'] == row['expected_category'] and
               max(abs(a-b) for a, b in zip(t['bbox_xyxy'], row['target_bbox_xyxy'])) <= 1.]
    if len(targets) != 1:
        raise ValueError('Planned target does not resolve uniquely to full-image truth')
    independent = any(p['class_name'] == row['expected_category'] and iou(p['bbox_xyxy'], row['target_bbox_xyxy']) >= .5 for p in predictions)
    assigned = targets[0] in used_t
    misses = []
    for index, t in enumerate(truth):
        if index not in used_t:
            competing = any(p['class_name'] == t['class_name'] and iou(p['bbox_xyxy'], t['bbox_xyxy']) >= .5 for p in predictions)
            misses.append({'truth_index': index, 'class_name': t['class_name'],
                           'reason': missed_reason(t, low), 'formal_matching_competition': competing})
    return {'pair_id': row['pair_id'], 'view_id': row['view_id'], 'variant': row['variant'],
        'category': row['expected_category'], 'object_id': row['expected_object_id'],
        'image_sha256': row['image_sha256'], 'truth': truth, 'predictions': predictions,
        'low_predictions': low, 'matches': matches, 'misses': misses,
        'planned_instance_hit': independent, 'planned_assigned_hit': assigned,
        'planned_truth_index': targets[0], 'matching_conflict': independent != assigned,
        'unmatched_prediction_count': len(predictions) - len(used_p)}


def summary(rows):
    result = summarize(rows)
    per_class = {}
    for name in NAMES:
        total = sum(t['class_name'] == name for r in rows for t in r['truth'])
        predicted = sum(p['class_name'] == name for r in rows for p in r['predictions'])
        matched = sum(m['class_name'] == name for r in rows for m in r['matches'])
        per_class[name] = {'truth_instances': total, 'matched_instances': matched,
            'instance_recall': matched / total if total else None,
            'matched_precision': matched / predicted if predicted else 0.,
            'unmatched_predictions': predicted - matched,
            'planned_hits': sum(r['planned_instance_hit'] for r in rows if r['category'] == name),
            'planned_frames': sum(r['category'] == name for r in rows)}
    result['per_class'] = per_class
    result['miss_reasons'] = dict(Counter(m['reason'] for r in rows for m in r['misses']))
    return result
