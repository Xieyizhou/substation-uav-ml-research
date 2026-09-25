"""Measure expected switchgear hits without inventing cabinet ground truth."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.analyze_recovery_paired_calibration import iou, indexed


def main():
    base = ROOT / 'data/research/ml_training_recovery_v1'
    prediction = base / 'control-baseline-v1/report.json'
    manifest = base / 'paired-calibration-v1/plan-manifest.json'
    inputs = {str(p):file_sha256(p) for p in (prediction, manifest, Path(__file__))}
    captures = {}
    for run in json.loads(manifest.read_text())['runs']:
        if run['mode'] != 'full_2d':
            continue
        plan_path = Path(run['plan_path'])
        receipt_path = base / 'paired-calibration-v1' / run['name'] / 'capture/collection-receipt.json'
        for p in (plan_path, receipt_path): inputs[str(p)] = file_sha256(p)
        plan = json.loads(plan_path.read_text())
        objects = {o['name']:o for o in plan['objects']}
        for row in json.loads(receipt_path.read_text())['views']:
            captures[(run['source_plan_identity'],row['view_id'])] = (row, objects)
    results = []
    for frame in json.loads(prediction.read_text())['frames']:
        source = frame['source']
        if source['expected_category'] != 'switchgear': continue
        row, objects = captures[(source['source_plan_identity'],source['view_id'])]
        labels = set(map(int, objects[source['expected_object_id']]['runtime_labels']))
        truth = indexed(row['truth']['objects'])
        targets = [o for label in labels for o in truth.get(label, [])]
        if len(targets) != 1: raise ValueError('Expected one bound switchgear target')
        overlaps = [{**p, 'iou':iou(p['bbox_xyxy'],targets[0]['bbox_xyxy'])} for p in frame['predictions']]
        results.append({'display_index':frame['display_index'], 'map_id':source['map_id'],
                        'target_bbox':targets[0]['bbox_xyxy'],
                        'correct_class_hit':any(p['iou'] >= .5 and p['class_name']=='switchgear' for p in overlaps),
                        'any_class_localization_hit':any(p['iou'] >= .5 for p in overlaps),
                        'predictions':overlaps})
    report = {'inputs':inputs, 'switchgear_results':results, 'matching_iou':.5,
              'switchgear_views':len(results), 'correct_class_hits':sum(r['correct_class_hit'] for r in results),
              'any_class_localization_hits':sum(r['any_class_localization_hit'] for r in results),
              'cabinet_visual_review':{'reviewer':'assistant', 'reviewed_display_indices':[1,2,6,8,10,12],
                 'clear_cabinet_false_positive_indices':[12], 'ambiguous_large_box_indices':[10],
                 'no_cabinet_prediction_indices':[1,2,6,8],
                 'method':'Visual attribution on prediction contact sheet; no cabinet ground-truth boxes available.',
                 'note':'Index 10 spans cabinet, background and left-side equipment; do not count as confirmed cabinet detection.'},
              'training_admitted':False,
              'limits':['Small development diagnostic, not population error estimates or model qualification.',
                        'Correct-class misses include wrong-class localization; these are reported separately.',
                        'Cabinet-attributed counts are not whole-frame false-positive rates.']}
    report['identity']=object_sha256(report)
    write_json(base/'control-baseline-v1/summary.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs','switchgear_results')},indent=2))


if __name__ == '__main__': main()
