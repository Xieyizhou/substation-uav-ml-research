"""Resumable fixed-threshold and diagnostic inference on viewed development data."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PIL import Image
from scripts.vision.exposure_protocol import BASE, OUT, PRIOR, SEEDS, prepare, read, save, verify, file_sha256
from scripts.vision.run_exposure_diagnosis import checked_cell
from scripts.vision.evaluate_paired_visual_factors import checked_rows
from scripts.vision.evaluate_visual_augmentation_abcd import paired_truth
from scripts.vision.exposure_metrics import score, summary


def predict(model, path, confidence):
    with Image.open(path) as image:
        result = model.predict(image.convert('RGB'), imgsz=640, conf=confidence, iou=.7,
            rect=False, device='cpu', verbose=False, agnostic_nms=False, max_det=300)[0]
    return [{'bbox_xyxy': box, 'class_name': model.names[int(cls)], 'confidence': conf}
            for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())]


def main():
    from ultralytics import YOLO
    protocol = prepare()
    reviewed, reviewed_path = checked_rows()
    paired, receipt_inputs = paired_truth(reviewed)
    negative_path = BASE / 'hard-negative-isolated-v2/semantic-review.json'
    negative = read(negative_path)
    if negative['status'] != 'reviewed' or negative['accepted'] != 48 or negative['held']:
        raise ValueError('Negative review incomplete')
    inputs = {str(p): file_sha256(p) for p in (Path(__file__), OUT/'protocol.json', reviewed_path, negative_path,
               ROOT/'scripts/vision/exposure_metrics.py')}
    inputs.update(receipt_inputs)
    for row in reviewed + negative['frames']:
        if row['decision'] != 'accepted' or file_sha256(row['image_path']) != row['image_sha256']:
            raise ValueError('Image or review changed')
        inputs[row['image_path']] = row['image_sha256']
    weights = {}
    for arm in ('A', 'D'):
        for seed in SEEDS:
            completion = PRIOR / f'arm-{arm}-seed-{seed}/completion.json'
            record = read(completion)
            if file_sha256(record['weights']) != record['weights_sha256']:
                raise ValueError('Historical weights changed')
            weights[f'historical-{arm}-{seed}'] = (record['weights'], completion)
    for key in protocol['schedules']:
        completion = OUT / key / 'completion.json'
        if completion.exists():
            record = checked_cell(completion, protocol)
            weights[key] = (record['weights'], completion)
    outputs = {}
    for key, (weight, completion) in weights.items():
        path = OUT / 'evaluation' / f'{key}.json'
        source_inputs = {**inputs, str(weight): file_sha256(weight), str(completion): file_sha256(completion)}
        if path.exists():
            record = read(path)
            verify(record)
            if record['inputs'] != source_inputs:
                raise ValueError('Cached inference inputs differ')
        else:
            print(f'EVALUATE {key}', flush=True)
            model = YOLO(weight)
            rows = [score(row, truth, predict(model, row['image_path'], .37), predict(model, row['image_path'], .001)) for row, truth in paired]
            negatives = []
            for row in negative['frames']:
                formal = predict(model, row['image_path'], .37)
                low = predict(model, row['image_path'], .001)
                negatives.append({'view_id': row['view_id'], 'variant': row['variant'], 'image_sha256': row['image_sha256'],
                                  'predictions': formal, 'low_predictions': low, 'frame_has_prediction': bool(formal)})
            deltas = []
            for pair_id in sorted({r['pair_id'] for r in rows}):
                group = {r['variant']: r for r in rows if r['pair_id'] == pair_id}
                for variant in ('material', 'background', 'lighting'):
                    original, other = group['original'], group[variant]
                    deltas.append({'pair_id': pair_id, 'variant': variant,
                        'planned_hit_delta': int(other['planned_instance_hit'])-int(original['planned_instance_hit']),
                        'matched_instance_delta': len(other['matches'])-len(original['matches']),
                        'unmatched_prediction_delta': other['unmatched_prediction_count']-original['unmatched_prediction_count']})
            record = save(path, {'status': 'complete', 'cell': key, 'inputs': source_inputs,
                'rows': rows, 'negative_rows': negatives, 'paired_deltas': deltas,
                'summary': {v: summary([r for r in rows if r['variant'] == v]) for v in ('original', 'material', 'background', 'lighting')},
                'negative_summary': {'frame_false_positive_rate': sum(r['frame_has_prediction'] for r in negatives)/48,
                                     'unmatched_predictions': sum(len(r['predictions']) for r in negatives)},
                'matching_conflicts': sum(r['matching_conflict'] for r in rows),
                'limits': ['Low-threshold predictions remain bounded by NMS and max_det=300; absence is not proof of no network candidate.']})
            del model
        outputs[str(path)] = file_sha256(path)
    save(OUT / 'evaluation-progress.json', {'status': 'complete' if len(outputs) == 24 else 'in_progress', 'inputs': outputs,
                                          'completed_evaluations': len(outputs)})


if __name__ == '__main__':
    main()
