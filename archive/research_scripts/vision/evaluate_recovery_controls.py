"""Frozen-baseline predictions on the twelve development control views."""
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json


def main():
    from ultralytics import YOLO
    base = ROOT / 'data/research/ml_training_recovery_v1'
    contract_path = ROOT / 'config/perception/visual_experiment_baseline_v1.json'
    contract = json.loads(contract_path.read_text())
    weights = list((ROOT / contract['model']['package_root']).rglob('*.pt'))
    weights = [p for p in weights if file_sha256(p) == contract['model']['weights_sha256']]
    if len(weights) != 1:
        raise ValueError('Expected exactly one frozen weight artifact')
    source = base / 'paired-calibration-v1/analysis/pairs.jsonl'
    rows = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    rows = [r for r in rows if not r['held_target_view']]
    assert len(rows) == 12
    inputs = {str(p): file_sha256(p) for p in (source, contract_path, weights[0], Path(__file__))}
    model = YOLO(str(weights[0]))
    assert [model.names[i] for i in range(4)] == contract['class_order']
    out = base / 'control-baseline-v1'
    out.mkdir(exist_ok=True)
    frames = []
    sheet = Image.new('RGB', (1280, 2400), '#202020')
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        path = Path(row['full_rgb_path'])
        inputs[str(path)] = file_sha256(path)
        assert inputs[str(path)] == row['full_image_sha256']
        with Image.open(path) as im:
            result = model.predict(im.convert('RGB'), imgsz=640, conf=.37, iou=.7,
                                   batch=1, rect=False, device='cpu', verbose=False,
                                   agnostic_nms=False, max_det=300)[0]
            x, y = (index % 2)*640, (index // 2)*400
            sheet.paste(im.resize((640, 360)), (x, y+40))
        predictions = []
        draw.text((x+4, y+4), f'{index+1}: {row["map_id"]} {row["expected_object_id"]}', fill='white')
        for box, score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
            label = model.names[int(cls)]
            predictions.append({'bbox_xyxy': box, 'confidence': score, 'class_name': label})
            a,b,c,d=box
            draw.rectangle((x+a/3,y+40+b/3,x+c/3,y+40+d/3), outline='#ffb347', width=2)
            draw.text((x+a/3,y+40+b/3), f'{label} {score:.2f}', fill='white')
        frames.append({'display_index': index+1, 'source': row, 'predictions': predictions})
    visual = ROOT / 'outputs/research/ml_training_recovery_v1/control-baseline-v1.jpg'
    sheet.save(visual, quality=95)
    report = {'schema_version':1, 'inputs':inputs, 'frames':frames,
              'settings':{'imgsz':640, 'confidence':.37, 'nms_iou':.7, 'rect':False, 'device':'cpu', 'batch':1,
                          'agnostic_nms':False, 'max_det':300},
              'ultralytics_version': __import__('ultralytics').__version__,
              'visual':{'path':str(visual), 'sha256':file_sha256(visual)},
              'training_admitted':False, 'status':'predictions_complete_pending_visual_attribution',
              'limits':['Twelve development diagnostic views, not independent validation.',
                        'Cabinet has no target box; attribution requires visual review.',
                        'PyTorch diagnostic inference is not full deployed runtime equivalence.']}
    report['identity'] = object_sha256(report)
    write_json(out / 'report.json', report)
    print(json.dumps({'frames':len(frames), 'predictions':sum(len(r['predictions']) for r in frames), 'visual':str(visual)}))


if __name__ == '__main__':
    main()
