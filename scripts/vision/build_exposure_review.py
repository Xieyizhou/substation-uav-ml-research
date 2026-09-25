"""Build hash-bound context/crop evidence, never review decisions."""
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw
from scripts.vision.exposure_protocol import BASE, OUT, PRIOR, read, save, file_sha256


def frame_key(frame):
    return frame['view_id'], frame['variant']


def prediction_id(seed, frame, index):
    return f'D-{seed}:{frame["view_id"]}:{frame["variant"]}:{index}'


def main():
    evaluation_path = PRIOR / 'development-evaluation.json'
    negative_path = BASE / 'hard-negative-isolated-v2/semantic-review.json'
    evaluation = read(evaluation_path)
    lookup = {frame_key(r): r for r in read(negative_path)['frames']}
    groups = defaultdict(list)
    for seed in (7, 17, 27):
        for frame in evaluation['results'][f'D_{seed}']['negative_rows']:
            for index, pred in enumerate(frame['predictions']):
                groups[frame_key(frame)].append({'prediction_id': prediction_id(seed, frame, index),
                                               'seed': seed, **pred})
    folder = OUT / 'review-evidence-v2'
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for number, ((view, variant), predictions) in enumerate(sorted(groups.items())):
        source = lookup[(view, variant)]
        if file_sha256(source['image_path']) != source['image_sha256']:
            raise ValueError('Source image changed')
        with Image.open(source['image_path']) as opened:
            original = opened.convert('RGB')
        canvas = Image.new('RGB', (960, 590 + 240 * ((len(predictions) + 2) // 3)), 'white')
        canvas.paste(original.resize((960, 540)), (0, 30))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f'Frame {number:02} {view[:16]} {variant} / {len(predictions)} predictions', fill='black')
        for i, pred in enumerate(predictions):
            box = pred['bbox_xyxy']
            draw.rectangle((box[0]/2, 30+box[1]/2, box[2]/2, 30+box[3]/2), outline='red', width=2)
            draw.text((box[0]/2, 30+box[1]/2), str(i), fill='red')
            crop = original.crop((max(0, int(box[0])), max(0, int(box[1])), min(1920, int(box[2])+1), min(1080, int(box[3])+1)))
            crop.thumbnail((310, 200))
            x, y = (i % 3)*320, 590 + (i//3)*240
            canvas.paste(crop, (x, y+30))
            draw.text((x+3, y), f'{i} seed={pred["seed"]} {pred["class_name"]} {pred["confidence"]:.3f}', fill='black')
        path = folder / f'{number:02}.png'
        canvas.save(path)
        frames.append({'number': number, 'view_id': view, 'variant': variant, 'image_path': source['image_path'],
                       'image_sha256': source['image_sha256'], 'evidence_path': str(path),
                       'evidence_sha256': file_sha256(path), 'predictions': predictions})
    result = save(OUT / 'review-manifest-v2.json', {'status': 'pending_visual_review', 'frames': frames,
         'prediction_count': sum(len(f['predictions']) for f in frames),
         'inputs': {str(p): file_sha256(p) for p in (evaluation_path, negative_path, Path(__file__))}})
    print(len(frames), result['prediction_count'])


if __name__ == '__main__':
    main()
