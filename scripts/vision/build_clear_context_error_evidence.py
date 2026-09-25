"""Freeze current-round FP pages without producing review decisions."""
from pathlib import Path
from PIL import Image, ImageDraw
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision.freeze_clear_context_training import OUT
from scripts.vision.prepare_clear_context_increment import checked
from scripts.vision.evaluate_reactor_visibility_expansion import _load_inputs


def run():
    source = OUT / 'evaluation-v1/summary.json'
    summary = checked(source)
    root = OUT / 'evaluation-v1/error-review-v1'
    target = root / 'evidence.json'
    if target.exists():
        return checked(target)
    root.mkdir(parents=True, exist_ok=True)
    _, negatives, _ = _load_inputs()
    by_hash = {r['image_sha256']: r for r in negatives}
    groups = {}
    for i, event in enumerate(summary['negative_fp_review_queue']):
        groups.setdefault(event['image_sha256'], []).append(dict(event, event_id=f'fp-{i:02}'))
    frames = []
    for i, (digest, events) in enumerate(sorted(groups.items())):
        row = by_hash[digest]
        if file_sha256(row['image_path']) != digest:
            raise ValueError('Stale source image')
        im = Image.open(row['image_path']).convert('RGB')
        overlay = im.copy()
        draw = ImageDraw.Draw(overlay)
        page = Image.new('RGB', (1920, 1140 + 350*((len(events)+3)//4)), 'white')
        pd = ImageDraw.Draw(page)
        for j, event in enumerate(events):
            pred = event['prediction']; box = pred['bbox_xyxy']
            draw.rectangle(box, outline='red', width=3)
            draw.text((box[0], box[1]), event['event_id'], fill='white', stroke_width=1, stroke_fill='black')
            crop = im.crop(tuple(map(int, box))); crop.thumbnail((470, 310))
            x, y = (j % 4)*480, 1140+(j//4)*350
            page.paste(crop, (x, y+30))
            pd.text((x, y), f"{event['event_id']} seed {event['seed']} {pred['class_name']} {pred['confidence']:.3f}", fill='black')
        page.paste(overlay, (0, 40))
        pd.text((10, 10), f"frame-{i:02} {row['view_id']} {row['variant']}", fill='black')
        path = root/f'frame-{i:02}.png'
        if path.exists():
            raise ValueError('Unreceipted evidence exists; preserve and investigate')
        page.save(path)
        frames.append(dict(frame_id=f'frame-{i:02}', image_path=row['image_path'], image_sha256=digest,
                           evidence_path=str(path.resolve()), evidence_sha256=file_sha256(path), events=events))
    deps = [source, Path(__file__)]
    return write_record(target, dict(status='evidence_only_review_pending', frames=frames,
        training_admitted=False, promotable=False, inputs={str(p.resolve()):file_sha256(p) for p in deps}))


if __name__ == '__main__':
    print(len(run()['frames']))
