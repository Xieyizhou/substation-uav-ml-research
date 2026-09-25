"""Build review evidence and numerical gates; never invent review decisions."""
from pathlib import Path
from PIL import Image, ImageDraw
from scripts.vision.run_visibility_repair_training_v2 import OUT as SOURCE, KEYS, read, save, file_sha256
from scripts.vision.finalize_exposure_diagnosis import aggregate, policy_checks

OUT = SOURCE / 'post-training-audit-v1'

def main():
    OUT.mkdir(exist_ok=True)
    p = read(SOURCE / 'protocol.json')
    negative_path = Path(p['evaluation']['negative_review'])
    images = {(r['view_id'], r['variant']): r for r in read(negative_path)['frames']}
    records = {}; events = []; inputs = {str(negative_path): file_sha256(negative_path)}
    for key in KEYS:
        path = SOURCE / f'evaluation-{key}.json'
        r = read(path); records[key] = r; inputs[str(path)] = file_sha256(path)
        if r['status'] != 'complete' or r['matching_conflicts']:
            raise ValueError('Incomplete or conflicting evaluation')
        for row in r['negative_rows']:
            for index, box in enumerate(row['predictions']):
                source = images[row['view_id'], row['variant']]; ip = Path(source['image_path'])
                if file_sha256(ip) != row['image_sha256']:
                    raise ValueError('Image hash changed')
                inputs[str(ip)] = row['image_sha256']
                im = Image.open(ip).convert('RGB'); overlay = im.copy()
                ImageDraw.Draw(overlay).rectangle(box['bbox_xyxy'], outline='red', width=5)
                panel = Image.new('RGB', (1200, 550), 'white')
                overlay.thumbnail((800, 480)); panel.paste(overlay, (0, 55))
                crop = im.crop(tuple(box['bbox_xyxy'])); crop.thumbnail((390, 480)); panel.paste(crop, (805, 55))
                event = f'F{len(events)+1:02}'
                ImageDraw.Draw(panel).text((10,10), f'{event} {key} {box["class_name"]} {box["confidence"]:.4f}', fill='black')
                dest = OUT / f'{event}.png'; panel.save(dest)
                events.append(dict(event_id=event,cell=key,view_id=row['view_id'],variant=row['variant'],prediction_index=index,
                    image_path=str(ip),image_sha256=row['image_sha256'],prediction=box,evidence_path=str(dest),evidence_sha256=file_sha256(dest)))
    hist_path = Path(p['evaluation']['historical_reference']); hist = read(hist_path)
    inputs[str(hist_path)] = file_sha256(hist_path)
    groups = {arm: aggregate([records[f'{arm}-300-{s}'] for s in (7,17,27)]) for arm in ('original_only','three_variant')}
    checks = {arm: policy_checks(g,hist['aggregate']['R-300'],hist['historical_A'],p) for arm,g in groups.items()}
    save(OUT/'evidence.json', dict(status='pending_explicit_visual_decisions',events=events,aggregate=groups,policy_checks=checks,inputs=inputs))
    print(OUT)
    print(checks)

if __name__ == '__main__': main()
