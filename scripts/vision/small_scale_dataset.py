"""Lossless reviewed small-scale export; never reads protected labels."""
import json
from collections import Counter
from pathlib import Path
from PIL import Image
from scripts.vision.small_scale_material_capture import OUT, freeze, prior
from scripts.vision.neutral_gray_control import OUT as REF
from scripts.vision.record_small_scale_review import main as review
from scripts.vision.verify_small_scale_material import verify_capture
from scripts.vision.freeze_visual_augmentation_240_v2 import pixel_hash, REFERENCE_INDEXES
from scripts.vision.export_material_candidate_batch import label_text


def export():
    p = freeze(); review()
    refpath = REF/'design.json'; old = prior.read(refpath); prior.verify(old)
    dest = OUT/'export/manifest.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    refs = list(old['pool_rows'])
    deps = [refpath, OUT/'capture-protocol.json', OUT/'quality-review.json', Path(__file__).resolve()]
    for name in ('paired_review', 'negative_review'):
        path = Path(old['evaluation'][name]); r = prior.read(path); prior.verify(r)
        refs += r['frames']; deps.append(path)
    hashes, pixels = set(), set()
    for r in refs:
        ip = Path(r['image_path']); sha = prior.file_sha256(ip)
        if sha != r['image_sha256']: raise ValueError('Reference image drift')
        hashes.add(sha); pixels.add(pixel_hash(ip)); deps.append(ip)
    for index in REFERENCE_INDEXES:
        deps.append(index)
        for line in index.read_text().splitlines():
            r = json.loads(line)
            hashes.add(r.get('image_sha256')); pixels.add(r.get('pixel_sha256'))
    members, seen, paired = [], set(), {}
    for u in p['units']:
        row, mapping, receipt = verify_capture(u)
        ip = Path(row['rgb_path']); px = pixel_hash(ip)
        if row['image_sha256'] in hashes or px in pixels or px in seen:
            raise ValueError('Exact existing/candidate overlap: '+u['unit_id'])
        seen.add(px)
        im = Image.open(ip).convert('RGB')
        text = label_text(row['truth'], *im.size)
        if paired.setdefault(u['pair_id'], text) != text:
            raise ValueError('Full paired labels differ')
        member = 'small-material-v1-'+u['unit_id']
        image = OUT/'export/images'/f'{member}.png'
        label = OUT/'export/labels'/f'{member}.txt'
        image.parent.mkdir(parents=True, exist_ok=True); label.parent.mkdir(parents=True, exist_ok=True)
        if image.exists():
            if pixel_hash(image) != px: raise ValueError('Export image drift')
        else: im.save(image)
        if label.exists():
            if label.read_text() != text: raise ValueError('Export label drift')
        else: label.write_text(text)
        if pixel_hash(image) != px: raise ValueError('Lossless pixel verification failed')
        members.append(dict(member_id=member, unit_id=u['unit_id'], subset='bridge_positive',
            variant=u['variant'], material_setting=u['variant'], illumination=u['illumination'],
            pair_id=u['pair_id'], lineage_id=u['pair_id'], lineage_resolution='saved_layout_asset_instance_pose',
            layout=u['layout'], planned_category=u['view']['category'],
            image_path=str(image), image_sha256=prior.file_sha256(image), pixel_sha256=px,
            label_path=str(label), label_sha256=prior.file_sha256(label),
            source_image_path=str(ip), source_receipt=str(receipt), source_plan=u['plan_path'],
            source_world=str(Path(u['plan_path']).parent/'world.sdf'), actual_pose=row['actual_pose'],
            instance_mapping=mapping, full_truth=row['truth'],
            class_instances=dict(Counter(t['class_name'] for t in row['truth']['objects'])),
            data_role='bounded_development_training_candidate', source_independent=False,
            asset_independence=False, training_admitted=False, promotable=False))
        deps += [image, label, ip, receipt, OUT/'replays'/u['unit_id']/'completion.json',
                 OUT/'evidence'/u['unit_id']/'evidence.json', Path(u['plan_path'])]
    if len(members) != 16 or len(paired) != 4: raise ValueError('Incomplete quartet')
    return prior.frozen(dest, dict(status='16_reviewed_lossless_small_scale_candidates', members=members,
        reference_images_checked=len(refs), protected_labels_read=False,
        pose_groups=4, layouts=len({u['layout'] for u in p['units']}), shared_assets=True,
        source_independence_claim=False,
        derivation_policy='Four variants per saved new pose remain in one training lineage. Layouts rearrange known assets; new pixel hashes are not independent scene evidence.',
        inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__ == '__main__': print(export()['status'])
