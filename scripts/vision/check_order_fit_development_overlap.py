"""Exact RGB/file exclusion against already-viewed development images only."""
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, TRAIN, prior
from scripts.vision.freeze_visual_augmentation_240_v2 import pixel_hash


def run():
    pp, dp = OUT/'protocol.json', TRAIN/'design.json'
    pool, design = prior.read(pp), prior.read(dp)
    prior.verify(pool); prior.verify(design)
    deps = [pp, dp, Path(__file__).resolve()]; refs = []
    for kind in ['paired_review', 'negative_review']:
        path = Path(design['evaluation'][kind]); record = prior.read(path); prior.verify(record); deps.append(path)
        for frame in record['frames']:
            ip = Path(frame['image_path'])
            if prior.file_sha256(ip) != frame['image_sha256']: raise ValueError('Development image drift')
            refs.append(dict(role=kind, image_path=str(ip), image_sha256=frame['image_sha256'], pixel_sha256=pixel_hash(ip)))
            deps.append(ip)
    matches = []; rows = []
    for member in pool['members']:
        ip = Path(member['image_path']); digest = prior.file_sha256(ip)
        if digest != member['image_sha256']: raise ValueError('Pool image drift')
        pixels = pixel_hash(ip); deps.append(ip)
        hits = [r for r in refs if r['image_sha256'] == digest or r['pixel_sha256'] == pixels]
        rows.append(dict(member_id=member['member_id'], image_sha256=digest, pixel_sha256=pixels,
            exact_development_overlap=bool(hits)))
        if hits: matches.append(dict(member_id=member['member_id'], development_matches=hits))
    dest = OUT/'development-exact-overlap-v1.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    return prior.frozen(dest, dict(status='exact_overlap_found' if matches else 'no_exact_file_or_rgb_overlap',
        members=rows, development_images=len(refs), matches=matches,
        limitation='Not a pose/lineage/assets independence test. Protected references not checked here; no protected labels read.',
        dataset_ready=False, inputs={str(path):prior.file_sha256(path) for path in deps}))


if __name__ == '__main__':
    r = run(); print(r['status'], len(r['members']), r['development_images'], len(r['matches']))
