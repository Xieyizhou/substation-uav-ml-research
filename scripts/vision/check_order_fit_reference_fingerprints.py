"""Metadata-only reference fingerprint matching; no protected image/label reads."""
import json
from collections import defaultdict
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior
from scripts.vision.freeze_visual_augmentation_240_v2 import REFERENCE_INDEXES


def run():
    pp = OUT/'development-exact-overlap-v1.json'; pool = prior.read(pp); prior.verify(pool)
    files, pixels = defaultdict(list), defaultdict(list)
    for path in REFERENCE_INDEXES:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            row = json.loads(line)
            ref = dict(index=str(path), line=line_number,
                recorded_role=row.get('role', 'not_explicitly_encoded'),
                recorded_path=row.get('verified_path', row.get('image_path')))
            if row.get('image_sha256'): files[row['image_sha256']].append(ref)
            if row.get('pixel_sha256'): pixels[row['pixel_sha256']].append(ref)
    matches = []
    for member in pool['members']:
        hits = files.get(member['image_sha256'], []) + pixels.get(member['pixel_sha256'], [])
        unique = {(r['index'], r['line']):r for r in hits}
        if unique:
            matches.append(dict(member_id=member['member_id'], references=list(unique.values()),
                status='requires_reference_role_resolution_not_automatic_training_exclusion'))
    dest = OUT/'reference-fingerprint-overlap-v1.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    deps = [pp, *REFERENCE_INDEXES, Path(__file__).resolve()]
    return prior.frozen(dest, dict(status='reference_matches_require_role_resolution' if matches else 'no_indexed_exact_reference_match',
        pool_members_checked=len(pool['members']), matches=matches,
        protected_images_read=False, protected_labels_read=False, dataset_ready=False,
        scope='Exact hashes in existing indexes only; no claim of complete lineage isolation or index completeness.',
        inputs={str(path):prior.file_sha256(path) for path in deps}))


if __name__ == '__main__':
    r = run(); print(r['status'], r['pool_members_checked'], len(r['matches']))
