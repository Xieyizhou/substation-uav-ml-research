"""Bind existing low-light review and full labels to pool members."""
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior
from scripts.vision import physical_low_light_dataset as source


def run():
    paths = [OUT/'protocol.json', source.OUT/'capture-protocol.json', source.OUT/'quality-review.json', source.OUT/'export/manifest.json']
    p, capture, review, manifest = [prior.read(path) for path in paths]
    for r in (p, capture, review, manifest): prior.verify(r)
    source.validate_review(capture, review)
    pool = {m['member_id']:m for m in p['members']}; rows = []
    for m in manifest['members']:
        current = pool[m['member_id']]
        for kind in ['image', 'label']:
            if current[kind+'_sha256'] != m[kind+'_sha256'] or prior.file_sha256(current[kind+'_path']) != m[kind+'_sha256']:
                raise ValueError('Member identity changed')
            paths.append(Path(current[kind+'_path']))
        if current['class_instances'] != m['class_instances']: raise ValueError('Full class counts changed')
        rows.append(dict(member_id=m['member_id'], review_path=str(source.OUT/'quality-review.json'),
            evidence_path=m['evidence_path'], image_sha256=m['image_sha256'], label_sha256=m['label_sha256'],
            status='existing_low_light_explicit_review_validated', training_eligible=False,
            limitation='Existing review link, not final new-dataset admission.'))
    paths.extend([Path(__file__).resolve(), Path(source.__file__).resolve()])
    dest = OUT/'low-light-review-links.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    return prior.frozen(dest, dict(status='32_low_light_review_links_verified', members=rows,
        inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__ == '__main__': print(run()['status'])
