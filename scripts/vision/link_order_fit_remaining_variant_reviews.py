"""Resolve diagnostic review links; never upgrade them into quality approval."""
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior
from scripts.vision import import_transfer_pilot_review as pilot
from scripts.vision import import_transfer_expansion_review as expansion
from scripts.vision.freeze_visual_augmentation_240_v2 import pixel_hash


def run():
    pp = OUT/'protocol.json'; sp = OUT/'source-resolution.json'
    p, resolution = prior.read(pp), prior.read(sp)
    prior.verify(p); prior.verify(resolution)
    idx = {m['member_id']:m for m in p['members']}; deps = [pp, sp, Path(__file__).resolve()]; rows = []
    for module in (pilot, expansion):
        ep, rp = module.DEST/'evidence.json', module.DEST/'label-review.json'
        e, r = prior.read(ep), prior.read(rp)
        prior.verify(e); prior.verify(r)
        if not module.validate(e, r['decisions']): raise ValueError('Unknown variant review')
        deps.extend([ep, rp, Path(module.__file__).resolve()])
        for page in e['pages']:
            if page['condition'] != 'gray_all_body': continue
            mid = page['source_id']+'-gray_all_body'; m = idx[mid]
            events = [x for x in e['events'] if x['source_id'] == page['source_id'] and x['condition'] == 'gray_all_body']
            if len(events) != len(m['truth']): raise ValueError('Full label coverage differs')
            if pixel_hash(events[0]['image_path']) != pixel_hash(m['image_path']): raise ValueError('Variant pixels differ')
            deps.extend([Path(m['image_path']), Path(m['label_path'])])
            rows.append(dict(member_id=mid, review_path=str(rp), event_ids=[x['event_id'] for x in events],
                status='diagnostic_visual_review_linked_not_quality_approval',
                full_label_count=len(events), training_eligible=False))
    for row in resolution['resolved_sources']:
        m = idx[row['member_id']]
        if len(row['review_decisions']) != len(m['truth']): raise ValueError('Lighting review coverage differs')
        if pixel_hash(row['actual_render_image']) != pixel_hash(m['image_path']): raise ValueError('Lighting pixels changed')
        deps.extend([Path(row['actual_render_image']), Path(m['image_path']), Path(m['label_path'])])
        rows.append(dict(member_id=m['member_id'], review_source=str(sp),
            status='existing_physical_lighting_review_with_limits_linked', training_eligible=False))
    if len(rows) != 16 or len({r['member_id'] for r in rows}) != 16: raise ValueError('Expected 16 variants')
    dest = OUT/'remaining-variant-review-links.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    return prior.frozen(dest, dict(status='sixteen_variant_review_links_not_admission', members=rows,
        limitation='Diagnostic content review does not certify full-scene coverage or dataset eligibility.',
        inputs={str(path):prior.file_sha256(path) for path in deps}))


if __name__ == '__main__': print(run()['status'])
