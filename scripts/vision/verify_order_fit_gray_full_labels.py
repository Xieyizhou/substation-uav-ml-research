"""Exact serialized full-label correspondence for gray review events."""
from pathlib import Path
from PIL import Image
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior
from scripts.vision import import_transfer_pilot_review as pilot
from scripts.vision import import_transfer_expansion_review as expansion
from scripts.vision.export_material_candidate_batch import label_text


def run():
    pp = OUT/'protocol.json'; p = prior.read(pp); prior.verify(p)
    idx = {m['member_id']:m for m in p['members']}; deps = [pp, Path(__file__).resolve()]; rows = []
    for module in (pilot, expansion):
        ep = module.DEST/'evidence.json'; rp = module.DEST/'label-review.json'
        e, review = prior.read(ep), prior.read(rp)
        prior.verify(e); prior.verify(review)
        if not module.validate(e, review['decisions']): raise ValueError('Incomplete review')
        deps.extend([ep, rp])
        for page in e['pages']:
            if page['condition'] != 'gray_all_body': continue
            m = idx[page['source_id']+'-gray_all_body']
            events = [x for x in e['events'] if x['source_id'] == page['source_id'] and x['condition'] == 'gray_all_body']
            ids = [x['object_id'] for x in events]
            if len(ids) != len(set(ids)): raise ValueError('Duplicate instance')
            with Image.open(m['image_path']) as image: width, height = image.size
            actual = Path(m['label_path']).read_text()
            if actual != label_text({'objects':[x['truth'] for x in events]}, width, height):
                raise ValueError('Complete serialized labels differ')
            if len(events) != len(m['truth']): raise ValueError('Truth count differs')
            for i, (event, truth) in enumerate(zip(events, m['truth'])):
                if truth['label_line_index'] != i or truth['class_name'] != event['truth']['class_name'] or max(abs(a-b) for a,b in zip(truth['bbox_xyxy'],event['truth']['bbox_xyxy'])) > 1e-5:
                    raise ValueError('Pool truth differs')
            rows.append(dict(member_id=m['member_id'], object_ids=ids, labels_checked=len(events),
                complete_label_bytes_equal=True, training_eligible=False))
            deps.extend([Path(m['image_path']), Path(m['label_path'])])
    if len(rows) != 12: raise ValueError('Incomplete cohort')
    dest = OUT/'gray-full-label-correspondence.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    return prior.frozen(dest, dict(status='twelve_gray_full_labels_correspond', members=rows,
        limitation='Correspondence only, not full-scene completeness or visual quality approval.',
        inputs={str(path):prior.file_sha256(path) for path in deps}))


if __name__ == '__main__':
    r = run(); print(len(r['members']), sum(m['labels_checked'] for m in r['members']))
