"""Import explicit observations only; never infer or approve missing decisions."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.build_small_scale_fp_review import DEST, prior
from scripts.vision import small_scale_fp_observations as notes


def validate(evidence, observations):
    images = evidence['images']
    if len({g['image_id'] for g in images}) != len(images):
        raise ValueError('Duplicate image identity')
    if set(observations) != {g['image_id'] for g in images}:
        raise ValueError('Missing or extra image review')
    events = set()
    decisions = []
    for g in images:
        expected = {e['box_index'] for e in g['events']}
        if len(expected) != len(g['events']) or set(observations[g['image_id']]) != expected:
            raise ValueError('Missing or duplicate box review')
        for e in g['events']:
            if e['event_id'] in events: raise ValueError('Duplicate prediction event')
            events.add(e['event_id'])
            category, reason = observations[g['image_id']][e['box_index']]
            if category not in {'cabinet_like','block_facade','mixed_structure','ground_shadow','pole','unknown'}:
                raise ValueError('Unknown category')
            if not reason.strip(): raise ValueError('Missing explicit reason')
            decisions.append(dict(**e, image_id=g['image_id'], image_sha256=g['source']['image_sha256'],
                                  visual_structure=category, reason=reason,
                                  asset_identity_status='not_certified_by_visual_review',
                                  review_nature='AI辅助审核',
                                  status='pending' if category == 'unknown' else 'visual_content_described'))
    if len(events) != evidence['predictions'] or len(images) != evidence['unique_images']:
        raise ValueError('Evidence count mismatch')
    return decisions


def run():
    ep = DEST/'evidence.json'; evidence = prior.read(ep); prior.verify(evidence)
    decisions = validate(evidence, notes.OBSERVATIONS)
    deps = [ep, Path(notes.__file__).resolve(), Path(__file__).resolve()]
    for d in decisions:
        d['page_sha256'] = prior.file_sha256(d['page_path'])
    counts = Counter(d['visual_structure'] for d in decisions)
    path = DEST/'review.json'
    if path.exists():
        r = prior.read(path); prior.verify(r); return r
    return prior.frozen(path, dict(status='explicit_visual_review_complete' if not counts['unknown'] else 'review_pending',
        recorded_at=datetime.now(timezone.utc).isoformat(), decisions=decisions,
        counts=dict(counts), unique_images=evidence['unique_images'],
        limitations=['Prediction events are not independent structures or scenes.',
                    'Visual appearance does not certify simulation asset identity.',
                    'Historical numerical summary remains unchanged.'],
        inputs={str(p):prior.file_sha256(p) for p in deps}))


if __name__ == '__main__':
    r = run(); print(r['status'], len(r['decisions']), r['counts'])
