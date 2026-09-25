"""Import authored pilot observations; this is not a training admission gate."""
from datetime import datetime, timezone
from pathlib import Path
from src.ml.artifacts import object_sha256
from scripts.vision.build_transfer_pilot_review import OUT, prior

DEST = OUT / 'review'
OBS = Path(__file__).with_name('transfer_pilot_observations.json')


def validate(evidence, decisions):
    events = {e['event_id']: e for e in evidence['events']}
    if len(events) != len(evidence['events']):
        raise ValueError('Duplicate evidence identity')
    if len(decisions) != len(events) or {d['event_id'] for d in decisions} != set(events):
        raise ValueError('Missing or duplicate review')
    for d in decisions:
        e = events[d['event_id']]
        if d['event_sha256'] != object_sha256(e) or d['object_id'] != e['object_id']:
            raise ValueError('Stale evidence or instance identity')
        for path, digest in [('image_path','image_sha256'), ('crop_path','crop_sha256'),
                             ('page_path','page_sha256'), ('source_receipt','source_receipt_sha256'),
                             ('replay_receipt','replay_receipt_sha256')]:
            if prior.file_sha256(e[path]) != e[digest]:
                raise ValueError('Changed evidence bytes')
        if not d['reason'].strip() or not d['reviewed_at'] or d['review_nature'] != 'AI-assisted':
            raise ValueError('Incomplete explicit review')
        if d['status'] not in ('diagnostic_content_reviewed', 'unknown'):
            raise ValueError('Unsupported review status')
        if any(d[k] for k in ('training_admitted', 'promotable', 'pixel_visibility_certified')):
            raise ValueError('Unsupported certification')
    return all(d['status'] == 'diagnostic_content_reviewed' for d in decisions)


def main():
    ep = DEST / 'evidence.json'
    e = prior.read(ep); prior.verify(e)
    rp = DEST / 'label-review.json'
    if rp.exists():
        r = prior.read(rp); prior.verify(r); validate(e, r['decisions']); return r
    observations = prior.read(OBS)
    expected = {f"{p['source_id']}-{p['condition']}": p['label_count'] for p in e['pages']}
    if set(observations) != set(expected) or any(len(observations[k]) != n for k,n in expected.items()):
        raise ValueError('Authored observation coverage mismatch')
    decisions = []
    for event in e['events']:
        key, index = event['event_id'].rsplit('-',1)
        decisions.append(dict(event_id=event['event_id'], object_id=event['object_id'],
            event_sha256=object_sha256(event), reason=observations[key][int(index)],
            status='diagnostic_content_reviewed', review_nature='AI-assisted',
            reviewed_at=datetime.now(timezone.utc).isoformat(), training_admitted=False,
            promotable=False, pixel_visibility_certified=False))
    validate(e, decisions)
    paths = [ep, OBS, Path(__file__).resolve()]
    return prior.frozen(rp, dict(status='all_label_observations_recorded_not_training_admitted',
        decisions=decisions, unique_frames=len(e['pages']), independent_source_poses=4,
        full_scene_source_audit_complete=False, expansion_authorized_by_this_receipt=False,
        training_ready=False, training_started=False,
        limits=['Visual review is not pixel visibility certification.',
                'Variant images are not independent source poses.',
                'Full-scene source audit and inference remain separate gates.'],
        inputs={str(p):prior.file_sha256(p) for p in paths}))


if __name__ == '__main__':
    r = main(); print(r['status'], len(r['decisions']))
