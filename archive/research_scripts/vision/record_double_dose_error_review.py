"""Import explicit per-box observations; never approve training or alter labels."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from src.ml.artifacts import object_sha256
from scripts.vision.build_double_dose_error_review import DEST, prior
from scripts.vision.double_dose_review_observations import OBS


def expected(e):
    result = {}
    for x in e['events']:
        for j in range(len(x['events']) if x['kind'] == 'FP' else 1):
            key = f"{x['event_id']}:{j}"
            if key in result:
                raise ValueError('Duplicate evidence identity')
            result[key] = (x, j)
    return result


def validate(e, decisions):
    want = expected(e)
    if len(decisions) != len(want) or {d['decision_id'] for d in decisions} != set(want):
        raise ValueError('Missing or duplicate explicit decision')
    for d in decisions:
        x, j = want[d['decision_id']]
        if d['image_sha256'] != x['source']['image_sha256'] or d['evidence_sha256'] != x['page_sha256']:
            raise ValueError('Stale image or evidence identity')
        if prior.file_sha256(x['page_path']) != d['evidence_sha256'] or prior.file_sha256(x['source']['image_path']) != d['image_sha256']:
            raise ValueError('Changed image/evidence bytes')
        if d['truth'] != x.get('truth') or d['prediction'] != (x['events'][j] if x['kind'] == 'FP' else None):
            raise ValueError('Changed truth or prediction')
        if d['source_identity'] != object_sha256(x['source']) or d['event_identity'] != object_sha256(x['events']):
            raise ValueError('Changed source or comparison events')
        if not d['reason'] or not d['reviewed_at'] or d['review_nature'] != 'AI-assisted':
            raise ValueError('Incomplete review')
        if d['pixel_visibility_certified'] or d['training_admitted'] or d['promotable']:
            raise ValueError('Unsupported certification')
        if d['status'] != ('pending_content_boundary' if d['content'] == 'unknown_instance_content' else 'observed_not_admitted'):
            raise ValueError('Unknown cannot be approved')


def main():
    ep = DEST/'evidence.json'; e = prior.read(ep); prior.verify(e)
    if set(OBS) != {x['event_id'] for x in e['events']}:
        raise ValueError('Explicit review coverage mismatch')
    rp = DEST/'review.json'
    if rp.exists():
        r = prior.read(rp); prior.verify(r); validate(e, r['decisions']); return r
    ds = []
    for x in e['events']:
        notes = OBS[x['event_id']]
        if len(notes) != (len(x['events']) if x['kind'] == 'FP' else 1):
            raise ValueError('Wrong number of per-box observations')
        for j, (content, reason) in enumerate(notes):
            ds.append(dict(decision_id=f"{x['event_id']}:{j}", content=content, reason=reason,
                status='pending_content_boundary' if content == 'unknown_instance_content' else 'observed_not_admitted',
                review_nature='AI-assisted', reviewed_at=datetime.now(timezone.utc).isoformat(),
                image_sha256=x['source']['image_sha256'], evidence_sha256=x['page_sha256'],
                source_identity=object_sha256(x['source']), event_identity=object_sha256(x['events']),
                prediction=x['events'][j] if x['kind'] == 'FP' else None, truth=x.get('truth'),
                pixel_visibility_certified=False, asset_identity_inferred_from_appearance=False,
                training_admitted=False, promotable=False))
    validate(e, ds)
    paths = [ep, Path(__file__).resolve(), Path(__file__).with_name('double_dose_review_observations.py')]
    r = prior.frozen(rp, dict(status='explicit_review_complete_with_named_content_gaps', decisions=ds,
        pending_ids=[d['decision_id'] for d in ds if d['status']=='pending_content_boundary'],
        counts=dict(Counter(d['content'] for d in ds)), selected_candidate=None,
        all_transitions_retained=True, full_metrics_unchanged=True,
        limits=['RGB content review, not instance-mask pixel certification',
                'Unknown development targets remain in full metrics; no label deletion',
                'Visual distractor shape is not simulation asset identity',
                'Repeated seeds/comparisons do not increase independent sample count'],
        inputs={str(p):prior.file_sha256(p) for p in paths}))
    print(r['status'], len(ds), 'decisions', len(r['pending_ids']), 'named content gaps')
    return r


if __name__ == '__main__': main()
