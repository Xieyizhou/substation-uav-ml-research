"""Import already written AI observations; no quality/role admission inference."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior
from scripts.vision import order_fit_remaining_observations as notes


def validate(evidence, decisions):
    events = {e['review_id']: e for e in evidence['events']}
    if len(events) != len(evidence['events']):
        raise ValueError('Duplicate evidence frame')
    if len({d['review_id'] for d in decisions}) != len(decisions):
        raise ValueError('Duplicate review frame')
    if {d['review_id'] for d in decisions} != set(events):
        raise ValueError('Missing or unknown review frame')
    for d in decisions:
        e = events[d['review_id']]
        for k in ('member_id','image_sha256','label_sha256','page_sha256','truth'):
            if d[k] != e[k]:
                raise ValueError('Stale or mismatched review evidence: '+k)
        labels = d['label_observations']
        if [x['truth_index'] for x in labels] != list(range(len(e['truth']))):
            raise ValueError('Missing or repeated complete-label decision')
        if any(x['status'] not in (notes.V, notes.L, notes.I) or not x['reason'] for x in labels):
            raise ValueError('Missing explicit label observation')
        if not d['full_frame_observation'] or not d['reviewed_at'] or d['review_nature'] != 'AI辅助审核':
            raise ValueError('Incomplete explicit review')
        if d['pixel_visibility_certified'] is not False or d['training_eligible'] is not False:
            raise ValueError('Visual review cannot auto-certify pixels or admission')


def run():
    ep = OUT/'remaining-review/evidence.json'
    rp = OUT/'remaining-review/partial-review-01.json'
    dest = OUT/'remaining-review/review.json'
    if dest.exists():
        record = prior.read(dest); prior.verify(record)
        e = prior.read(ep); prior.verify(e); validate(e, record['decisions'])
        return record
    e = prior.read(ep); prior.verify(e)
    previous = prior.read(rp); prior.verify(previous)
    decisions = list(previous['decisions'])
    existing = {d['review_id'] for d in decisions}
    if set(notes.OBSERVATIONS) != {x['review_id'] for x in e['events']} - existing:
        raise ValueError('New observation coverage differs')
    deps = [ep, rp, Path(notes.__file__).resolve(), Path(__file__).resolve(), OUT/'protocol.json']
    p = prior.read(OUT/'protocol.json'); prior.verify(p)
    members = {m['member_id']:m for m in p['members']}
    timestamp = datetime.now(timezone.utc).isoformat()
    for event in e['events']:
        m = members[event['member_id']]
        for field in ('image','label'):
            path = Path(m[field+'_path'])
            if prior.file_sha256(path) != event[field+'_sha256']:
                raise ValueError('Current input changed')
            deps.append(path)
        page = Path(event['page_path'])
        if prior.file_sha256(page) != event['page_sha256']:
            raise ValueError('Review page changed')
        deps.append(page)
        if event['review_id'] in existing:
            continue
        observations, whole = notes.OBSERVATIONS[event['review_id']]
        decisions.append(dict(**{k:event[k] for k in ('review_id','member_id','image_sha256','label_sha256','page_sha256','truth')},
            label_observations=[dict(truth_index=i,status=s,reason=r) for i,(s,r) in enumerate(observations)],
            full_frame_observation=whole, status='observations_complete_quality_and_source_gate_pending',
            review_nature='AI辅助审核', reviewed_at=timestamp,
            pixel_visibility_certified=False, training_eligible=False))
    decisions.sort(key=lambda d:d['review_id'])
    validate(e, decisions)
    return prior.frozen(dest, dict(status='51_frames_full_label_visual_observations_complete_not_admitted',
        decisions=decisions, reviewed_frames=len(decisions),
        reviewed_labels=sum(len(d['label_observations']) for d in decisions),
        content_counts=dict(Counter(x['status'] for d in decisions for x in d['label_observations'])),
        insufficient_content_members=[d['member_id'] for d in decisions if any(x['status']==notes.I for x in d['label_observations'])],
        remaining_gates=['Resolve named full-image structures against source identity',
            'Bounded content disposition including accepted closed exterior semantics',
            'Source-role isolation, complete eligible inventory, export and real loader validation'],
        inputs={str(x):prior.file_sha256(x) for x in deps}))


if __name__ == '__main__':
    r = run(); print(r['status'],r['reviewed_labels'],r['content_counts'])
