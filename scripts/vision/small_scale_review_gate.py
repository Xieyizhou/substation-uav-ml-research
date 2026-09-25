"""Pure full-inventory review checks; never generates review decisions."""
def validate_decisions(frames, decisions, viewed):
    if len(viewed) != len(set(viewed)) or set(viewed) != set(frames):
        raise ValueError('Missing/duplicate full-frame review')
    expected = {}
    for e in frames.values():
        for x in e['events']:
            if x['event_id'] in expected:
                raise ValueError('Duplicate evidence event')
            expected[x['event_id']] = (e, x)
    ids = [d['event_id'] for d in decisions]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('Missing/duplicate decision')
    for d in decisions:
        e, x = expected[d['event_id']]
        for field, actual in (('evidence_identity', e['identity']),
                              ('image_sha256', e['image_sha256']),
                              ('crop_sha256', x['crop_sha256']),
                              ('object_id', x['object_id'])):
            if d.get(field) != actual:
                raise ValueError('Stale or conflicting review: '+field)
        if d.get('status') != 'content_sufficient_for_bounded_research':
            raise ValueError('Unresolved content')
        if not d.get('reason') or not d.get('review_time') or d.get('review_nature') != 'AI辅助审核':
            raise ValueError('Missing explicit rationale/nature/time')
        if d.get('training_admitted') is not False or d.get('promotable') is not False:
            raise ValueError('Promotion flag drift')
    return len(expected)
