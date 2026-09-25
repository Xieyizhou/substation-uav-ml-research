"""Import explicit per-instance observations after exact frame replay."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.small_scale_material_capture import OUT, freeze, prior
from scripts.vision.small_scale_pilot_observations import OBSERVATIONS as PILOT
from scripts.vision.small_scale_cool_observations import OBSERVATIONS as COOL
from scripts.vision.small_scale_review_gate import validate_decisions
from scripts.vision.verify_small_scale_material import evidence, verify_capture


def main():
    p = freeze()
    if set(PILOT) & set(COOL): raise ValueError('Duplicate observation frame')
    notes = dict(PILOT, **COOL)
    if set(notes) != {u['unit_id'] for u in p['units']}:
        raise ValueError('Missing explicit observations')
    frames, decisions, coverage = {}, [], []
    deps = [OUT/'capture-protocol.json', OUT/'pilot-quality.json', Path(__file__).resolve()]
    deps += [Path(__file__).with_name(n+'.py') for n in (
        'small_scale_pilot_observations', 'small_scale_cool_observations', 'small_scale_review_gate')]
    prior.verify(prior.read(OUT/'pilot-quality.json'))
    for u in p['units']:
        uid = u['unit_id']; verify_capture(u)
        e = evidence(u); frames[uid] = e
        cp = OUT/'replays'/uid/'completion.json'
        c = prior.read(cp); prior.verify(c)
        rp = Path(c['receipt_path']); r = prior.read(rp); prior.verify(r)
        if c['status'] != 'small_capture_exact_replay_verified' or r['status'] != 'original_pixel_evidence_certified':
            raise ValueError('Replay certification missing')
        if len(r['full_mask_coverage']) != 3 or any(x['missing_targets'] for x in r['full_mask_coverage']):
            raise ValueError('Missing full target coverage')
        if set(notes[uid]) != {x['object_id'] for x in e['events']}:
            raise ValueError('Explicit label observation missing')
        coverage.append(dict(unit_id=uid, receipt_path=str(rp), stable_frames=3,
                             all_mapped_visible_targets_boxed=True))
        deps += [cp, rp, OUT/'evidence'/uid/'evidence.json', Path(e['image_path']), Path(e['card_path'])]
        for x in e['events']:
            decisions.append(dict(event_id=x['event_id'], object_id=x['object_id'],
                reason=notes[uid][x['object_id']], review_nature='AI辅助审核',
                review_time=datetime.now(timezone.utc).isoformat(),
                evidence_identity=e['identity'], image_sha256=e['image_sha256'],
                crop_sha256=x['crop_sha256'], status='content_sufficient_for_bounded_research',
                training_admitted=False, promotable=False))
    validate_decisions(frames, decisions, list(notes))
    if len(decisions) != 64: raise ValueError('Frozen full-label count changed')
    dest = OUT/'quality-review.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r)
        validate_decisions(frames, r['decisions'], r['full_frames_viewed'])
        return r
    return prior.frozen(dest, dict(status='16_frames_64_labels_reviewed_exact_replay_complete',
        decisions=decisions, full_frames_viewed=list(notes), coverage=coverage,
        limitations=['Partial occlusion and back-facing cabinets remain explicitly documented.',
                    'Masks certify instance pixels, not component segmentation.',
                    'Four new pose groups share existing assets and known development appearance recipes.',
                    'No general training admission or promotion is granted.'],
        inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__ == '__main__': print(main()['status'])
