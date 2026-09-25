"""Record the eight explicitly inspected pilot frames, with exact replay gates."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.small_scale_material_capture import OUT, freeze, prior
from scripts.vision.small_scale_pilot_observations import OBSERVATIONS
from scripts.vision.verify_small_scale_material import verify_capture


def main():
    p = freeze()
    if set(OBSERVATIONS) != set(p['pilot_units']):
        raise ValueError('Pilot observation coverage mismatch')
    deps = [OUT/'capture-protocol.json', Path(__file__).resolve(),
            Path(__file__).with_name('small_scale_pilot_observations.py')]
    decisions = []
    for u in p['units']:
        uid = u['unit_id']
        if uid not in OBSERVATIONS:
            continue
        verify_capture(u)
        ep = OUT/'evidence'/uid/'evidence.json'
        cp = OUT/'replays'/uid/'completion.json'
        e, c = prior.read(ep), prior.read(cp)
        prior.verify(e); prior.verify(c)
        rp = Path(c['receipt_path'])
        r = prior.read(rp); prior.verify(r)
        if c['status'] != 'small_capture_exact_replay_verified' or r['status'] != 'original_pixel_evidence_certified':
            raise ValueError('Exact replay missing')
        if len(r['full_mask_coverage']) != 3 or any(x['missing_targets'] for x in r['full_mask_coverage']):
            raise ValueError('Incomplete full-instance coverage')
        names = [x['object_id'] for x in e['events']]
        if len(names) != len(set(names)) or set(names) != set(OBSERVATIONS[uid]):
            raise ValueError('Missing or duplicate explicit observations')
        deps.extend([ep, cp, rp, Path(e['card_path']), Path(e['image_path']), Path(u['plan_path']).parent/'world.sdf'])
        for x in e['events']:
            decisions.append(dict(event_id=x['event_id'], object_id=x['object_id'],
                reason=OBSERVATIONS[uid][x['object_id']], review_nature='AI辅助审核',
                review_time=datetime.now(timezone.utc).isoformat(),
                status='content_sufficient_for_bounded_research', evidence_identity=e['identity'],
                image_sha256=e['image_sha256'], crop_sha256=x['crop_sha256'],
                training_admitted=False, promotable=False))
    if len(decisions) != 32:
        raise ValueError('Pilot label inventory changed')
    dest = OUT/'pilot-quality.json'
    if dest.exists():
        result = prior.read(dest); prior.verify(result); return result
    return prior.frozen(dest, dict(status='pilot_reviewed_expand_remaining_eight',
        decisions=decisions, full_frames_viewed=list(OBSERVATIONS),
        background_source_check='Saved S14 world retains cabinet_center at (3,1), cyan diffuse .02 .32 .43, category cabinet; control_building at (8,6), category control_building. These are non-target assets, not unboxed target equipment. Full mapped-instance coverage is separately verified for each frame.',
        limitations=['Partial foreground occlusion remains explicitly recorded, not certified as unobstructed.',
                    'Instance masks do not distinguish panel/body/base components.',
                    'Four pose groups share existing assets; not independent asset generalization.',
                    'This authorizes remaining frozen captures, not training or model promotion.'],
        inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__ == '__main__':
    print(main()['status'])
