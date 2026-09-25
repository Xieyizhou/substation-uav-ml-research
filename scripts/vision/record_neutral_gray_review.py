"""Import direct observations; never manufacture decisions from capture success."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.neutral_gray_capture import OUT, freeze, prior
from scripts.vision.neutral_gray_observations import NOTES
from scripts.vision.neutral_gray_dataset import validate_review


def main():
    p = freeze()
    if set(NOTES) != {u['unit_id'] for u in p['units']}:
        raise ValueError('Missing explicit frame observations')
    deps = [OUT/'capture-protocol.json', Path(__file__).with_name('review_neutral_gray_pilot.py'),
            Path(__file__).resolve(), Path(__file__).with_name('neutral_gray_observations.py')]
    decisions, coverage = [], []
    for u in p['units']:
        uid = u['unit_id']
        ep, cp = OUT/'evidence'/uid/'evidence.json', OUT/'replays'/uid/'completion.json'
        e, c = prior.read(ep), prior.read(cp)
        prior.verify(e); prior.verify(c)
        rp = Path(c['receipt_path'])
        r = prior.read(rp); prior.verify(r)
        if c['status'] != 'gray_capture_exact_replay_verified' or r['status'] != 'original_pixel_evidence_certified':
            raise ValueError('Replay not certified: '+uid)
        if len(r['full_mask_coverage']) != 3 or any(x['missing_targets'] for x in r['full_mask_coverage']):
            raise ValueError('Incomplete mapped-instance coverage: '+uid)
        if len(e['events']) != len(NOTES[uid]) or set(NOTES[uid]) != {x['object_id'] for x in e['events']}:
            raise ValueError('Missing or duplicate label decision: '+uid)
        deps.extend([ep, cp, rp, Path(u['plan_path']).parent/'world.sdf', Path(e['card_path']), Path(e['image_path'])])
        coverage.append(dict(unit_id=uid, stable_frames=3, all_mapped_visible_targets_boxed=True, receipt_path=str(rp)))
        for x in e['events']:
            decisions.append(dict(event_id=x['event_id'], reason=NOTES[uid][x['object_id']],
                object_id=x['object_id'], review_nature='AI辅助审核', review_time=datetime.now(timezone.utc).isoformat(),
                evidence_identity=e['identity'], image_sha256=e['image_sha256'], crop_sha256=x['crop_sha256'],
                status='content_sufficient_for_bounded_research', training_admitted=False, promotable=False))
    dest = OUT/'quality-review.json'
    if dest.exists():
        r = prior.read(dest); validate_review(p, r); return r
    r = prior.frozen(dest, dict(status='16_frames_explicitly_reviewed_exact_replay_complete',
        decisions=decisions, full_frames_viewed=list(NOTES), coverage=coverage,
        background_source_check=dict(frames=['N11','N12'],
            observation='Right cyan block and rear gray block remain visible without target boxes.',
            source='Saved worlds: cabinet_center at (3,1), cyan body diffuse .02 .32 .43; control_building at (8,6), gray body .32 .34 .35. Both are non-target assets, not target equipment mapped by visual-instance labels.',
            limitation='Visual exterior is distinct from source identity; full mask coverage covers mapped target instances only.'),
        limitations=['Some neutral-material top boundaries have low contrast against the floor; dark faces and major body content remain identifiable.',
                    'Posterior cabinet lower edges partially occluded in layout A; not labelled completely unoccluded.',
                    '16 derivatives of eight training poses with shared assets and known viewed development material value.',
                    'Exact replay certifies each new capture; it does not assert equality across different materials.',
                    'Only body/front_panel/reactor .42 to .35 changes are allowed by the frozen structural checker.'],
        inputs={str(d):prior.file_sha256(d) for d in deps}))
    validate_review(p, r)
    return r


if __name__ == '__main__': print(main()['status'])
