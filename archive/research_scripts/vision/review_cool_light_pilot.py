"""Explicit cool-light pilot observations; not a full-dataset admission."""
from datetime import datetime, timezone
from pathlib import Path

from scripts.vision.cool_light_capture import OUT, freeze, prior
from scripts.vision.verify_cool_light import verify_capture

NOTES = {
    'L03': {'capacitor_east': '冷色光照下宽箱体顶面、暗立面及基座可辨，未见主体遮挡或图缘截断；视觉外形描述不替代来源类别身份。'},
    'L07': {'reactor_north': '圆柱椭圆顶面、深色侧壁和方形基座可辨，主体轮廓完整，未见前景遮挡或图缘截断。'},
    'L11': {
        'west_switchgear_02': '后方右侧窄柜顶面和主要面板可辨，底部附近被近柜局部遮挡；不标为完全无遮挡。',
        'switchgear_west': '后方左侧宽柜的面板边框、顶面、右侧面和基座可辨，未见图缘截断。',
        'entry_switchgear': '近柜暗面板、顶面、侧边及基座可辨，主体完整且未被图缘截断。',
    },
    'L15': {
        'west_switchgear_03': '右侧柜体顶面、前面板、侧面和基座可辨，虽靠近右缘但主体未被截断。',
        'transformer_mid': '中央箱体顶面、三个顶部突出部、暗立面及基座可辨；俯视条件限制侧向结构覆盖，不据此声称多视角充分覆盖。',
    },
}


def main():
    protocol = freeze()
    if set(NOTES) != set(protocol['pilot_units']):
        raise ValueError('Pilot inventory mismatch')
    deps = [OUT/'capture-protocol.json', OUT/'condition-audit.json', Path(__file__).resolve()]
    decisions, coverage = [], []
    for unit in protocol['units']:
        uid = unit['unit_id']
        if uid not in NOTES:
            continue
        verify_capture(unit)
        ep, cp = OUT/'evidence'/uid/'evidence.json', OUT/'replays'/uid/'completion.json'
        evidence, completion = prior.read(ep), prior.read(cp)
        prior.verify(evidence)
        prior.verify(completion)
        rp = Path(completion['receipt_path'])
        replay = prior.read(rp)
        prior.verify(replay)
        if completion['status'] != 'low_light_capture_exact_replay_verified' or replay['status'] != 'original_pixel_evidence_certified':
            raise ValueError('Replay certification missing')
        frames = replay['full_mask_coverage']
        if len(frames) != 3 or any(f['missing_targets'] for f in frames):
            raise ValueError('Full mapped-instance coverage missing')
        events = evidence['events']
        if len(events) != len(NOTES[uid]) or {e['object_id'] for e in events} != set(NOTES[uid]):
            raise ValueError('Missing or duplicate label review')
        deps.extend([ep, cp, rp, Path(evidence['card_path']), Path(evidence['image_path'])])
        coverage.append(dict(unit_id=uid, stable_frames=3, all_mapped_visible_targets_boxed=True, receipt_path=str(rp)))
        for event in events:
            decisions.append(dict(event_id=event['event_id'], object_id=event['object_id'],
                reason=NOTES[uid][event['object_id']], review_nature='AI辅助审核',
                review_time=datetime.now(timezone.utc).isoformat(), evidence_identity=evidence['identity'],
                image_sha256=evidence['image_sha256'], crop_sha256=event['crop_sha256'],
                status='content_sufficient_for_bounded_research', training_admitted=False, promotable=False))
    dest = OUT/'pilot-quality.json'
    if dest.exists():
        result = prior.read(dest)
        prior.verify(result)
        return result
    return prior.frozen(dest, dict(status='pilot_explicitly_reviewed_expand_frozen_32',
        decisions=decisions, full_frames_viewed=list(NOTES), coverage=coverage,
        scope='Four cool-light pilot frames and seven own-instance crops explicitly reviewed; remaining 28 frames are not approved.',
        limitations=['Known development lighting recipe adaptation, not blind generalization.',
                    'Card heading lower_light is inherited display text; actual cool-light XML and condition-audit are authoritative.',
                    'Mapped-instance coverage does not certify component masks or eliminate all historical supervision risks.'],
        inputs={str(p): prior.file_sha256(p) for p in deps}))


if __name__ == '__main__':
    print(main()['status'])
