"""Direct observations of four gray-calibration pilot cards and own crops."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.neutral_gray_capture import OUT, freeze, prior
from scripts.vision.verify_neutral_gray import verify_capture

NOTES = {
    'N02': {'capacitor_east': '灰色宽箱体顶面、暗立面和底座完整可辨；顶面与地面反差有限，但左右边界及前缘可见，未见主体遮挡或图缘截断。'},
    'N04': {'reactor_north': '圆柱椭圆顶面、暗侧壁和方形基座完整可辨，顶面与后墙有清楚轮廓；未见前景遮挡和图缘截断。'},
    'N06': {
        'west_switchgear_02': '后方右窄柜的顶面、面板和右侧边可辨；下沿局部被近柜遮挡，不标为完全无遮挡。',
        'switchgear_west': '后方左宽柜顶面、面板边框、右侧面和底座可辨，主体未被图缘截断。',
        'entry_switchgear': '近柜顶面、暗面板、侧边和底座完整可辨，面板斜纹对比较低但外轮廓清晰。',
    },
    'N08': {
        'west_switchgear_03': '右侧柜体的灰顶面、左面板、侧面和底座可辨，靠近右缘但主体完整。',
        'transformer_mid': '中央箱体灰顶面、三个浅色顶部突出部、前立面和底座可辨；俯视视角限制侧部细节，未見主体遮挡或图缘截断。',
    },
}


def main():
    p = freeze()
    if set(NOTES) != set(p['pilot_units']): raise ValueError('Pilot inventory mismatch')
    deps = [OUT/'capture-protocol.json', Path(__file__).resolve()]
    decisions, coverage = [], []
    for u in p['units']:
        uid = u['unit_id']
        if uid not in NOTES: continue
        verify_capture(u)
        ep, cp = OUT/'evidence'/uid/'evidence.json', OUT/'replays'/uid/'completion.json'
        e, c = prior.read(ep), prior.read(cp)
        prior.verify(e); prior.verify(c)
        rp = Path(c['receipt_path']); r = prior.read(rp); prior.verify(r)
        if c['status'] != 'gray_capture_exact_replay_verified' or r['status'] != 'original_pixel_evidence_certified':
            raise ValueError('Replay certification missing')
        if len(r['full_mask_coverage']) != 3 or any(f['missing_targets'] for f in r['full_mask_coverage']):
            raise ValueError('Full mapped-instance coverage missing')
        if len(e['events']) != len(NOTES[uid]) or {x['object_id'] for x in e['events']} != set(NOTES[uid]):
            raise ValueError('Missing or duplicate label review')
        deps.extend([ep, cp, rp, Path(e['card_path']), Path(e['image_path'])])
        coverage.append(dict(unit_id=uid, stable_frames=3, all_mapped_visible_targets_boxed=True, receipt_path=str(rp)))
        for x in e['events']:
            decisions.append(dict(event_id=x['event_id'], object_id=x['object_id'], reason=NOTES[uid][x['object_id']],
                review_nature='AI辅助审核', review_time=datetime.now(timezone.utc).isoformat(),
                evidence_identity=e['identity'], image_sha256=e['image_sha256'], crop_sha256=x['crop_sha256'],
                status='content_sufficient_for_bounded_research', training_admitted=False, promotable=False))
    dest = OUT/'pilot-quality.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    return prior.frozen(dest, dict(status='pilot_explicitly_reviewed_expand_frozen_16', decisions=decisions,
        full_frames_viewed=list(NOTES), coverage=coverage,
        scope='Four pilot frames and seven own-instance crops explicitly reviewed; other twelve frames not approved.',
        limitations=['Known-condition material calibration, not independent scene generalization.',
                    'Mapped-instance masks do not distinguish components or clear all historical supervision risks.'],
        inputs={str(d):prior.file_sha256(d) for d in deps}))


if __name__ == '__main__': print(main()['status'])
