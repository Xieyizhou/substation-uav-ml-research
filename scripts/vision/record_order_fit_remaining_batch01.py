"""Persist explicit partial visual observations; never grant training admission."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior


OBSERVATIONS = {
    'U01': {
        'labels': [(0, 'content_visible', '变压器主体、面板、顶部三个端子及基座清楚可见。')],
        'full_frame': '左侧另有未框出的蓝色柜状设备，面板和基座可见；需来源世界及实例映射确认其是否属于目标类别，不能仅凭颜色确定身份。',
        'status': 'pending_possible_unboxed_target',
    },
    'U02': {
        'labels': [(0, 'content_visible', '变压器面板、主体和顶部端子可辨；左下局部被前景柜体遮挡。'),
                   (1, 'limited_content', '近景蓝色柜体主体及基座完整，主要为平整侧背面，左侧仅见少量面板边缘。')],
        'full_frame': '右后方另有未框出的蓝色柜状设备，正面凹面和基座可见；需唯一实例映射核实，暂记疑似全图标签覆盖缺口。',
        'status': 'pending_possible_unboxed_target',
    },
    'U03': {
        'labels': [(0, 'limited_content', '完整蓝色块状柜体背面及基座可见，未见清楚正面面板；仅凭当前外形不能充分区分类别。')],
        'full_frame': '已检查全图，未发现其他明确的未框目标；该视觉观察不替代来源映射或实例掩码认证。',
        'status': 'pending_limited_identifiable_content',
    },
}


def run():
    ep = OUT / 'remaining-review/evidence.json'
    evidence = prior.read(ep)
    prior.verify(evidence)
    events = {e['review_id']: e for e in evidence['events']}
    deps = [ep, Path(__file__).resolve()]
    decisions = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for rid, observation in OBSERVATIONS.items():
        event = events[rid]
        assert [x[0] for x in observation['labels']] == list(range(len(event['truth'])))
        page = Path(event['page_path'])
        assert prior.file_sha256(page) == event['page_sha256']
        deps.append(page)
        decisions.append(dict(review_id=rid, member_id=event['member_id'],
            image_sha256=event['image_sha256'], label_sha256=event['label_sha256'],
            page_sha256=event['page_sha256'], truth=event['truth'],
            label_observations=[dict(truth_index=i, status=s, reason=r)
                                for i, s, r in observation['labels']],
            full_frame_observation=observation['full_frame'], status=observation['status'],
            review_nature='AI辅助审核', reviewed_at=timestamp,
            pixel_visibility_certified=False, training_eligible=False))
    dest = OUT / 'remaining-review/partial-review-01.json'
    if dest.exists():
        previous = prior.read(dest)
        prior.verify(previous)
        return previous
    return prior.frozen(dest, dict(status='partial_review_with_named_gaps_not_admitted',
        decisions=decisions, reviewed_frames=3, pending_unreviewed_frames=len(events)-3,
        inputs={str(p): prior.file_sha256(p) for p in deps}))


if __name__ == '__main__':
    print(run()['status'])
