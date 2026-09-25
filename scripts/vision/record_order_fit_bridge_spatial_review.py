"""Explicit per-variant visual associations; auxiliary, not instance masks."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior

NOTES = {
    'U38': ('前景变压器右后方蓝色局部与cabinet_center对应。', '左缘灰色立面与control_building对应，底部被前景目标遮挡。'),
    'U39': ('背景变体中，前景变压器右后方蓝色局部仍与cabinet_center对应。', '蓝灰天空条件下左缘灰色立面与control_building对应。'),
    'U40': ('中性灰目标条件下，前景右后方保持蓝色的局部与cabinet_center对应。', '左缘建筑灰色立面可见；不将灰色建筑误称为灰色目标柜体。'),
    'U41': ('左侧灰色变压器后方露出的窄蓝色侧面与cabinet_center对应。', '中后方电容器顶部后露出的深灰立面与control_building对应；投影重叠不等于整栋可见。'),
    'U42': ('左侧蓝色变压器后方窄蓝色侧面与cabinet_center对应。', '电容器后露出的建筑顶部及深灰立面与control_building对应。'),
}


def run():
    dest = OUT/'source-spatial-context/bridge-review-03.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    ep = OUT/'source-spatial-context/evidence.json'; tp = OUT/'legacy-world-source-trace.json'
    e, t = prior.read(ep), prior.read(tp)
    prior.verify(e); prior.verify(t)
    deps = [ep, tp, Path(__file__).resolve()]; decisions = []
    for rid, notes in NOTES.items():
        row = next(x for x in e['rows'] if x['review_id'] == rid)
        source = next(x for x in t['members'] if x['member_id'] == row['member_id'])
        pp = Path(source['source_plan']); op = pp.parent/'obstacles.json'; page = Path(row['page_path'])
        if prior.read(pp)['files']['obstacles.json'] != prior.file_sha256(op):
            raise ValueError('Source taxonomy hash changed')
        deps.extend([pp, op, page])
        for obj, category, reason in zip(['cabinet_center', 'control_building'], ['cabinet', 'control_building'], notes):
            matches = [x for x in prior.read(op)['obstacles'] if x['name'] == obj]
            projected = [x for x in row['projected'] if x['object_id'] == obj]
            if len(matches) != 1 or matches[0]['visual_category'] != category or len(projected) != 1 or projected[0]['saved_labels']:
                raise ValueError('Source association changed')
            decisions.append(dict(review_id=rid, member_id=row['member_id'], object_id=obj,
                source_visual_category=category, status='named_source_background_association_supported',
                reason=reason, review_nature='AI辅助审核', reviewed_at=datetime.now(timezone.utc).isoformat(),
                page_sha256=prior.file_sha256(page), pixel_visibility_certified=False, training_eligible=False))
    return prior.frozen(dest, dict(status='five_variants_ten_named_associations_reviewed', decisions=decisions,
        limitation='Each variant inspected; projections do not establish visibility masks or whole-frame label completeness.',
        inputs={str(p): prior.file_sha256(p) for p in deps}))


if __name__ == '__main__': print(run()['status'])
