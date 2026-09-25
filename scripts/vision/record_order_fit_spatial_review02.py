"""Explicit second spatial review; never grants training or pixel certification."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior

# Each association was visually inspected; projection is auxiliary evidence only.
OBSERVATIONS = [
    ('U25', 'cabinet_2', '左侧图缘未框蓝色柜体，主体轮廓与来源普通柜体位置相符。'),
    ('U26', 'cabinet_2', '前景下缘未框蓝色柜体，顶部和侧面与来源普通柜体位置相符。'),
    ('U27', 'cabinet_2', '后方中央未框蓝色小柜体，与两台变压器及右侧开关柜为不同对象。'),
    ('U32', 'cabinet_2', '中左未框蓝色柜体位于远处开关柜之前，与普通柜体投影对应。'),
    ('U33', 'cabinet_1', '右后方未框蓝色柜体与来源cabinet_1对应。'),
    ('U33', 'cabinet_3', '右侧较近未框蓝色柜体与来源cabinet_3对应。'),
]


def run():
    dest = OUT/'source-spatial-context/spatial-review-02.json'
    if dest.exists():
        result = prior.read(dest); prior.verify(result); return result
    ep = OUT/'source-spatial-context/evidence.json'
    tp = OUT/'legacy-world-source-trace.json'
    bp = prior.ROOT/'config/perception/visual_experiment_baseline_v1.json'
    evidence, trace = prior.read(ep), prior.read(tp)
    prior.verify(evidence); prior.verify(trace)
    if prior.read(bp)['ordinary_cabinet_is_target'] is not False:
        raise ValueError('Taxonomy changed')
    deps = [ep, tp, bp, Path(__file__).resolve()]
    decisions = []
    for rid, obj, reason in OBSERVATIONS + [('U29', 'transformer_sw',
        '左缘大块深色截断平面与transformer_sw主体投影相符；来源实例标签76为目标资产，完整训练标签仅包含三台开关柜。支持具名漏标风险，尚无实例掩码认证，不新增或修改标签。')]:
        row = next(x for x in evidence['rows'] if x['review_id'] == rid)
        source = next(x for x in trace['members'] if x['member_id'] == row['member_id'])
        association = next(x for x in row['projected'] if x['object_id'] == obj)
        page = Path(row['page_path']); deps.append(page)
        if rid == 'U29':
            if association['saved_labels'] != ['76'] or any(x['object_id'] == obj for x in source['annotations']):
                raise ValueError('Risk evidence changed')
            status = 'hold_pending_named_target_coverage_risk'
        else:
            pp = Path(source['source_plan']); op = pp.parent/'obstacles.json'
            if prior.read(pp)['files']['obstacles.json'] != prior.file_sha256(op):
                raise ValueError('Source taxonomy changed')
            matches = [x for x in prior.read(op)['obstacles'] if x['name'] == obj]
            if len(matches) != 1 or matches[0]['visual_category'] != 'cabinet' or association['saved_labels']:
                raise ValueError('Not uniquely non-target cabinet')
            deps.extend([pp, op]); status = 'named_structure_is_source_defined_non_target_cabinet'
        decisions.append(dict(review_id=rid, member_id=row['member_id'], object_id=obj,
            status=status, reason=reason, review_nature='AI辅助审核',
            reviewed_at=datetime.now(timezone.utc).isoformat(),
            spatial_page_sha256=prior.file_sha256(page), pixel_visibility_certified=False,
            training_eligible=False, scope='Named structure only; not whole-frame approval.'))
    return prior.frozen(dest, dict(status='six_background_associations_and_one_target_risk_recorded',
        decisions=decisions, training_started=False, history_modified=False,
        inputs={str(p): prior.file_sha256(p) for p in deps}))


if __name__ == '__main__':
    print(run()['status'])
