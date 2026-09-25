"""Explicit observations from the current sixteen full-frame/crop evidence panels."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.lineage_capped_control import OUT, read, verify, frozen, file_sha256
from scripts.vision.audit_lineage_capped_results import verify_decisions

OBS = {
 'FP01': ('cabinet_like_block', '灰色长方块的大侧面及右侧深色矩形面板，底部少量地面；不是单独杆体。'),
 'FP02': ('cabinet_like_block', '与 FP01 相同登记视点的常规光照变体；灰色块体侧面、深色面板及底座。'),
 'FP03': ('mixed_structure', '灰色块体前有深色竖杆穿过，右下叠入青色块体；预测覆盖混合结构，不能只归为杆体。'),
 'FP04': ('cabinet_like_block', '灰色块体正面深色矩形面板和底座，主体框内没有四类目标的可确认特征。'),
 'FP05': ('cabinet_like_block', '青色小块体的两个平整面及黑色底座，无可见面板细节。'),
 'FP06': ('mixed_structure', '近处灰色块体上半部及棱边，框向上包含天空和后方杆体；未覆盖完整底座。'),
 'LOSS01': ('foreground_occluded', '远处圆柱只露出上段，下部被前景青色设备遮挡；不能认证完整主体可见。'),
 'LOSS02': ('foreground_occluded', '同登记视点原始条件，圆柱上段可见，下部被前景设备遮挡。'),
 'LOSS03': ('foreground_occluded', '后方矩形主体仅上部和侧边可见，前景大型设备的顶部与接线柱侵入真值框。'),
 'LOSS04': ('boundary_truncated', '画面右边截断圆柱及基座；仍可见明显圆柱曲面、顶面和部分基座。'),
 'LOSS05': ('clear_cylindrical_body', '圆柱顶面、连续侧面及基座清晰且较大，无明显主体前景遮挡；不能将该漏检归结为仅基座或极端截断。'),
 'LOSS06': ('clear_block_body', '大面积青灰色矩形主体及基座清楚，表面特征较少；视觉观察不独立认证类别身份。'),
 'LOSS07': ('foreground_occluded', '远处矩形主体只露上段，前景设备顶部与多个接线柱覆盖框的下部。'),
 'LOSS08': ('partially_occluded_block', '矩形主体的大侧面及底边可见，左下被前景青色设备遮挡，尺度较小。'),
 'LOSS09': ('foreground_occluded', '圆柱上段轮廓明显，下部被前景设备顶面遮挡；框内还包含前景接线柱。'),
 'LOSS10': ('foreground_occluded', '原始条件下圆柱上段可见，下部遮挡；与光照变体分别核对，不用另一帧代替本帧。'),
}

def main():
    path=OUT/'error-review-v1/evidence.json'; e=read(path); verify(e)
    if set(OBS)!={r['event_id'] for r in e['events']}: raise ValueError('Observation coverage mismatch')
    ds=[]; now=datetime.now(timezone.utc).isoformat()
    for r in e['events']:
        content,reason=OBS[r['event_id']]
        for i in range(len(r['predictions']) if r['kind']=='FP' else 1):
            d=dict(review_id=f"{r['event_id']}:{i}",visual_content=content,reason=reason,
                review_nature='AI辅助审核',reviewed_at=now,pixel_visibility_certified=False,
                image_sha256=r['source']['image_sha256'],evidence_sha256=r['evidence_sha256'],
                crop_sha256=r['crops'][i]['sha256'],asset_identity_inferred_from_shape=False,
                training_admitted=False,promotable=False)
            if r['kind']=='FP': d.update(prediction=r['predictions'][i],asset_identity='unknown')
            else: d.update(truth=r['truth'],diagnostic_events=r['events'])
            ds.append(d)
    verify_decisions(e,ds)
    frozen(OUT/'error-review-v1/review.json',dict(status='explicit_AI_visual_review_complete',decisions=ds,
        repeat_relations=[dict(events=['FP01','FP02'],relation='same_registered_view_different_lighting_not_independent_scene')],
        scope='Six FP boxes and ten focused new-loss image instances; all other transitions retained in evidence, not claimed visually approved.',
        inputs={str(p):file_sha256(p) for p in [path,Path(__file__),Path('scripts/vision/audit_lineage_capped_results.py').resolve()]}))
    print('EXPLICIT_REVIEWS',len(ds))

if __name__=='__main__': main()
