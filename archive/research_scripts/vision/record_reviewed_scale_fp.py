"""Current scale experiment: explicit observations, not inherited approvals."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.freeze_reviewed_scale_control import OUT, prior
from scripts.vision.record_reviewed_endpoint_review import validate

# Authored after viewing all nine current full-image and per-prediction pages.
NOTES = {
    'F01': [('block_side', '右图缘灰色块体大片侧背面和黑色基座，右侧截断；框内无可辨圆柱或端子。')],
    'F02': [('cabinet_like', '青色低矮块体顶部、侧背面和黑色基座清楚，当前面无明显面板。')],
    'F03': [('cabinet_like', '青色柜状体的深色矩形面板、顶部、侧面及基座，左侧邻近图缘。')],
    'F04': [('mixed_structure', '大框同时包含灰色面板块体、上方细杆和大片天空，不能视为单一圆柱轮廓。'),
            ('block_with_panel', '较紧框覆盖灰色方块、深色正面板、侧面和黑色基座；顶缘有后方杆体小片段。')],
    'F05': [('ground_shadow', '局部框内是地面网格与杆体投影的突起边缘，不包含实体柜体。')],
    'F06': [('truncated_panel_block', '右图缘灰色块体面板、侧边和基座截断片段；同时含地面及围墙边界。'),
            ('truncated_panel_block', '右图缘同一灰色面板块体截断片段，面板与基座可见，不见圆柱。'),
            ('truncated_panel_block', '右图缘同一灰色面板块体与底部地面，框边略有差异，非新增独立结构。')],
    'F07': [('ground_shadow', '框主要覆盖灰块下方地面网格及阴影，上沿邻接黑色基座，非完整实体柜体。'),
            ('ground_shadow', '图像底缘的极窄横条框覆盖地面／阴影片段，无法辨识独立设备内容。')],
    'F08': [('block_with_panel', '灰色方块正面大深色面板、窄边框及基座，外形清楚但不据此认定资产类别。')],
    'F09': [('block_with_panel', '灰色块体大侧背面、右侧深色面板、顶部及底座，右侧邻接细杆。')],
}

def run():
    evidence = OUT/'evaluation/false-positive-review-v1/evidence.json'
    e = prior.read(evidence)
    prior.verify(e)
    if {g['image_id'] for g in e['images']} != set(NOTES):
        raise ValueError('Review population changed')
    decisions = []
    now = datetime.now(timezone.utc).isoformat()
    for g in e['images']:
        notes = NOTES[g['image_id']]
        if len(notes) != len(g['events']):
            raise ValueError('Prediction population changed')
        for event, (structure, reason) in zip(g['events'], notes):
            paths = [evidence, Path(event['page_path']), Path(g['source']['image_path'])]
            decisions.append(dict(event_id=event['event_id'], image_id=g['image_id'],
                prediction=event['prediction'], cell=event['cell'], seed=event['seed'],
                visual_structure=structure, reason=reason, review_nature='AI辅助审核',
                reviewed_at=now, simulation_asset_identity='unknown_not_resolved_by_visual_review',
                same_image_seeds=sorted({v['seed'] for v in g['events']}),
                independent_sample=False, training_admitted=False, promotable=False,
                evidence_hashes={str(p): prior.file_sha256(p) for p in paths}))
    expected = [v['event_id'] for g in e['images'] for v in g['events']]
    validate(decisions, expected, 'event_id')
    dest = evidence.parent/'explicit-review.json'
    if dest.exists():
        old = prior.read(dest)
        prior.verify(old)
        validate(old['decisions'], expected, 'event_id')
        return old
    return prior.frozen(dest, dict(status='false_positive_visual_review_complete_with_asset_identity_limits',
        decisions=decisions, unique_images=len(e['images']), prediction_events=len(decisions),
        endpoint_miss_review_complete=False, selected_candidate=None,
        inputs={str(p): prior.file_sha256(p) for p in (evidence, Path(__file__).resolve(),
            Path(__file__).with_name('record_reviewed_endpoint_review.py'))}))

if __name__ == '__main__':
    print(run()['status'])
