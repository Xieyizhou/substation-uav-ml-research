"""Import explicit observations authored after viewing all 13 pages/19 crops."""
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision.freeze_clear_context_training import OUT
from scripts.vision.prepare_clear_context_increment import checked
from scripts.vision.audit_reviewed_order_results import validate_review

OBS = {
0: ('teal_block', '青色块体的无面板大面、顶部、侧面与底座；前景杆基本在框外。'),
1: ('teal_block', '冷暗青色块体的大面、顶部和底座；框向右扩入地面及远处杆体下段。'),
2: ('teal_panel', '青色块体正面暗矩形面板、边框、顶部和底座，非杆体。'),
3: ('gray_panel', '近处灰色块体的大侧面、正面暗面板和底座均入框。'),
4: ('gray_panel', '冷暗灰色块体主体和右侧暗面板、底座；右边贴近竖杆。'),
5: ('mixed_structure', '远处灰色块体被粗竖杆遮挡，框同时含右下青色块体局部。'),
6: ('gray_panel', '冷暗灰色块体主体、侧面暗面板和底座；并非仅地面阴影。'),
7: ('mixed_structure', '灰色块体与前景粗竖杆重叠，右下含青色结构局部。'),
8: ('mixed_structure', '冷暗灰色块体与前景粗杆和右下青色块体共同入框。'),
9: ('gray_panel', '正对灰色块体的大暗矩形面板、灰色边框及底座。'),
10: ('teal_panel', '左侧青色块体的正面暗面板、右侧面和底座，非画面中央灰色块体。'),
11: ('gray_body', '近处灰色块体无面板大面与底座；右缘少量杆体和背景。'),
12: ('gray_panel', '灰色块体主体、右侧暗矩形面板和底座；右缘靠近杆体。'),
13: ('gray_panel', '与同图另一seed相同灰色块体，主体、侧面面板和底座入框。'),
14: ('mixed_structure', '灰色块体被前景粗杆分割，框含右下青色局部。'),
15: ('mixed_structure', '冷暗灰色块体与前景竖杆及青色局部交叠。'),
16: ('gray_body', '灰色块体无面板大面和底座为主，右下被青色块体部分遮挡。'),
17: ('gray_body', '冷暗灰色块体大面和底座，右下青色遮挡；面板在右侧框缘附近。'),
18: ('gray_panel', '正对灰色块体暗矩形面板、外围灰框和底座。'),
}


def run():
    root = OUT/'evaluation-v1/error-review-v1'; ep = root/'evidence.json'
    evidence = checked(ep); decisions = []; stamp = datetime.now(timezone.utc).isoformat()
    for frame in evidence['frames']:
        for event in frame['events']:
            category, reason = OBS[int(event['event_id'].split('-')[-1])]
            decisions.append(dict(event_id=event['event_id'], seed=event['seed'], view_id=event['view_id'],
                prediction=event['prediction'], image_sha256=frame['image_sha256'], evidence_sha256=frame['evidence_sha256'],
                content_category=category, reason=reason, review_nature='AI辅助审核', reviewed_at=stamp,
                asset_identity='unknown_not_inferred_from_visual_shape', decision='reviewed_false_positive_content'))
    validate_review(evidence, decisions)
    dest=root/'review.json'
    if dest.exists():
        old=checked(dest);validate_review(evidence,old['decisions']);return old
    return write_record(dest, dict(status='nineteen_fp_crops_reviewed', decisions=decisions,
        unique_images=len(evidence['frames']), unique_view_ids=len({d['view_id'] for d in decisions}),
        content_counts=dict(Counter(d['content_category'] for d in decisions)),
        same_image_seed_counts={f['frame_id']:len({e['seed'] for e in f['events']}) for f in evidence['frames']},
        limits='Visual shape only; asset identity not certified. Seeds and lighting variants do not add independent scenes. Positive-error review remains pending.',
        training_admitted=False, promotable=False, inputs={str(p.resolve()):file_sha256(p) for p in (ep,Path(__file__))}))


if __name__ == '__main__':
    print(run()['content_counts'])
