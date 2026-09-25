"""Explicit observations after viewing all eight current negative evidence pages."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision import routed_fixed_bn_control as arm
from scripts.vision.audit_reviewed_order_results import validate_review
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

NOTES={
0:('mixed_structure','灰块体侧面、粗杆和青块顶面共同入框。'),
1:('mixed_structure','冷暗条件下灰块体侧面、粗杆和青块顶面共同入框。'),
2:('gray_block_body','灰块体大侧面与底部基座，框右部含地面和围墙。'),
3:('gray_block_body','灰块体大侧背面和底座为主，右下含少量青块，不据外形认定资产类别。'),
4:('mixed_structure','冷暗灰块体侧背面和暗面板，粗杆及右下青块共同入框。'),
5:('gray_block_body','灰块体正面暗面板和边框，下缘接近基座。'),
6:('gray_block_body','冷暗灰块体正面暗面板、边框和底部基座。'),
7:('mixed_structure','远处灰块体侧面与粗杆、青块局部共同入框，杆遮挡块体。'),
8:('mixed_structure','灰块体侧背面、暗面板、粗杆及右下青块共同入框。'),
9:('mixed_structure','冷暗灰块体侧背面、暗面板、粗杆及右下青块共同入框。'),
10:('colored_block_body','青色块体侧面、暗面板、顶面和完整基座。视觉柜状不等于目标资产。'),
11:('gray_block_body','灰块体正面暗面板、边框和基座，与另一seed重复覆盖同一结构。'),
12:('gray_block_body','冷暗灰块体正面暗面板、边框和基座，与另一seed重复覆盖同一结构。'),
}

def run():
    ep=arm.OUT/'evaluation-v1/error-review-v1/evidence.json';e=arm.checked(ep)
    assert {x['event_id'] for f in e['frames'] for x in f['events']}=={f'fp-{i:02}' for i in NOTES}
    decisions=[];stamp=datetime.now(timezone.utc).isoformat()
    for f in e['frames']:
        for x in f['events']:
            category,reason=NOTES[int(x['event_id'].split('-')[-1])]
            decisions.append(dict(event_id=x['event_id'],seed=x['seed'],prediction=x['prediction'],content_category=category,reason=reason,review_nature='AI辅助审核',reviewed_at=stamp,image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],asset_identity='unknown_visual_shape_not_asset_identity'))
    validate_review(e,decisions)
    dest=arm.OUT/'audit-v1/negative-review.json'
    if dest.exists():return arm.checked(dest)
    return write_record(dest,dict(status='negative_content_reviewed_not_candidate_passed',negative_decisions=decisions,unique_images=len(e['frames']),training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in (ep,Path(__file__))}))

if __name__=='__main__':print(len(run()['negative_decisions']))
