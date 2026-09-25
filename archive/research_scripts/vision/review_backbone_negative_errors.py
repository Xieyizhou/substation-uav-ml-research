"""Explicit current-arm observations of all 17 negative-frame predictions."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.routed_backbone_control import OUT,checked
from scripts.vision.audit_reviewed_order_results import validate_review
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

NOTES={
0:('gray_block_body','冷暗灰块体大侧背面、右侧暗面板及基座，右框缘紧邻杆体与青色局部。'),
1:('mixed_structure','灰块体侧背面、中央前景粗杆及右下青色块体局部共同入框，不是独立圆柱主体。'),
2:('gray_block_body','灰块体大侧背面与基座，右缘包含少量暗面板和青色前景，主杆体在框外。'),
3:('teal_block_body','冷暗青色块体大侧面、右暗面板与完整基座清楚，不是后方灰块体。'),
4:('gray_block_body','冷暗灰块体正面暗面板、灰边框及完整基座。'),
5:('teal_block_body','远处青色块体两个主体面与基座，无面板可见，未见主要遮挡。'),
6:('teal_block_body','青色块体顶面、大主体面与完整基座；框还包含地面阴影及右上杆脚，左前粗杆主要在框外。'),
7:('gray_block_body','近灰块体两个大主体面与基座，底部图缘截断；杆体顶部在主体后方，框内未见变压器顶柱。'),
8:('gray_block_body','冷暗灰块体侧背面、右暗面板及基座，与同图其他seed覆盖同一结构。'),
9:('gray_block_body','冷暗灰块体侧背面、右暗面板及基座，与本seed开关柜预测覆盖同一结构，不是额外独立对象。'),
10:('gray_block_body','灰块体大侧背面及基座，右缘包含青色前景局部；与同图另一seed覆盖同一结构。'),
11:('teal_block_body','左缘青色块体前暗面板、顶面、右侧与基座，左图缘轻度截断。'),
12:('gray_panel','灰块体正面暗矩形面板及少量灰边框；框没有覆盖完整主体与基座。'),
13:('gray_block_body','冷暗灰块体正面暗面板、灰边框及基座，与另外两seed覆盖同一结构。'),
14:('gray_block_body','冷暗灰块体侧背面、右暗面板及基座，与同图其他seed覆盖同一结构。'),
15:('teal_block_body','冷暗青色块体大侧面、右暗面板及基座，与同图另一seed指向同一青色结构。'),
16:('gray_block_body','冷暗灰块体正面暗面板、灰边框及基座，与另外两seed覆盖同一结构。'),
}

def run():
    root=OUT/'evaluation-v1/error-review-v1';ep=root/'evidence.json';e=checked(ep);decisions=[]
    for frame in e['frames']:
        for event in frame['events']:
            category,reason=NOTES[int(event['event_id'].split('-')[-1])]
            decisions.append(dict(event_id=event['event_id'],seed=event['seed'],prediction=event['prediction'],content_category=category,reason=reason,
                review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),image_sha256=frame['image_sha256'],evidence_sha256=frame['evidence_sha256'],
                asset_identity='unknown_visual_shape_not_asset_identity'))
    validate_review(e,decisions)
    return write_record(root/'content-review.json',dict(status='negative_content_review_complete_not_candidate_admission',decisions=decisions,
        unique_images=len(e['frames']),prediction_count=len(decisions),training_admitted=False,promotable=False,
        inputs={str(p.resolve()):file_sha256(p) for p in (ep,Path(__file__))}))

if __name__=='__main__':print(run()['status'])
