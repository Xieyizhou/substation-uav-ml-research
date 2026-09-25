"""Explicit AI observations from the six inspected evidence pages, not admission."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.diagnose_lr_material_fit import OUT, checked
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

NOTES={
 ('G04-gray_all_body',1):'圆柱顶面及大部分主体可见，右下部被前景变压器遮挡；不是空框。无实例掩码，不认证精确可见比例。',
 ('S07-gray_all_body',1):'圆柱顶面、主体和基座可见，前景变压器邻近右边缘；主体具有圆柱轮廓，但低置信度。',
 ('small-material-v1-S02',4):'灰色块体顶面、正面及基座可见，左下部被变压器遮挡；此视角未见独特前面板，视觉类别线索有限。',
 ('small-material-v1-S04',0):'柜体顶面、正面板轮廓可见，左下角被近处柜体局部遮挡；不能将错类全归为不可见。',
 ('small-material-v1-S14',1):'灰色柜状主体、顶面和基座可见，前景杆遮住右部；此视角没有明显面板，类别可辨线索有限。',
 ('small-material-v1-S14',5):'远处灰色块体顶面和主体可见，右部与近处变压器重叠；尺度较小且无明显面板。',
 ('small-material-v1-S16',1):'较暗灰色柜状主体、顶面及基座可见，前景杆遮挡右部；与S14共享视角，不能算新增独立结构。',
}

def main():
    root=OUT/'residual-review-v1';ep=root/'evidence.json';e=checked(ep);decisions=[]
    for f in e['frames']:
        for event in f['events']:
            key=(f['member_id'],event['truth_index'])
            decisions.append(dict(**event,member_id=f['member_id'],visual_reason=NOTES[key],decision='diagnostic_observation_not_training_admission',
                review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),image_sha256=f['image_sha256'],label_sha256=f['label_sha256'],page_sha256=f['page_sha256']))
    if len(decisions)!=7 or len(NOTES)!=7:raise ValueError('Review coverage mismatch')
    write_record(root/'review.json',dict(status='seven_residual_events_reviewed_six_unique_frames',decisions=decisions,
        limitations='No new identity certification, no mask-derived visibility, no historical label edits. High fitting recall is not proof of comprehensive coverage.',
        training_admitted=False,promotable=False,inputs={str(x.resolve()):file_sha256(x) for x in (ep,Path(__file__))}))

if __name__=='__main__':main()
