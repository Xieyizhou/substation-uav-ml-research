"""Explicit observations after viewing all eight scale preview pages."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.preview_routed_scale import OUT
from scripts.vision.material_routed_contrast_control import checked
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

NOTES={
'neutral':'俯视变压器三柱及右柜两个框随主体居中缩小，地面和围墙同步缩放，未新增裁切；成员名不用于重新判定材质身份。',
'cool':'近变压器和柜体队列的全部框同步缩小；后柜原有遮挡保留，未把框留在旧位置。',
'warm':'远柜与右变压器两框随主体缩小，顶柱与基座保留，新增灰边属于填充不是新场景。',
'gray_all_body':'圆柱、基座和整体标注框同步居中缩小，阴影及后杆保留，无新增裁切。',
'gray_target_body':'俯视灰块体及青色变压器两个框正确缩小，前景普通青色块体和杆体一起变换，没有据外形新增标签。',
'gray035':'远圆柱及右侧局部目标两框同步缩小，原有图缘限制不被宣称恢复；目标更小但未丢标签。',
'original':'左右图和全部框保持恒等，近左原有截断、柜队列、圆柱和右变压器未变化。',
'negative':'灰块体、杆体、地面标记左右保持恒等，空标签保留，没有因变换新增框。',
}

def run():
    ep=OUT/'preview-v1/evidence.json';e=checked(ep)
    if {r['group'] for r in e['rows']}!=set(NOTES):raise ValueError('Preview coverage changed')
    decisions=[dict(member_id=r['member_id'],image_sha256=r['image_sha256'],label_sha256=r['label_sha256'],page_sha256=r['page_sha256'],decision='bounded_transform_test_allowed',reason=NOTES[r['group']],review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat()) for r in e['rows']]
    return write_record(OUT/'preview-v1/review.json',dict(decisions=decisions,limits='Eight actual-loader geometry examples, not full-pool semantic reapproval or pixel visibility certification.',training_admitted=False,promotable=False,inputs={str(q.resolve()):file_sha256(q) for q in (ep,Path(__file__))}))

if __name__=='__main__':print(len(run()['decisions']))
