"""Explicit review after viewing all eight current before/after pages."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.routed_reflection_control import OUT,checked
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
NOTES={
'original':'近柜和后变压器两个完整框随各自主体左右反射；面板、顶柱、阴影与围墙同步翻转，未出现新增裁切。',
'negative':'大块体、后杆、地面标记及围墙整体反射，仍为空标签，不因镜像新增目标标注。',
'neutral':'灰块体顶面、大主体、基座及对应框左右反射；旁侧杆体和阴影同步变换，颜色未进一步去除。',
'cool':'远景多个柜体和变压器完整框与主体同步反射，三柱仍在原主体上，无框留在旧侧。',
'warm':'圆柱、方形基座与完整框同步反射，顶面和明暗边界保留，阴影方向随整帧反射。',
'gray_all_body':'多灰色设备与全部框同步反射，近变压器三柱和后柜队列保留；不把相似派生图算作新场景。',
'gray_target_body':'俯视变压器与灰块体两框均对应反射后位置，非目标青色前景也同步反射，未新增标签。',
'gray035':'远景五个设备框随对应主体反射，杆体遮挡关系一起反射；小目标仍小，不宣称改善尺度覆盖。',
}
def run():
    ep=OUT/'preview-v1/evidence.json';e=checked(ep)
    if {x['group'] for x in e['rows']}!=set(NOTES):raise ValueError('Incomplete preview')
    decisions=[dict(member_id=x['member_id'],page_sha256=x['page_sha256'],image_sha256=x['image_sha256'],label_sha256=x['label_sha256'],
        decision='bounded_transform_test_allowed',reason=NOTES[x['group']],review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat()) for x in e['rows']]
    return write_record(OUT/'preview-v1/review.json',dict(decisions=decisions,limits='Eight deterministic examples, not a full-pool re-admission or physical mirrored-world certification.',
        training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in (ep,Path(__file__))}))
if __name__=='__main__':print(len(run()['decisions']))
