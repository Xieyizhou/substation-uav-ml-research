"""Authored six-variant transform preview observations."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.routed_gray_transfer_control import OUT,checked
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
NOTES={
'neutral':'近灰块体顶面、主体与基座明暗边界仍可见；冷色偏色被移除，未见空间位移。',
'cool':'圆柱顶面、主体阴影与方形底座仍可区分；偏蓝色去除，轮廓位置不变。',
'warm':'暖色圆柱转灰，顶面与主体、底座及阴影仍可见，未见裁切变化。',
'gray_all_body':'原灰圆柱与底座轮廓基本保持，地面及远处橙色标记也随整帧去色；并非设备局部变换。',
'gray_target_body':'俯视图三柱、变压器主体、灰块及右下柜体轮廓仍可见；非目标柜体同样去色，不能称物理材质替换。',
'gray035':'远景多设备三柱、顶面、队列及地面边界仍可见；小目标仍小，不把去色视为尺度覆盖改善。',
}
def run():
    ep=OUT/'preview-v1/evidence.json';e=checked(ep)
    if {r['variant'] for r in e['rows']}!=set(NOTES):raise ValueError('Incomplete preview')
    decisions=[dict(member_id=r['member_id'],page_sha256=r['page_sha256'],image_sha256=r['image_sha256'],label_sha256=r['label_sha256'],
        decision='bounded_transform_test_allowed',reason=NOTES[r['variant']],review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat()) for r in e['rows']]
    return write_record(OUT/'preview-v1/review.json',dict(decisions=decisions,
        limits='Six deterministic loader examples only, not full-pool quality recertification. Color-only edges may weaken; this is an experimental risk, not evidence of benefit.',
        training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in (ep,Path(__file__))}))
if __name__=='__main__':print(len(run()['decisions']))
