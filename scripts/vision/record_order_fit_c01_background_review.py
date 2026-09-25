"""Explicit observations from the viewed C01 background projection page."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior


def run():
    ep = OUT/'c01-background-projection.json'; tp = OUT/'c01-source-correspondence.json'
    e, t = prior.read(ep), prior.read(tp)
    prior.verify(e); prior.verify(t)
    page = Path(e['page_path'])
    expected = {'cabinet_center':'cabinet', 'control_building':'control_building'}
    if {r['object_id']:r['source_category'] for r in e['rows']} != expected:
        raise ValueError('Source categories changed')
    notes = {
        'cabinet_center':'中左未框蓝色柜体的顶面、侧面和底座位置与cabinet_center辅助投影相符，来源配置定义为普通柜体；不是右后方已标注的开关柜。',
        'control_building':'下缘大块灰色屋顶及右侧阴影邻域与control_building投影位置相符；可见屋顶被图缘截断，不把它判成电气目标。',
    }
    decisions = [dict(object_id=obj, source_category=expected[obj], reason=reason,
        status='named_background_source_association_supported', review_nature='AI辅助审核',
        reviewed_at=datetime.now(timezone.utc).isoformat(), page_sha256=prior.file_sha256(page),
        pixel_visibility_certified=False, training_eligible=False) for obj,reason in notes.items()]
    dest = OUT/'c01-background-review.json'
    if dest.exists():
        r = prior.read(dest); prior.verify(r); return r
    deps = [ep,tp,page,Path(__file__).resolve()]
    return prior.frozen(dest,dict(status='two_C01_background_questions_reviewed',member_id='C01-original',
        decisions=decisions, scope='Named structures only; no automatic full-frame or dataset admission.',
        inputs={str(p):prior.file_sha256(p) for p in deps}))


if __name__=='__main__': print(run()['status'])
