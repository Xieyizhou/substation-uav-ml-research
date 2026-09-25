"""Explicit review of twelve extremal-gain preview triplets."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.prepare_brightness_transfer import OUT,read,verify,frozen,file_sha256
OBS={
'P01':'远处矩形目标及近处设备轮廓在三档均可辨；暗档面板更暗但未消失。',
'P02':'完整圆柱顶面、侧面和底座在两端增益下均保留，杆体位置不变。',
'P03':'近景圆柱轮廓及基座保持，暗档曲面细节减少但主体仍清楚。',
'P04':'独立矩形块体顶面、侧面和基座清楚，阴影位置不变。',
'P05':'前景块体与右图缘圆柱维持原截断关系；不把边界风险重新认证为完整。',
'P06':'前景遮挡及右图缘圆柱保持，亮度不能修复原可见性限制。',
'P07':'青色负例块体、基座、杆和地面格线在暗亮端仍可分辨。',
'P08':'细杆与横件、围墙边界仍可辨，亮端天空趋白但杆轮廓保留。',
'P09':'灰色块体大面、基座和地面保留，三档未出现几何改动。',
'P10':'灰色块体棱边、侧部深色面板和基座仍可辨，亮端对比有变化。',
'P11':'前景竖杆及后方块体重叠关系保持，暗档杆体没有完全并入背景。',
'P12':'局部底座、块体边缘和地面格线三档均可见；只是原局部负例的亮度变化。',
}
def main():
    path=OUT/'preview/evidence.json';e=read(path);verify(e)
    if set(OBS)!={r['event_id'] for r in e['rows']}:raise ValueError('Preview coverage mismatch')
    ds=[dict(event_id=r['event_id'],evidence_sha256=r['evidence_sha256'],image_sha256=r['member']['image_sha256'],review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),reason=OBS[r['event_id']],decision='bounded_preview_usable',pixel_visibility_certified=False,training_approved=False) for r in e['rows']]
    frozen(OUT/'preview/review.json',dict(status='explicit_sample_review_complete',decisions=ds,scope='Only twelve displayed sample triplets. Not all augmented frames or label admission.',inputs={str(x):file_sha256(x) for x in [path,Path(__file__)]}))
if __name__=='__main__':main()
