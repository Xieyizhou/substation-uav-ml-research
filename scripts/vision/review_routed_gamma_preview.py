"""Record individually observed six preview pages, not automatic pool approval."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.preview_routed_gamma import OUT,source
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

NOTES={
 'cool':'俯视灰块顶部、侧面和基座边界保留；青色设备三柱及顶部可辨。暗端正面很暗，不能补认不可见的面板。',
 'gray035':'近处柜状结构面板、顶部、侧面与左侧三柱结构仍可辨；亮端对比降低、暗端阴影增强。远处杆件遮挡不能认证为清晰。',
 'gray_all_body':'全灰外观下三柱、顶部和基座轮廓保持，中央块体侧背面可见；暗端正面近黑，不把侧背面当正面证据。',
 'gray_target_body':'中央灰块顶侧面与青色三柱结构保持边界；暗端正面明显变暗，批准仅限有界变换测验。',
 'neutral':'近垂直俯视下三柱及大顶部轮廓仍可辨，右侧块体面板侧面与基座保持；亮端变淡，视角本身限制正面内容。',
 'warm':'暖色块体顶侧面和基座、青色设备柱体保留；暗端正面很暗但边缘可见，不宣称像素级可见性认证。',
}

def run():
    ep=OUT/'preview-v1/evidence.json';e=source.checked(ep);dest=OUT/'preview-v1/review.json'
    if dest.exists():return source.checked(dest)
    decisions=[]
    for r in e['rows']:
        decisions.append(dict(member_id=r['member_id'],decision='bounded_transform_test_allowed',reason=NOTES[r['variant']],review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),**{f:r[f] for f in ('image_sha256','label_sha256','page_sha256')}))
    return write_record(dest,dict(decisions=decisions,limits='Six previously visually inspected deterministic examples only; representative resized previews, not exact padded loader tensors. Not full-pool approval, asset identity certification or physical relighting.',training_admitted=False,promotable=False,inputs={str(q.resolve()):file_sha256(q) for q in (ep,Path(__file__))}))

if __name__=='__main__':print(len(run()['decisions']))
