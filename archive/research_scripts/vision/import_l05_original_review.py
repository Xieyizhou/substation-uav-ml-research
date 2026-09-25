"""Import the explicitly authored eleven-box review; no blanket decisions."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.prepare_l05_compensation_review import OUT,prior

DECISIONS={
 ('L01','208'):'圆柱主体和平台清晰完整；邻近杆体不是本实例，掩码已区分。',
 ('L01','57'):'后方变压器右段主体及顶部附件可辨；左侧被前方变压器遮挡，不能称完整主体。',
 ('L01','64'):'右部大块主体、顶部两柱及底部平台可辨，左侧被电抗器遮挡。',
 ('L02','113'):'柜状长侧面与平台可辨，未见前面板，不据此外形重新赋予资产身份。',
 ('L02','208'):'圆柱主体及底座清晰完整，背景重叠对象由实例掩码区分。',
 ('L02','57'):'后方右段主体及附件可辨，前方同类实例遮挡左部；不是无遮挡样本。',
 ('L02','64'):'右部主体、顶部两柱及底部可辨，左部被圆柱遮挡。',
 ('L02','76'):'仅上部主体带与三个顶部附件可辨，下部被开关柜严重遮挡。与旧审核同样记录此限制；实例归属由对齐掩码确认，不把框内柜体当作变压器。',
 ('L03','208'):'圆柱主体及平台清晰，未见主体遮挡或图缘截断。',
 ('L03','64'):'右边缘截断，但可辨大片主体侧面、顶部附件和平台；不同于L05仅窄边片段。',
 ('L04','208'):'俯视下主体顶部、侧面及平台清晰，图内完整，无明显前景遮挡。',
}

def main():
    ep=OUT/'evidence/manifest.json';e=prior.read(ep);prior.verify(e)
    targets={(f['pair_id'],t['runtime_label']):(f,t) for f in e['events'] for t in f['targets']}
    if set(targets)!=set(DECISIONS):raise ValueError('Review coverage changed')
    decisions=[]
    for key,reason in DECISIONS.items():
        f,t=targets[key]
        decisions.append(dict(pair_id=key[0],runtime_label=key[1],member_id=f['member_id'],instance=t['instance'],
            status='identifiable_geometry_with_recorded_limits',reason=reason,review_nature='AI辅助审核',
            reviewed_at=datetime.now(timezone.utc).isoformat(),component_identity='unknown_no_component_ID',
            full_image_sha256=prior.file_sha256(Path(f['full'])),crop_sha256=prior.file_sha256(Path(t['crop']))))
    prior.frozen(OUT/'original-review.json',dict(status='four_originals_explicitly_reviewed_with_limits',decisions=decisions,
        training_ready=False,scope='Original images only; no lighting-variant approval or whole-pool risk certification.',
        inputs={str(x):prior.file_sha256(x) for x in [ep,Path(__file__).resolve()]}))

if __name__=='__main__':main()
