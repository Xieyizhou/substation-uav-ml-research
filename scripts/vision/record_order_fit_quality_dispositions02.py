"""Authored remaining-frame quality decisions; not an automatic pass rule."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior

SUPPORTED={
 'U02':'变压器仍有面板与端子，近柜主体和基座完整；普通背景柜体已解析。接受侧背外壳限制。',
 'U03':'柜体完整背面与基座可见，身份有来源依据；可作封闭外观监督，不声称凭该视图可独立识别类别。',
 'U04':'变压器保留主体及三个端子，近远两柜有可辨主体，远柜遮挡限制保留。',
 'U06':'图缘变压器仍有顶面、端子和侧面，另一台完整；远柜主体及基座可见，额外背景已核定。',
 'U07':'变压器主体端子完整，后柜主体及底座仍可见，仅局部遮挡。',
 'U09':'变压器面板与端子清楚，开关柜主体背侧面保留，前景普通柜体已解析。',
 'U16':'变压器清楚，开关柜虽左下截断仍保留侧面、顶面及底座；以截断外壳样本记录。',
 'U18':'左缘变压器仍有两个端子、主体与基座，另一台完整；小柜主体及基座可见，背景已区分。',
 'U21':'变压器虽右缘截断仍有三个端子、面板部分及基座；另一柜主体完整，底缘背景片段来源已有支持。',
 'U33':'两台变压器可辨，左缘开关柜有侧面和基座；右侧背景柜体及左缘普通柜体基座有来源支持，不作像素认证。',
 'U38':'近变压器截断但保留多个端子、顶面及主体，远变压器可辨，电容器大部外壳可见；背景已逐图核定。',
 'U39':'背景变体中近变压器仍有顶面端子及主体部分，远变压器可辨，电容器外壳大部可见；不传播原图像素认证。',
 'U40':'灰色变体中近变压器端子、顶面和主体部分可辨，远变压器及电容器外壳保留；背景来源已核定。',
}
HELD={
 'U05':'后方开关柜只露上部块体和顶面，下部被变压器遮挡；当前无法充分确认此标签的有效监督内容，整图暂缓。',
 'U36':'左缘变压器仅侧面及基座片段，顶端附件在画外；不能用另一台清楚变压器替代该标签的质量判断。',
 'U43':'右后柜截断并遮挡、近变压器大幅截断只见部分主体和一个端子；多结构归属仍弱，整图暂缓。',
 'U44':'近柜大幅图缘截断，同时另两柜受遮挡或截断；完整远柜不能补足其他标签，整图暂缓而非删除单框。',
 'U46':'左缘变压器仅侧面基座片段，另一近变压器只有截断顶面及端子而无主体侧面；全部标签无法共同通过。',
 'U48':'更远柜只剩上部并多重遮挡，左缘变压器仅一个端子及部分主体；暂缓完整帧，保留其他可见目标记录。',
}


def run():
    paths=[OUT/'remaining-review/review.json',OUT/'quality-isolation-v3.json',OUT/'full-frame-evidence-index.json',
           OUT/'u33-fragment-followup.json',OUT/'authored-quality-dispositions-01.json',Path(__file__).resolve()]
    review,q,index,followup,first=[prior.read(p) for p in paths[:5]]
    for r in (review,q,index,followup,first):prior.verify(r)
    byid={d['review_id']:d for d in review['decisions']};old={d['member_id'] for d in first['decisions']}
    holds={m['member_id'] for m in q['members'] if m['status']=='quarantined'}
    if set(SUPPORTED)&set(HELD):raise ValueError('Conflicting authored decisions')
    rows=[]
    for rid,reason in {**SUPPORTED,**HELD}.items():
        d=byid[rid]
        if d['member_id'] in old or d['member_id'] in holds:raise ValueError('Existing disposition collision')
        rows.append(dict(member_id=d['member_id'],review_id=rid,
            quality_status='whole_frame_held_content_evidence_insufficient' if rid in HELD else 'supported_for_bounded_research_with_recorded_limits',
            reason=reason,review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),
            image_sha256=d['image_sha256'],label_sha256=d['label_sha256'],page_sha256=d['page_sha256'],
            retained_label_limits=d['label_observations'],pixel_visibility_certified=False,training_eligible=False))
    dest=OUT/'authored-quality-dispositions-02.json'
    if dest.exists():
        r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='19_authored_quality_decisions_13_supported_6_held',decisions=rows,
        policy='No area cutoff, model-score selection, label edits or automatic ignore. Source-role and loader gates remain separate.',
        inputs={str(p):prior.file_sha256(p) for p in paths}))


if __name__=='__main__':print(run()['status'])
