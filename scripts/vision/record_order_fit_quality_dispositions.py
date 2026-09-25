"""Authored frame-level judgments, separate from role and loader admission."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior

# Explicitly authored after inspecting prior per-label observations and source evidence.
SUPPORTED = {
 'U01':'单个变压器主体和端子清楚；额外柜体已解析为普通背景。',
 'U08':'两台变压器特征完整；开关柜侧背面为已确认实例的封闭外观，背景柜体已区分。',
 'U10':'两台变压器可辨，小柜仍有完整主体与底座；三处背景柜体已解析。',
 'U11':'变压器完整，柜体保留主体顶面及底座；接受封闭侧背视角，不声称面板可见。',
 'U12':'杆遮挡未消除变压器两侧及顶端线索；另一台清楚，柜体外壳足够，背景已解析。',
 'U13':'变压器虽有杆遮挡仍可辨；完整柜体背面及基座按封闭外观保留。',
 'U14':'变压器端子和面板可辨，柜体外壳完整；两处额外柜体为背景来源。',
 'U15':'两台变压器保留主体和端子，小柜有背面与基座；普通背景柜体已区分。',
 'U17':'变压器轻微截断仍有面板端子；柜体完整背面及基座可作封闭外观监督。',
 'U19':'变压器面板顶端可辨，小柜主体及底座完整；近景额外柜体为普通背景。',
 'U20':'变压器完整；柜体有主体和顶面，侧背面局限必须随数据保留。',
 'U23':'变压器完整，两个柜体均有可辨主体与基座，局部遮挡不等于仅基座。',
 'U24':'变压器主体端子可辨，两柜保留主体及底座，封闭外壳限制保留。',
 'U25':'变压器清楚，后柜有背侧面顶面而非孤立基座；额外柜体来源已核定。',
 'U26':'变压器可辨，后柜完整外壳和底座；前景普通柜体已区分。',
 'U27':'两台变压器完整，后柜主体顶面和基座可见；普通柜体已区分。',
 'U30':'变压器散热结构及端子清楚，后柜完整外壳，未见明确未框目标。',
 'U32':'两台变压器特征充分，远柜仍有主体和底座；普通背景柜体已解析。',
 'U34':'变压器完整，柜体完整封闭外壳，接受侧背视角限制。',
 'U35':'四个设备均保留主体；电容器只按来源确认的封闭箱体监督，不声称内部组件可见。',
 'U37':'变压器大部主体及端子可辨，电容器完整外壳；背景片段来源已核定。',
 'U41':'灰色变压器大部主体端子可辨，灰色电容器外壳及底座清楚，背景已逐变体核对。',
 'U42':'原配色变压器大部主体端子可辨，电容器外壳及底座清楚，背景已逐图核对。',
 'U51':'十一标签逐框已查，变压器及电抗器清楚，柜体和电容器接受封闭外壳限制；两处额外背景已解析。',
}


def run():
    ep=OUT/'remaining-review/review.json'; qp=OUT/'quality-isolation-v3.json'; ip=OUT/'full-frame-evidence-index.json'
    review,q,index=[prior.read(p) for p in (ep,qp,ip)]
    for r in (review,q,index):prior.verify(r)
    byid={d['review_id']:d for d in review['decisions']};holds={m['member_id'] for m in q['members'] if m['status']=='quarantined'}
    decisions=[]
    for rid,reason in SUPPORTED.items():
        d=byid[rid]
        if d['member_id'] in holds or any(x['status']=='insufficient_content' for x in d['label_observations']):
            raise ValueError('Authored decision conflicts with hold: '+rid)
        decisions.append(dict(member_id=d['member_id'],review_id=rid,
            quality_status='supported_for_bounded_research_with_recorded_limits',reason=reason,
            review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),
            image_sha256=d['image_sha256'],label_sha256=d['label_sha256'],page_sha256=d['page_sha256'],
            retained_label_limits=d['label_observations'],pixel_visibility_certified=False,
            training_eligible=False,remaining_gates=['source_role_isolation','final_dataset_freeze','real_loader_preflight']))
    dest=OUT/'authored-quality-dispositions-01.json'
    if dest.exists():
        r=prior.read(dest);prior.verify(r);return r
    deps=[ep,qp,ip,OUT/'c01-background-review.json',Path(__file__).resolve()]
    return prior.frozen(dest,dict(status='24_explicit_quality_dispositions_not_dataset_admission',decisions=decisions,
        inputs={str(p):prior.file_sha256(p) for p in deps}))


if __name__=='__main__':print(run()['status'])
