"""Explicit observations after viewing all seven full frames and 35 crops."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.reviewed_hold_control import OUT,prior

NOTES={
'Q01':['封闭电容器完整主体、顶面及底座清楚','变压器宽主体三柱与底座清楚','左边截断但宽主体与三柱底座可辨','右下截断但宽主体顶面及附件片段可见'],
'Q02':['开关柜宽正侧面顶部底座，左下被变压器局部遮挡','开关柜宽侧壁顶部底座清楚','电容器完整侧面顶部底座清楚','变压器完整宽主体及三柱底座','左边截断但主体附件底座可辨'],
'Q03':['开关柜完整宽侧壁顶部底座','电容器完整宽主体顶部底座','变压器完整宽主体底座，顶部附件叠列'],
'Q04':['灰色柜体面板宽主体顶部底座清楚','左边截断但宽侧壁顶部底座可辨','后排柜体面板上部顶部可见，下部前景遮挡','圆柱主体顶部可辨，右下被变压器遮挡','变压器完整宽主体三柱底座'],
'Q05':['青色柜体面板宽主体顶部底座清楚','左边截断但宽侧壁顶部底座可辨','后排柜体面板上部顶部可见，下部前景遮挡','圆柱主体顶部可辨，右下被变压器遮挡','变压器完整宽主体三柱底座'],
'Q06':['青色柜体面板宽主体顶部底座清楚','左边截断但宽侧壁顶部底座可辨','后排柜体面板上部顶部可见，下部前景遮挡','圆柱主体顶部可辨，右下被变压器遮挡','变压器完整宽主体三柱底座'],
'Q07':['柜体宽正面顶面，右下前景遮挡','柜体右侧宽平面顶部底座可辨，左边杆体遮挡','柜体完整宽主体顶部底座','柜体宽上部顶部可辨，右下被前景柜体遮挡','后方柜体顶部及上部平面可见，下部前景柜体遮挡','右缘截断但宽圆柱曲面顶面可辨，下部前景遮挡','近处变压器宽顶面三柱及正侧面，右下截断','左边截断和杆遮挡，宽主体顶部附件仍可辨']}


def main():
    ep=OUT/'review-evidence.json';e=prior.read(ep);prior.verify(e);ds=[]
    if set(NOTES)!={x['review_id'] for x in e['events']}:raise ValueError('Review scope changed')
    for f in e['events']:
        notes=NOTES[f['review_id']]
        if len(notes)!=len(f['labels']):raise ValueError('Reinspection required')
        for l,note in zip(f['labels'],notes):
            ds.append(dict(review_id=f['review_id'],member_id=f['member_id'],runtime_label=l['runtime_label'],reason=note,
                status='identifiable_content_with_recorded_limits',review_nature='AI-assisted',reviewed_at=datetime.now(timezone.utc).isoformat(),
                evidence_identity=e['identity'],page_sha256=f['page_sha256'],label=l,training_admitted=False,promotable=False))
    dest=OUT/'quality-review.json'
    if dest.exists():prior.verify(prior.read(dest));return
    prior.frozen(dest,dict(status='seven_increment_members_reviewed_no_new_hold',decisions=ds,new_risk_members=[],
        scope='All existing labels and mapped equipment instance coverage in seven independently aligned frames; not a full-pool admission.',
        background_note='Unlabelled blue cabinet-like background models exist in saved worlds (cabinet_center/cabinet_west); appearance alone is not a target-class identity. No mapped target instance missing from these aligned masks.',
        training_ready=False,training_started=False,inputs={str(x):prior.file_sha256(x) for x in (ep,Path(__file__))}))
    print('EXPLICIT_LABEL_DECISIONS',len(ds))


if __name__=='__main__':main()
