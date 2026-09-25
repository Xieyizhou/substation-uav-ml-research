"""Explicit full-image screening and twelve localized pending targets; no pass-all."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.prepare_redistribution_review import OUT,COUNTS,ROOT,read,verify,frozen,file_sha256
from scripts.vision.redistribution_risk_crops import TARGETS
from src.ml.artifacts import object_sha256

OBS=[
'前景灰建筑不作为设备；后方变压器有遮挡，右侧圆柱受黑杆遮挡。',
'近处圆柱清楚；后方两变压器局部被其遮挡。',
'中远电容器组清楚；左右变压器接触图缘，仍见大块主体。',
'近处柜面板可辨，下缘截断；后方电容器组部分遮挡。',
'左缘电容器组框为窄带，主要见地面及边缘碎片，需要局部核验。',
'圆柱和柜面板可辨，左下前景柜体截断。',
'电容器组宽主体清楚；右侧变压器大面积截断，仍有顶部附件。',
'近处变压器主体清楚；后排柜体小且遮挡，右缘圆柱部分截断。',
'右上变压器框只见浅色附件状圆柱，主体不可辨，需归属核验。',
'近处圆柱清楚；后方变压器被遮挡，右侧柜状设备可辨。',
'电容器组宽主体和底座清楚，后方变压器部分遮挡。',
'宽电容器组及侧背柜体可辨；左侧变压器部分截断。',
'圆柱主体清楚；左缘柜体截断，后排柜体有交叠。',
'三个设备宽主体及基座可辨，变压器顶部附件短小。',
'后方圆柱清楚；近处变压器下缘截断，左缘柜体部分可见。',
'近处圆柱清楚；右缘变压器截断但主体和附件可见。',
'右缘变压器框只留狭窄侧身和底座，图框包含大量地面。',
'近处变压器仅上部入画；中部电容器组局部遮挡。',
'电容器组可辨；杆体遮挡变压器及右侧柜体。',
'近处变压器主体截断，左侧变压器被黑杆遮挡；电容器组可辨。',
'俯视电容器组宽顶部和主体清楚，后方柜体可辨。',
'左侧电容器组边界截断但大块主体可辨，其余设备清楚。',
'灰材质柜体及后方圆柱有前景黑杆和设备遮挡，非像素级认证。',
'前景柜体可辨；电抗器下部被柜体遮挡，圆柱上部可见。',
'灰材质柜体和变压器可辨，后方圆柱下部局部遮挡。',
'灰材质柜体面板对比低但可见；圆柱上部在后方露出。',
'背景变体中柜面板可辨；后方圆柱部分遮挡。',
'背景变体前景柜体清楚，后方圆柱下部被遮挡。',
'黑杆穿过中部柜体框，后方圆柱及其他柜体局部可见。',
'柜体面板和变压器可辨；圆柱被变压器局部遮挡。',
'电容器组框与前景变压器顶部和附件重叠，需要局部归属核验。',
'圆柱清楚；近处变压器和左缘柜体部分截断。',
'右缘电容器组只留极窄片段；需核验可辨识内容。',
'单个俯视圆柱，顶面、主体和基座清楚。',
'圆柱清楚，后排多个柜体交叠；近处变压器大幅截断。',
'圆柱和柜面板可辨；左右前景设备截断。',
'近处变压器上部和附件清楚，后方电容器组可辨。',
'电容器组部分被前景变压器遮挡，左侧变压器截断。',
'电容器组在右缘且被近处变压器遮挡，仍见宽上身。',
'下缘开关柜框只见顶部平面，需要内容充分性核验。',
'多设备密集交叠，电容器组主体清楚；非计划远柜体存在遮挡风险，未做像素级认证。',
'单个圆柱，主体与底座清楚。',
'电容器组部分遮挡，近处变压器只上部入画。',
'电容器组小框内含前景变压器边缘，实例内容不能直接确认。',
'近处圆柱清楚；右侧变压器截断。',
'左下变压器框仅下部与底座/地面，需要局部核验。',
'电容器组框内以变压器顶部和侧壁为主，需实例归属核验。',
'电容器组顶部、宽主体及底座清楚。',
'电容器组被黑杆局部遮挡；后方变压器被前景蓝柜遮挡。',
'近处圆柱清楚；右缘变压器部分截断。',
'电容器组宽主体清楚，阴影不遮去主要内容。',
'圆柱位于右缘且部分被前景变压器遮挡，后方柜体密集交叠。',
'电容器组框内混入变压器浅色附件，需核验后方目标可辨识内容。',
'左下变压器框只见底座及极少下身，需要内容核验。',
'近处柜体和后方圆柱主体清楚，右侧变压器截断。',
'圆柱主体清楚，但右缘变压器框为极窄底座/地面片段。',
]
RISK={
'P05':'极窄电容器组框内主要为地面网格和少量边缘碎片，无法确认可辨识设备主体。',
'P09':'变压器框仅见浅色圆柱附件状内容，不能仅凭此确认变压器主体；组件归属unknown。',
'P17':'变压器仅露右缘窄侧身及底座，内容不足风险。',
'P31':'前景变压器顶部和附件占据框的大部分，后方电容器组可辨识范围尚未确认。',
'P33':'电容器组框为右缘窄带，主要为墙/地面和设备极小片段。',
'P40':'开关柜框只含下图缘蓝色顶部平面，缺少可辨识主体/面板。',
'P44':'框内多块侧面与前景边缘交叠，电容器组实例可辨识范围待确认。',
'P46':'变压器框集中于底座、地面和少量下身，内容充分性待确认。',
'P47':'电容器组框主要覆盖前景变压器顶部及侧壁，后方实例尚无独立像素证据。',
'P53':'电容器组框包含前景变压器顶部及浅色附件，不能把附件当成电容器组证据。',
'P54':'变压器框只见底座、地面和少量下部，不能确认足够主体内容。',
'P56':'变压器框为右图缘极窄条，主要见地面、底座边缘；主体不可辨识。',
}

def validate(e,decisions):
    index={r['event_id']:r for r in e['events']}
    if len(decisions)!=len(index) or {d['event_id'] for d in decisions}!=set(index):raise ValueError('Missing or duplicate decision')
    for d in decisions:
        r=index[d['event_id']]
        if d['source_identity']!=object_sha256(r) or not d['reason'] or not d['reviewed_at']:raise ValueError('Stale/incomplete review')
        if d['review_nature']!='AI辅助审核' or d['training_approved'] is not False:raise ValueError('No automatic training approval')
        if file_sha256(r['evidence_path'])!=d['evidence_sha256'] or file_sha256(r['member']['image_path'])!=d['image_sha256']:raise ValueError('Stale pixels')

def main():
    e=read(OUT/'evidence.json');verify(e);c=read(OUT/'risk-crops.json');verify(c);crops={x['event_id']:x for x in c['targets']}
    if len(OBS)!=56:raise ValueError('Incomplete explicit observations')
    now=datetime.now(timezone.utc).isoformat();ds=[]
    for r,reason in zip(e['events'],OBS):
        eid=r['event_id'];risk=RISK.get(eid)
        d=dict(event_id=eid,member_id=r['member']['member_id'],reason=reason,status='pending_target_evidence' if risk else 'screened_no_additional_priority_flag_not_admission',
            review_nature='AI辅助审核',reviewed_at=now,source_identity=object_sha256(r),evidence_sha256=r['evidence_sha256'],image_sha256=r['member']['image_sha256'],
            full_image_viewed=True,all_boxes_individually_certified=False,pixel_visibility_certified=False,training_approved=False,
            historical_source_gaps=r['source']['gaps'])
        if risk:d.update(risk_reason=risk,risk_target=crops[eid],source_annotation=r['source']['annotations'][TARGETS[eid]])
        ds.append(d)
    validate(e,ds)
    paths=[OUT/'evidence.json',OUT/'risk-crops.json',Path(__file__)]
    frozen(OUT/'review.json',dict(status='quality_gate_blocked_no_sequence',decisions=ds,pending_events=sorted(RISK),
        labels_modified=False,inputs={str(x):file_sha256(x) for x in paths}))
    print('SCREENED56_PENDING12_NO_SEQUENCE_NO_TRAINING')
if __name__=='__main__':main()
