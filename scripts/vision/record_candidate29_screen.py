"""Explicit full-frame AI screening notes, not per-box training approval."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.audit_candidate29_completeness import OUT,prior

NOTES={
'A01':'单一箱状设备及平台，图内完整；本轮实例集合核对无冲突。',
'A02':'近处灰色结构遮挡背景，电抗器另被杆体局部遮挡；未凭灰色结构外形改称目标设备。',
'A03':'前景圆柱清晰，后方两变压器相互遮挡；沿用L01已记录限制。',
'A04':'左右近处设备均有边缘截断，中间小设备与背景结构须按实例映射区分，不能仅按箱体外观认类。',
'A05':'单一箱状主体与平台清晰，影子未作为目标像素。',
'A06':'前景带面板结构和后方设备重叠；无成员冲突不代表遮挡内容充足。',
'A07':'右侧近景设备严重截断；中心主体清晰，左侧结构不能凭外观推断标签遗漏。',
'A08':'圆柱清晰，右后变压器仅上部及附件可见，属于已记录重遮挡限制。',
'A09':'前后两个设备有局部重叠，主体可见；没有新增实例成员冲突。',
'A10':'底缘近处结构、左边截断设备和远处小目标并存；保留尺度及截断限制。',
'A11':'多个箱状设备及杆体，主体分离较清楚；杆体邻接区域不算设备像素。',
'A12':'底缘前景局部和后方多设备并存，右后目标存在遮挡；不认证所有标签可辨识内容充足。',
'A13':'电抗器清晰，右侧变压器截断但有大片主体；沿用L03限制。',
'A14':'左侧变压器与杆体邻接，右后设备被杆体遮挡；实例核对无新增冲突。',
'A15':'近景右侧变压器大幅截断，左侧变压器被杆体遮挡；需要保留这些质量限制。',
'A16':'左下近处箱体截断、右侧设备亦截断；原与重放框一致，但框下界浮点越界，不能给予严格门禁通过。',
'A17':'灰色桥接变体，多实例与前景杆体；左边截断、后方圆柱局部遮挡，不增加独立场景数量。',
'A18':'与桥接同源位姿的背景变体，边缘局部和遮挡关系不因颜色改变而解除。',
'A19':'同源桥接原条件，多设备和杆体重叠；成员一致不是新的独立场景证据。',
'A20':'近处主体、顶部附件及平台清晰；背景带面板结构不凭外观自动标成目标。',
'A21':'俯视圆柱顶部、侧面与平台清楚，沿用L04已审核限制。',
'A22':'前景变压器占画面较大且左下截断，后方目标重叠；保留截断而非全框可见性解释。',
'A23':'左缘变压器截断，右侧主体较完整；中心杆体为独立前景结构。',
'A24':'前景主体与顶部附件清楚但底缘接触，右后设备被部分遮挡。',
'A25':'对应L05。右边缘transformer_mid的侧面及底部片段在对齐实例掩码中存在，历史标签遗漏；整图风险待处理。',
'A26':'近处箱状设备清晰，后方杆体和左缘结构需保持来源区分；本轮无新增成员冲突。',
'A27':'右侧后方变压器下部被近处结构遮挡，左侧目标与杆体邻接；不将可见上部视为完整主体。',
'A28':'对应I17。左下边缘transformer_se侧面局部在对齐掩码中存在，历史标签遗漏；整图风险待处理。',
'A29':'多实例密集，前景设备及右边圆柱均截断，后方小目标遮挡明显；本轮实例集合一致但非完整内容批准。',
}

def validate_screen(decisions,expected):
    ids=[d['pair_id'] for d in decisions]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):raise ValueError('Missing/duplicate screening decisions')
    if any(not d.get('reason') or d.get('training_approved') is not False for d in decisions):raise ValueError('Invalid screening scope')

def main():
    ep=OUT/'evidence/manifest.json';pp=OUT/'protocol.json';vp=OUT/'review-pages/manifest.json'
    e,p=prior.read(ep),prior.read(pp)
    for x in [e,p,prior.read(vp)]:prior.verify(x)
    events={x['pair_id']:x for x in e['events']};decisions=[]
    for f in p['frames']:
        pid=f['pair_id'];ev=events[pid];ip=Path(ev.get('full',f['source_image']))
        decisions.append(dict(pair_id=pid,member_id=f['member_id'],reason=NOTES[pid],review_nature='AI辅助审核',
            reviewed_at=datetime.now(timezone.utc).isoformat(),training_approved=False,
            status='confirmed_omission_risk' if pid in ('A25','A28') else 'boundary_numeric_gap' if pid=='A16' else 'no_membership_conflict_detected_not_quality_approval',
            image_sha256=prior.file_sha256(ip),scope='Whole-frame screening and aligned instance membership; not all-label content approval.'))
    validate_screen(decisions,NOTES)
    paths=[ep,pp,vp,Path(__file__).resolve()]
    prior.frozen(OUT/'screening.json',dict(status='29_full_frame_screenings_recorded',decisions=decisions,
        training_ready=False,training_started=False,inputs={str(x):prior.file_sha256(x) for x in paths}))

if __name__=='__main__':main()
