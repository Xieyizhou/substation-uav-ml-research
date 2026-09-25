"""Explicit observations recorded after viewing six full images and 24 mask crops."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.replay_retained_bridge_pilot import OUT, read, save, file_sha256, verify_tree
from scripts.vision.trace_retained_bridge_sources import ROOT, baseline_verify

# Individually inspected observations, not a threshold-based approval generator.
OBS={
 'B001':('insufficient','左边界处仅顶部窄条属于该实例；下方大面积柜体属于前景另一实例，内容不足。'),
 'B002':('identifiable','右侧截断，但较大主体、顶部三个突出结构及基座仍清楚；不是仅边缘碎片。'),
 'B003':('identifiable','左侧截断；仍可辨认大面积柜体、前面板轮廓、侧面和基座。'),
 'B004':('identifiable','中央圆柱主体及基座完整可辨，无明显前景遮挡。'),
 'B005':('insufficient','中性材质下仍仅露顶部窄条；主要框内区域为前景柜体，不足以辨认此实例内容。'),
 'B006':('identifiable','中性材质下大块主体、顶部突出结构和基座仍可辨，右边界截断未消除这些内容。'),
 'B007':('identifiable','中性材质下可见主体、面板边线与侧面及基座；左边界截断。'),
 'B008':('identifiable','中性材质圆柱主体及基座完整可辨。'),
 'B009':('insufficient','原始条件下仅左上窄条标为本实例；完整框内大面积是遮挡物。'),
 'B010':('identifiable','原始条件下主体、顶部三个突出结构和基座可辨；右边界截断。'),
 'B011':('identifiable','原始条件下大面积柜体、面板轮廓和基座可辨，保留左侧截断说明。'),
 'B012':('identifiable','原始条件下圆柱及基座完整可辨。'),
 'B013':('identifiable','远处柜体较大连续表面、侧面和基座可辨；左侧局部被前景遮挡，前面板特征未确认。'),
 'B014':('insufficient','仅上沿横条及两个顶部突出片段可见；大部分主体被前方设备与灰色结构遮挡。'),
 'B015':('insufficient','灰色前景结构遮住绝大部分实例，只露右侧窄片及部分基座，内容不足。'),
 'B016':('identifiable','可见连续箱状主体两面和基座，右侧被遮挡；仅确认实例内容可辨，不确认类别独有结构。'),
 'B017':('identifiable','中性条件下远处柜体主体与基座可辨，左侧局部遮挡；类别区别线索仍未确认。'),
 'B018':('insufficient','中性条件下仍只见顶部横条及两个突出片段，主体被前景遮挡。'),
 'B019':('insufficient','中性条件下仅右侧窄片和部分基座露出，框内主要为灰色遮挡结构。'),
 'B020':('identifiable','中性条件下连续箱状主体与基座可辨，右侧遮挡；不凭实例映射声称有独特类别外形。'),
 'B021':('identifiable','原始条件下远处柜体主体两面及基座可辨；局部遮挡且前面板特征未确认。'),
 'B022':('insufficient','原始条件下只有顶部横条和两个突出片段，主体大部分被遮挡。'),
 'B023':('insufficient','原始条件下仅右侧窄片和基座一部分可见，其余被灰色前景遮住。'),
 'B024':('identifiable','原始条件下连续箱状主体和基座可辨，右侧遮挡；可见性结论不等于类别可分性认证。'),
}

def validate(items, observations):
    ids=[i['review_id'] for i in items]
    if len(set(ids))!=len(ids) or set(ids)!=set(observations):raise ValueError('Missing/duplicate review coverage')
    for item in items:
        for key,hash_key in [('source_image','source_sha256'),('crop_path','crop_sha256')]:
            if file_sha256(item[key])!=item[hash_key]:raise ValueError('Stale review evidence')
        if observations[item['review_id']][0] not in ('identifiable','insufficient'):raise ValueError('Unknown decision')

def main():
    dest=OUT/'decisions.json'
    if dest.exists():verify_tree(dest);print('VERIFIED_EXISTING');return
    manifest=OUT/'review-manifest.json';verify_tree(manifest);items=read(manifest)['items'];validate(items,OBS)
    reviewed=[]
    for item in items:
        state,reason=OBS[item['review_id']]
        reviewed.append({**item,'review_status':'实例可见且内容可辨识' if state=='identifiable' else '实例有可见证据，但内容不足',
            'review_nature':'AI辅助审核','reviewed_at':datetime.now(timezone.utc).isoformat(),'reason':reason,
            'component_evidence':'unknown','class_discriminability':'not_certified',
            'component_note':'主体、面板和基座为图像形态描述；没有组件级实例掩码认证。',
            'training_admitted':False,'promotable':False})
    groups={key:dict(status='held_for_full_image_supervision_risk',member_ids=sorted({i['member_id'] for i in items if i['lineage_id']==key})) for key in {i['lineage_id'] for i in items}}
    save(dest,dict(status='pilot_diagnosis_complete_both_groups_held',items=reviewed,groups=groups,
        counts=dict(Counter(i['review_status'] for i in reviewed)),remaining_frames_not_replayed=33,
        baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json'),
        limitations=['Two selected lineages are diagnostic cases, not an unbiased prevalence estimate.',
            'Visible content is not proof of class separability or training eligibility.',
            'Original full_2d boxes can include occluded extent; this alone does not establish a corrupt label.'],
        next_action='Audit remaining 11 lineages before designing an independent revised pool; never remove a box alone to turn a still-visible target into background.',
        inputs={str(p):file_sha256(p) for p in (manifest,Path(__file__))}))
    print('REVIEW_COMPLETE',Counter(i['review_status'] for i in reviewed))

if __name__=='__main__':main()
