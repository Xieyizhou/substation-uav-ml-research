"""Record explicit ten-image observations and bounded fit conclusion."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.diagnose_reviewed_target_fit import OUT,KEYS,prior,freeze,runtime

OBS={
'T01':('side_back','完整蓝色侧背面及基座清楚，面板不可见；与 M05/M08 的无面板块体有外形对应，但尺度和视角不同。',['M05','M08']),
'T02':('panel_side','柜体侧面、深色面板和顶部清楚，近处基座贴近画面底缘；与 M14/M16 同属面板加侧面结构，非同视角同实例认证。',['M14','M16']),
'T03':('side_back_partial_overlap','蓝色背面和顶面可辨，右下有前景变压器边角；支持背侧形状覆盖，不等价于开发条件。',['M05','M08']),
'T04':('panel_top_high_angle','俯视角蓝色柜体、端部面板及基座清楚；相较 M14/M16 俯视更强，不能称视角已充分覆盖。',['M14','M16']),
'T05':('gray_panel','灰色柜体面板轮廓与顶面清楚，区别于开发中的蓝色侧面条件；仅支持面板结构训练拟合。',['M14','M16']),
'T06':('cylinder','灰色圆柱主体、顶边和方形底座可辨，后方有设备但未明显遮挡主体；与 M06/M26/M34 有结构对应。',['M06','M26','M34']),
'T07':('cylinder','较大圆柱侧面及底座完整可见，椭圆顶面窄；与开发圆柱同类外形，但拍摄角度不同。',['M06','M26','M34']),
'T08':('cylinder_high_angle','高角度圆柱椭圆顶面、主体和底座清楚，环境较简单，不能替代复杂开发上下文。',['M06','M26','M34']),
'T09':('neutral_cylinder','中性灰圆柱与底座完整，俯视明显，周围杆体未遮挡目标；支持材质变体训练拟合而非泛化。',['M06','M26','M34']),
'T10':('cylinder_high_angle','高俯视圆柱和椭圆顶面清楚，尺度大、基座下缘接近边界；不代表所有小尺度圆柱。',['M06','M26','M34'])}

def run():
    p=freeze();ep=OUT/'evidence.json';sp=OUT/'summary.json';e=prior.read(ep);s=prior.read(sp)
    for r in (e,s):prior.verify(r)
    if set(OBS)!={t['target_id'] for t in p['targets']}:raise ValueError('Review coverage mismatch')
    decisions=[];now=datetime.now(timezone.utc).isoformat();members={m['member_id']:m for m in p['members']}
    for t in p['targets']:
        state,reason,dev=OBS[t['target_id']];m=members[t['member_id']];page=OUT/(t['target_id']+'.png')
        decisions.append(dict(target_id=t['target_id'],visual_structure=state,reason=reason,development_error_ids=dev,
            correspondence='visual_structural_analogy_only_not_same_instance_or_condition',review_nature='AI辅助审核',reviewed_at=now,
            pixel_visibility_certified=False,inputs={str(x):prior.file_sha256(x) for x in (page,Path(m['image_path']),Path(m['label_path']))}))
    totals={};deps=[OUT/'protocol.json',ep,sp,Path(__file__).resolve()]
    for k in KEYS:
        path=OUT/(k+'.json');r=prior.read(path);runtime.validate(r,k,p);deps.append(path)
        if any(p['actual_exposures'][k].get(m['member_id'],0)<=0 for m in p['members']):raise ValueError('Unexposed fit member')
        totals[k]=dict(full_truth_instances=sum(len(x['truth']) for x in r['rows']),matched_instances=sum(len(x['matches']) for x in r['rows']),unmatched_predictions=sum(x['unmatched_prediction_count'] for x in r['rows']))
    dest=OUT/'completion.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='bounded_fit_and_visual_correspondence_complete',decisions=decisions,full_label_fit=totals,
        targets=len(p['targets']),images=len(p['members']),target_model_hits=sum(v['hit'] for t in s['targets'] for v in t['outcomes'].values()),
        target_model_events=sum(len(t['outcomes']) for t in s['targets']),
        conclusion='Selected exposed structures fit correctly in all six endpoints; prioritize controlled viewpoint/scale/context transfer diagnosis rather than assuming insufficient optimization steps.',
        limits=['Deterministic median/maximum size sample, not full-pool fit or small/occluded coverage.',
                'Shared scene/assets; structural analogy does not isolate viewpoint, scale, context or material causes.',
                'Original images tested, not every brightness-transformed training tensor. No new training, sampling or label revision.'],
        inputs={str(x):prior.file_sha256(x) for x in deps}))

if __name__=='__main__':
    r=run();print(r['status'],r['target_model_hits'],r['target_model_events'])
