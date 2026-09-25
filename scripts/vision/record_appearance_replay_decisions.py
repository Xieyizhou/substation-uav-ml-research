"""Explicit AI decisions after inspecting all 21 certified instance crops."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.replay_appearance_recovery import OUT,read,save,file_sha256,verify_tree,ROOT

OBS={
 'P001':('insufficient','仅露顶面与窄上沿，前景柜体遮住主体／面板。'),
 'P002':('identifiable','掩码确认连续顶面、较宽正面和面板上部属于同一柜体；虽遮挡下部，内容可辨。'),
 'P003':('insufficient','大面积顶面但主体大部分出画，仅窄正面和面板片段；大像素数不等于内容充分。'),
 'P004':('identifiable','掩码确认边缘可见连续宽侧面、转角和基座；归属明确，非仅基座。'),
 'P005':('insufficient','掩码几乎仅覆盖出画柜体顶面，缺少其他可辨结构。'),
 'P006':('insufficient','掩码只有前景变压器左侧窄竖条，主体被遮挡。'),
 'P007':('insufficient','掩码仅为前景变压器右侧极小边缘片段，不能形成足够对象内容。'),
 'P008':('insufficient','钢灰图中仅后排柜体顶面和窄上沿可见。'),
 'P009':('identifiable','钢灰图中较宽正面和顶面归属明确，面板轮廓较弱但可观察。'),
 'P010':('insufficient','钢灰图中主内容仍为近处顶面，面板只有出画片段。'),
 'P011':('identifiable','钢灰图中侧面、转角和基座形成连续同实例内容。'),
 'P012':('insufficient','钢灰图中只见大面积顶面，不因灰色或面积大放行。'),
 'P013':('insufficient','钢灰图中目标只剩窄竖条，框中大部分为前景变压器。'),
 'P014':('insufficient','钢灰图中只有极小边缘像素属于后方柜体。'),
 'P015':('insufficient','暖暗图中后排柜体仍仅顶面和窄上沿。'),
 'P016':('identifiable','暖暗图中连续顶面及较宽正面仍可辨认，掩码排除前景柜体。'),
 'P017':('insufficient','暖暗图中出画顶面主导，正面／面板片段不足。'),
 'P018':('identifiable','暖暗图中侧面和基座仍归属于同一实例，内容连续。'),
 'P019':('insufficient','暖暗图中可见区域仍只有大面积顶面。'),
 'P020':('insufficient','暖暗图中只见遮挡缝隙的窄竖条。'),
 'P021':('insufficient','暖暗图中只见前景右边的小片段，不足以辨识后方柜体。'),
}

def main():
    path=OUT/'decisions.json'
    if path.exists():verify_tree(path);print('VERIFIED_EXISTING');return
    verify_tree(OUT/'review-manifest.json');m=read(OUT/'review-manifest.json')
    if set(OBS)!={x['review_id'] for x in m['items']}:raise ValueError('Explicit review coverage mismatch')
    items=[]
    for item in m['items']:
        state,reason=OBS[item['review_id']]
        items.append({**item,'review_status':'实例可见且内容可辨识' if state=='identifiable' else '实例有可见证据，但内容不足','review_nature':'AI辅助审核','reviewed_at':datetime.now(timezone.utc).isoformat(),'reason':reason,'component_evidence':'unknown','component_note':'RGB形态描述不等于组件级掩码认证；当前只认证设备实例。'})
    from scripts.vision.verify_experiment_baseline import verify
    save(path,dict(status='visibility_diagnosis_complete_pilot_not_admitted',items=items,identifiable=6,content_insufficient=15,certified_frames=9,total_reviewed_boxes=21,baseline=verify(ROOT/'config/perception/visual_experiment_baseline_v1.json'),next_action='Preserve current pilot and labels; do not expand remaining 16. Revise new-pose screening to inspect all full-image supervised instances, not only planned target visibility. Freeze a separate bounded pose pilot before further collection; use actual masks and explicit review, not an unvalidated area threshold.',inputs={str(p):file_sha256(p) for p in (OUT/'review-manifest.json',Path(__file__))}))
    print('REVIEW_COMPLETE 6 identifiable / 15 content insufficient; pilot not admitted',flush=True)

if __name__=='__main__':main()
