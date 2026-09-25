"""Explicit observations after viewing all twelve full-image overlays."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.prepare_appearance_recovery_candidates import OUT,ROOT,read,save,file_sha256,verify_tree

OBS={
 'capacitor_bank':{'transformer_se':('visible','右边缘截断，但箱体主体、顶面附属结构和基座仍可见。'),'capacitor_east':('visible','左边被杆遮挡、下方被前景箱体遮挡；仍有连续宽主体和顶面。')},
 'switchgear':{'west_switchgear_04':('held','后排柜体被前排遮挡，主要露出顶面及窄面板，需核验可见内容。'),'west_switchgear_03':('held','中排柜体的下部被前排遮挡，顶面占主要可见区域；保留内容风险。'),'west_switchgear_02':('held','底部出画，主要是大面积顶面和极窄正面，不能仅因大框而认定内容充分。'),'transformer_mid':('visible','完整宽主体、顶部三处附属结构和基座可见。'),'switchgear_west':('visible','计划柜体主体、前面板与基座清楚；钢灰条件面板对比降低但轮廓可见。'),'reactor_north':('visible','圆柱主体上半段在计划柜体后方可见，下半段遮挡；尚非空框。')},
 'reactor':{'west_switchgear_04':('held','最左侧截断柜体只露出部分侧边和基座，需要确认内容是否足够。'),'transformer_mid':('visible','右侧截断，但仍有宽主体和基座。'),'switchgear_west':('held','前景仅有大面积顶面，正面几乎出画；需逐实例可见性核验。'),'reactor_north':('visible','计划圆柱主体和基座完整可辨，无明显前景遮挡。')},
 'transformer':{'west_switchgear_04':('visible','右后方柜体主体部分露出，顶面和正面轮廓可见。'),'west_switchgear_01':('held','框覆盖计划变压器左上主体，后方柜体归属无法从 RGB 确认，需实例掩码。'),'transformer_sw':('visible','左侧被建筑边缘及杆遮挡，但仍有连续主体和顶部附属结构。'),'transformer_mid':('visible','计划变压器主体、顶部附属结构和基座清楚。'),'switchgear_west':('held','极窄横框落在计划变压器正面区域，看不出后方目标的可靠内容；需实例掩码。'),'reactor_north':('visible','右边缘截断，圆柱主体及基座仍可辨。')},
}

def main():
    path=OUT/'review/decisions.json'
    if path.exists():verify_tree(path);print('VERIFIED_EXISTING');return
    verify_tree(OUT/'review/manifest.json');m=read(OUT/'review/manifest.json');frames=[];held=[]
    if len(m['frames'])!=12:raise ValueError('Wrong frame count')
    for frame in m['frames']:
        expected=OBS[frame['category']]
        if set(expected)!={x['object_id'] for x in frame['objects']}:raise ValueError('Explicit object observations incomplete')
        objects=[]
        for obj in frame['objects']:
            status,reason=expected[obj['object_id']]
            objects.append({**obj,'review_status':'content_risk_pending_replay' if status=='held' else 'visible_content_observed','reason':reason,'pixel_instance_certified':False})
            if status=='held':held.append(dict(view_id=frame['view_id'],variant=frame['variant'],object_id=obj['object_id'],reason=reason))
        frames.append({**frame,'objects':objects,'review_status':'held_for_content_risk' if any(x['review_status']=='content_risk_pending_replay' for x in objects) else 'visual_review_only_pending_dedup','review_nature':'AI辅助审核','reviewed_at':datetime.now(timezone.utc).isoformat(),'variant_observation':'暖暗条件亮度降低，目标边界仍能观察；不以该观察代替掩码认证。' if frame['variant']=='steel_warm_dim' else '已单独查看本帧全图叠框，非从同位姿其他变体代填。'})
    from scripts.vision.verify_experiment_baseline import verify
    save(path,dict(status='pilot_held_no_expansion_or_training',frames=frames,held_boxes=held,held_box_count=len(held),held_frames=sum(x['review_status']=='held_for_content_risk' for x in frames),all_frame_count=12,all_box_count=54,baseline=verify(ROOT/'config/perception/visual_experiment_baseline_v1.json'),next_step='Replay held instances with exact RGB alignment using cleanup-fixed segmentation helper; preserve labels and all variants, no remaining 16-frame expansion until pilot gate passes.',inputs={str(p):file_sha256(p) for p in (OUT/'review/manifest.json',Path(__file__))}))
    print('REVIEW_RECORDED',len(held),'HELD_BOXES',sum(x['review_status']=='held_for_content_risk' for x in frames),'HELD_FRAMES',flush=True)

if __name__=='__main__':main()
