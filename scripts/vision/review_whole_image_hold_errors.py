"""Explicit AI observations for inspected priority evidence; remaining items stay pending."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.locate_whole_image_hold_errors import OUT,RUN,read,verify,frozen,file_sha256,ROOT
from src.ml.artifacts import object_sha256

OBS={
'L001':('partial','后方宽块体下部被前景蓝设备遮挡，上部和侧面仍可见。'),
'L002':('clear','圆柱主体与基座完整可辨，没有明显前景遮挡。'),
'L003':('boundary','近处柜体顶面和面板可辨，下缘被图像边界截断。'),
'L004':('partial','远处柜体位于成排设备后方，只露顶部与窄横带。'),
'L005':('partial','后排小柜体部分被邻柜及浅色附件遮挡，面板仅局部可见。'),
'L006':('partial','大块体及顶部附件可辨，黑杆穿过主体，左侧接触图缘。'),
'L007':('partial','远处目标下部被前方变压器挡住，只见顶面、附件和上部。'),
'L008':('partial','目标前下部被圆柱电抗器遮住，顶面、上身和附件可见。'),
'L009':('boundary','右图缘仅留柜体侧面与面板边缘，内容和轮廓不完整。'),
'L010':('clear','近处圆柱宽主体、顶面及基座清楚，无明显遮挡。'),
'L011':('clear','宽幅块体及底座清楚，正面有阴影，但不是仅基座或窄片段。'),
'L012':('clear','蓝色柜状块体的顶部、侧背面与基座清楚；该视角未见前面板。'),
'L013':('partial','远排设备被前方同形设备遮挡，仅露顶面与横带。'),
'L014':('boundary','近处柜体顶部和侧面可辨，右/下图缘截断。'),
'L015':('clear','柜体宽背面、顶部及基座可辨，未见前面板不等于实例不存在。'),
'L016':('boundary','左图缘与前景同形设备交叠，仅有顶面和窄侧片。'),
'L017':('partial','电抗器上部圆柱在变压器后方，下部被遮挡；浅色附件不能归属给电抗器。'),
'L052':('partial','后方电容器组的宽上身可见，下部被前景设备遮挡。'),
'L054':('partial','电抗器仅露出柜体上方的一段圆柱，底部不可辨。'),
'L056':('partial','后方块体被前景大型变压器与附件遮挡，顶部和上侧面可见。'),
'L059':('clear','近处电抗器顶面、宽圆柱和基座清楚，损失不能只用遮挡或小尺度解释。'),
'L060':('clear','电容器组宽主体与基座清楚，主体有斜阴影，无严重遮挡。'),
}
NEG={
'N01':('block_back_base','灰块体的背面下半部与黑色底座，框不覆盖完整顶部。'),
'N02':('block_panel','灰色块体的面板、侧面及底座。'),
'N03':('block_side_panel','远处灰块体侧背面、窄面板与底座。'),
'N04':('block_side_panel','远处灰块体侧背面、窄面板与底座；多seed框覆盖同一结构。'),
'N05':('mixed_pole_block','框内黑杆与后方灰块体、旁侧蓝块体交叠；不能只判为杆或柜。'),
'N06':('wall_sky_ground','左图缘围墙—天空—地面交界，框中没有独立柜体。'),
'N07':('block_panel','灰色正面面板块体与底座。'),
'N08':('block_panel','灰色正面面板块体与底座；不同seed可能预测成不同类别。'),
'N09':('block_side_panel','灰块体宽侧面和右侧面板，底座可见。'),
'N10':('boundary_block_base','右图缘灰块体与面板/底座局部；不同框的覆盖范围不同。'),
'N11':('mixed_block_pole','灰块体侧背面和底座为主，右侧黑杆进入框内。'),
}

def validate(e,decisions):
    expected=set(e['focus_ids'])|{f"{r['event_id']}:{i}" for r in e['negative'] for i in range(len(r['predictions']))}
    if len(decisions)!=len(expected) or {d['review_id'] for d in decisions}!=expected:raise ValueError('Missing/duplicate review')
    for d in decisions:
        if not d['reason'] or d['review_nature']!='AI辅助审核' or not d['reviewed_at']:raise ValueError('Incomplete explicit review')
        for kind in ('image','evidence'):
            if file_sha256(d[kind+'_path'])!=d[kind+'_sha256']:raise ValueError('Stale evidence')
        if d['pixel_visibility_certified'] is not False:raise ValueError('No mask certification')

def main():
    e=read(OUT/'evidence.json');verify(e);ds=[];now=datetime.now(timezone.utc).isoformat()
    if set(OBS)!=set(e['focus_ids']):raise ValueError('Observation scope mismatch')
    def common(r,rid,reason):
        s=r['source']
        return dict(review_id=rid,event_id=r['event_id'],image_path=s['image_path'],image_sha256=s['image_sha256'],
          evidence_path=r['evidence_path'],evidence_sha256=r['evidence_sha256'],reviewed_at=now,review_nature='AI辅助审核',
          reason=reason,pixel_visibility_certified=False,source_record_identity=object_sha256(r),prior_model_results_known_not_blind=True)
    for r in e['all_losses']:
        if r['event_id'] not in OBS:continue
        content,reason=OBS[r['event_id']];d=common(r,r['event_id'],reason)
        d.update(visual_content=content,truth=r['truth'],model_events=r['events'],decision='visual_observation_not_label_revision')
        ds.append(d)
    for r in e['negative']:
        for i,p in enumerate(r['predictions']):
            content,reason=NEG[r['event_id']]
            if r['event_id']=='N07' and i==1:content,reason='blue_block_panel','左图缘蓝色小块体及其面板和底座，不是灰色后方块体。'
            if r['event_id']=='N10' and i==2:content,reason='boundary_block_base','低位框主要覆盖灰块体下部、黑底座及邻近地面，不是整个块体。'
            d=common(r,f"{r['event_id']}:{i}",reason);d.update(visual_structure=content,prediction=p,
                simulation_asset_identity='unknown_not_inferred_from_visual_shape',decision='false_prediction_content_localized')
            ds.append(d)
    validate(e,ds)
    pending=[dict(event_id=r['event_id'],gap='Outside this priority visual pass; full inference identity retained, not reviewed') for r in e['all_losses'] if r['event_id'] not in OBS]
    paths=[OUT/'evidence.json',Path(__file__)]
    frozen(OUT/'review.json',dict(status='priority_review_complete_other_losses_pending',decisions=ds,pending=pending,inputs={str(x):file_sha256(x) for x in paths}))
    losses=e['all_losses'];focus=[r for r in losses if r['event_id'] in OBS]
    summary=dict(all_unique_loss_entries=len(losses),all_seed_loss_events=sum(len(r['events']) for r in losses),
      focused_entries=len(focus),focused_seed_loss_events=sum(len(r['events']) for r in focus),
      focused_reasons=dict(Counter(v['reason'] for r in focus for v in r['events'])),
      focused_visual=dict(Counter(OBS[r['event_id']][0] for r in focus)),negative_images=len(e['negative']),
      reviewed_prediction_boxes=sum(len(r['predictions']) for r in e['negative']),pending_entries=len(pending))
    print(summary)
    frozen(OUT/'summary.json',dict(status='localized_with_named_gaps',summary=summary,inputs={str(OUT/'review.json'):file_sha256(OUT/'review.json')}))

if __name__=='__main__':main()
