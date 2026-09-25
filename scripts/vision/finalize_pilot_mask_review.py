"""Explicit AI visual decisions after viewing all 19 certified mask crops."""
from datetime import datetime, timezone
from pathlib import Path
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import read_record, write_record
from scripts.vision.record_pilot_full_label_audit import BASE
from scripts.vision.replay_pilot_audit_remaining import checked

REASONS = {
 'frame-00':['仅露出柜体上部窄带，主体被前景遮挡','仅露出上沿窄带，框内大部分是前景柜体'],
 'frame-02':['仅露出上部横带','只见上沿及右侧极窄L形片段','左图缘仅极窄片段','仅小矩形上沿片段','只见电抗器右侧的一段侧面及底部，主体被遮挡'],
 'frame-03':['长而窄的上沿，不见主体','仅露出小上角片段'],
 'frame-05':['仅有上部窄带，主体被遮挡'],
 'frame-06':['仅有上部窄带','仅有上沿窄带','变压器与前景之间只见窄L形片段','仅有极小竖片'],
 'frame-08':['左缘只露出一段上沿，其余遮挡'],
 'frame-10':['仅在杆体右侧地面附近露出极小片段','右图缘只见截断竖片及基座局部','杆体后只见一小段侧面，缺少完整可辨结构'],
 'frame-11':['电容器组后只露出极小上角，变压器主体不可辨'],
}

def run():
    root=BASE/'pilot-full-label-replay-v1';summary=root/'technical-summary.json'
    s=read_record(summary); prior=BASE/'pilot-full-label-audit-v1/full-label-review.json';r=read_record(prior)
    protocol=read_record(root/'protocol.json'); frames={f['review_ids'][0]:f for f in protocol['frames']}
    deps={str(p.resolve()):file_sha256(p) for p in (summary,prior,root/'protocol.json',Path(__file__))};decisions=[]
    for f in s['frames']:
        rec=checked(Path(f['receipt']));deps[f['receipt']]=file_sha256(f['receipt'])
        if rec['status']!='original_pixel_evidence_certified':raise ValueError('Uncertified replay')
        if len(REASONS[f['frame']])!=len(f['targets']):raise ValueError('Missing visual decision')
        for t,reason in zip(f['targets'],REASONS[f['frame']]):
            image=Path(f['receipt']).parent/'first-stable-window'/f'1-{t["review_id"]}-crop.png'
            deps[str(image)]=file_sha256(image)
            decisions.append(dict(view_id=frames[f['frame']]['member_id'],review_id=t['review_id'],runtime_label=t['runtime_label'],
                state='instance_visible_content_insufficient',reason=reason,visible_pixels=t['visible_pixel_count'],
                visible_bbox=t['visible_bbox_xyxy'],pixel_visibility_certified=True,components='unknown'))
    held=sorted({d['view_id'] for d in decisions});reviewed=sorted({b['view_id'] for b in r['boxes']});clear=sorted(set(reviewed)-set(held))
    if len(decisions)!=19 or len(held)!=8 or len(clear)!=4:raise ValueError('Unexpected review population')
    for vid in clear:
        if any(b['state']!='visible_content' for b in r['boxes'] if b['view_id']==vid):raise ValueError('Unresolved clear-frame label')
    return write_record(root/'mask-review.json',dict(status='complete_with_eight_whole_frame_holds',review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),
        decisions=decisions,held_view_ids=held,clear_view_ids=clear,clear_frame_scope='Four frames have identifiable content for every recorded label in full-frame visual review, not certification of all pool supervision. Existing simulation mapping supplies class identity.',
        policy='Hold entire affected frames in new experiments; retain historical labels. No numerical area threshold. Presence alone is not approval.',
        inputs=deps,training_admitted=False,promotable=False))

if __name__=='__main__':print(run()['status'])
