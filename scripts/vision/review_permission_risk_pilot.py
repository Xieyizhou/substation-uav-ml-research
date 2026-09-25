"""Explicit AI review supplement for the two aligned pilot frames."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.resume_supervision_risk_pilot import OUT,PRIOR,ROOT,read,verify,frozen,file_sha256
from scripts.vision.supervision_risk_proposal import validate
from src.ml.artifacts import object_sha256

OBS={
 'T30':'全图及实例叠图确认目标仅在左图缘留下极小色片，局部掩码可定位但无法辨识电抗器类别内容；前景蓝设备占完整框大部。建议未来整图暂缓，不删框留图，不按掩码缩写 full_2d 框。',
 'T08':'全图及局部叠图确认后方目标上部窄带，前景设备与浅色附件遮挡明显。实例有像素证据，但完整可辨识主体不足；建议未来整图暂缓，不能用训练命中证明监督质量。',
}

def validate_alignment(r,eid):
    if r['status']!='original_pixel_evidence_certified' or not r['process_cleanup_complete'] or r['stable_frames']!=3:raise ValueError('Incomplete aligned replay')
    if len(r['records'])!=3:raise ValueError('Missing stable frames')
    targets=[]
    for frame in r['records']:
        if not frame['rgb_exact'] or frame['maximum_box_delta_px']>1 or frame['skew_ms']>33.334+1e-6:raise ValueError('Alignment gate failed')
        ts=[t for t in frame['targets'] if t['review_id']==eid]
        if len(ts)!=1:raise ValueError('Target not unique')
        targets.append(ts[0])
    if targets[1:]!=targets[:-1] or targets[0]['visible_pixel_count']<=0:raise ValueError('Unstable or empty target evidence')
    return targets[0]

def review():
    p=read(OUT/'protocol.json');verify(p);verify(read(PRIOR/'protocol.json'))
    inputs={str(x):file_sha256(x) for x in (OUT/'protocol.json',PRIOR/'protocol.json',Path(__file__))};decisions=[]
    for f in p['frames']:
        eid=f['event_id'];rp=OUT/'replay'/eid/'attempt-01/receipt.json';r=read(rp);verify(r);target=validate_alignment(r,eid)
        inputs[str(rp)]=file_sha256(rp)
        crop=rp.parent/'first-stable-window'/f'1-{eid}-crop.png';overlay=rp.parent/'first-stable-window'/f'1-{eid}-overlay.png'
        for path in (crop,overlay):inputs[str(path)]=file_sha256(path)
        decisions.append(dict(event_id=eid,source_identity=object_sha256(f),evidence_sha256=f['evidence_sha256'],
            proposal='hold_whole_image_proposed',review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),reason=OBS[eid],
            original_pixel_certified=True,visible_pixel_count=target['visible_pixel_count'],visible_bbox_xyxy=target['visible_bbox_xyxy'],
            component_identity='unknown',apply_label_change=False,delete_box_keep_image=False,
            historical_truncation_status=f['historical_truncation_status'],observed_image_boundary_contact='left',
            truncation_evidence_status='visible_region_contacts_boundary_not_full_geometry_certificate',
            replay_receipt=str(rp),crop_path=str(crop),crop_sha256=file_sha256(crop),overlay_path=str(overlay),overlay_sha256=file_sha256(overlay)))
    validate(p,decisions)
    frozen(OUT/'review.json',dict(status='aligned_visible_content_insufficient_proposed_whole_image_hold',decisions=decisions,
        area_admission_threshold_used=False,prior_models_already_viewed_not_blind=True,inputs=inputs))
    print('TWO_EXPLICIT_REVIEW_SUPPLEMENTS_RECORDED')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--review',action='store_true');args=ap.parse_args()
    if args.review:review()
    else:print('PREFLIGHT_ONLY_NO_TRAINING')
