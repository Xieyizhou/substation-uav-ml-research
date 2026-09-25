"""Explicit eight-image observations, not a generic automatic approval tool."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.build_full_image_pose_review import OUT, read, save, file_sha256, verify_tree

OBS = {
 'F01': {'transformer_se':'前景蓝柜遮挡下部，但宽主体上部、右侧及顶部附属结构连续可见。', 'capacitor_east':'主体、顶面和基座可见，前景杆遮挡右边缘；不是仅基座或窄片段。'},
 'F02': {'capacitor_east':'宽主体和完整基座清楚，主体未被前景杆遮挡。'},
 'F03': {'entry_switchgear':'完整箱体、顶面、侧面与基座可见；背侧视角，前面板不可见，不能计作面板覆盖。'},
 'F04': {'reactor_north':'圆柱主体、顶面与基座清晰，无明显前景遮挡。'},
 'F05': {'reactor_north':'较低视角下圆柱连续主体和基座清晰，无明显遮挡。'},
 'F06': {'transformer_se':'宽主体、顶部三处附属结构和基座完整可见。', 'capacitor_east':'宽主体、顶面和基座可见，左下角被前景普通柜遮挡。'},
 'F07': {'transformer_se':'主体、顶部附属结构和基座清楚。', 'capacitor_east':'右下部被变压器遮挡，但左侧宽主体、顶面与部分基座仍可见。'},
 'F08': {'entry_switchgear':'完整箱体、顶面、侧面与基座可见；前面板不可见，与 F03 是同源近邻视角。'},
}

def validate_observations(frames, observations):
    ids=[f['frame_id'] for f in frames]
    if len(set(ids))!=len(ids) or set(ids)!=set(observations):
        raise ValueError('Missing or duplicate frame review')
    for f in frames:
        ids=[o['object_id'] for o in f['objects']]
        if len(set(ids))!=len(ids) or set(ids)!=set(observations[f['frame_id']]):
            raise ValueError('Missing or duplicate object review')
        if not all(observations[f['frame_id']].values()):raise ValueError('Empty reason')

def main():
    path=OUT/'review/decisions.json'
    if path.exists():verify_tree(path);print('VERIFIED_EXISTING');return
    manifest=OUT/'review/manifest.json';verify_tree(manifest);m=read(manifest)
    validate_observations(m['frames'],OBS)
    frames=[]
    for f in m['frames']:
        frames.append({**f,'review_status':'visual_review_only_pending_dedup',
          'review_nature':'AI辅助审核','reviewed_at':datetime.now(timezone.utc).isoformat(),
          'objects':[{**o,'review_status':'visible_content_observed','reason':OBS[f['frame_id']][o['object_id']],
            'pixel_instance_certified':False,'component_mask_status':'unknown'} for o in f['objects']]})
    save(path,dict(status='visual_review_complete_pending_intake_checks',frames=frames,
      frame_count=len(frames),box_count=sum(len(f['objects']) for f in frames),
      limitations=['RGB visual observations plus saved mapping, not instance-mask certification.',
       'Same map and assets; F03/F08 linked, no independent-scene claim.',
       'Both switchgear views lack visible front panels.'],
      inputs={str(p):file_sha256(p) for p in (manifest,Path(__file__))}))
    print('EXPLICIT_REVIEW_RECORDED',len(frames))

if __name__=='__main__':main()
