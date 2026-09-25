"""Explicit AI decisions after viewing four RGBs, overlays and target crops."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from scripts.vision.replay_redistribution_occlusion import OUT, PRIOR, OCCLUDED, read, verify, frozen, file_sha256, ROOT
from scripts.vision.prepare_whole_image_hold import baseline_verify

OBS = {
 'P31': '可见后方块体的顶面及上部侧面，下部被前景变压器遮挡；浅色柱状附件不属于目标掩码。实例归属已确认，但未见足以辨识电容器组类别的面板或结构，内容判为不足。',
 'P44': '两处前景块体之间仅露出窄小侧面及极薄顶边；目标与同色前景视觉粘连，实例掩码分离后仍缺少可辨识内容。',
 'P47': '前景变压器后方只露出薄顶面及窄上部侧面，浅色柱状附件被目标掩码排除；类别辨识内容不足。',
 'P53': '后方目标可见较宽顶面及上部两个平面，下部被前景变压器遮挡；较大像素数不等于类别可辨，未见足够区别于普通块体的结构。',
}

def validate_decisions(ds):
    if len(ds)!=4 or {d['event_id'] for d in ds}!=OCCLUDED: raise ValueError('Missing or duplicate review')
    for d in ds:
        if not d['reason'] or d['review_nature']!='AI辅助审核': raise ValueError('Missing explicit review')
        if d['training_approved'] is not False: raise ValueError('Visibility is not training admission')
        if d['replay_status']!='original_pixel_evidence_certified': raise ValueError('Unaligned original evidence')
        if d['visible_pixel_count']<=0: raise ValueError('Empty mask cannot establish visible content')

def main():
    p=read(OUT/'protocol.json');verify(p)
    ds=[]; paths=[OUT/'protocol.json', PRIOR/'review.json', Path(__file__)]
    for f in p['frames']:
        eid=f['event_id']; rp=OUT/'replay'/eid/'attempt-01/receipt.json'; r=read(rp);verify(r)
        if r['status']!='original_pixel_evidence_certified' or not r['process_cleanup_complete'] or r['stable_frames']!=3: raise ValueError('Replay not certified')
        if any(not x['rgb_exact'] or x['maximum_box_delta_px']>1 or x['skew_ms']>33.334+1e-6 for x in r['records']): raise ValueError('Alignment failed')
        targets=[x['targets'][0] for x in r['records']]
        if targets.count(targets[0])!=3: raise ValueError('Unstable target')
        t=targets[0]; folder=rp.parent/'first-stable-window'
        evidence=[folder/'frame-1-rgb.png',folder/f'1-{eid}-overlay.png',folder/f'1-{eid}-crop.png',folder/'frame-1-difference.png']
        paths.extend([rp,*evidence])
        ds.append(dict(event_id=eid, object_id=f['events'][0]['object_id'], runtime_label=t['runtime_label'],
            source_image=f['source_image'], source_image_sha256=file_sha256(f['source_image']),
            replay_receipt=str(rp), replay_identity=r['identity'], replay_status=r['status'],
            visible_pixel_count=t['visible_pixel_count'], visible_bbox_xyxy=t['visible_bbox_xyxy'],
            status='实例有可见证据，但内容不足', reason=OBS[eid], component_identity='unknown',
            pixel_visibility_certified=True, review_nature='AI辅助审核', reviewed_at=datetime.now(timezone.utc).isoformat(),
            full_RGB_overlay_and_crop_viewed=True, training_approved=False,
            evidence={str(x):file_sha256(x) for x in evidence}))
    validate_decisions(ds)
    frozen(OUT/'review.json',dict(status='four_attributions_resolved_content_risk_remains', decisions=ds,
        no_area_admission_threshold=True, inputs={str(x):file_sha256(x) for x in paths}))
    tests=['tests.test_redistribution_occlusion_review','tests.test_redistribution_target_attribution','tests.test_permission_risk_pilot','tests.test_instance_visibility_diagnosis']
    result=subprocess.run([sys.executable,'-m','unittest',*tests],cwd=ROOT,capture_output=True,text=True,timeout=60)
    if result.returncode: raise ValueError(result.stderr)
    baseline=baseline_verify(ROOT/'config/perception/visual_experiment_baseline_v1.json')
    if not baseline['integrity_passed'] or baseline['pinned_files_verified']!=40: raise ValueError('Baseline failure')
    paths=[OUT/'review.json',Path(__file__),ROOT/'docs/results/ml_redistribution_occlusion_replay_20260909.md',ROOT/'config/perception/visual_experiment_baseline_v1.json']
    paths.extend(ROOT/(t.replace('.','/')+'.py') for t in tests)
    frozen(OUT/'completion.json',dict(status='four_same_frame_attributions_confirmed_quality_gate_blocked',
        original_frames_certified=4, new_training_images=0, labels_modified=False, training_started=False,
        readiness_issued=False, baseline=baseline, regression_output=result.stderr, whole_repository_tested=False,
        test_runner_note='pytest unavailable; unittest is the repository-compatible runner and passed.',
        next_direction='Freeze a separate no-increase constraint for twelve risk members and recheck count feasibility; not yet applied.',
        inputs={str(x):file_sha256(x) for x in paths}))
    print('FOUR_ALIGNED_VISIBLE; CONTENT_RISK_REMAINS; PINNED40_PASSED; NO_TRAINING')

if __name__=='__main__':main()
