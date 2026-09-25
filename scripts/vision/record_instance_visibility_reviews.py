"""Record reviewed overlays plus inherited explicit pending observations, never auto-pass."""
from datetime import datetime,timezone
from scripts.vision.instance_visibility_diagnosis import OUT,ROOT,FOLLOWUP,read,save,file_sha256,prepare,Path
from scripts.vision.finalize_instance_visibility import validate_reviews

OBSERVATIONS={
'T049':('visible_content_insufficient','已查看紫色实例叠图：画面右边缘只有小片基部及极窄结构，未构成可辨识的完整柜体。连续稳定窗口与原图逐像素一致；不按空框删除。','replay/T049/attempt-03/receipt.json','replay/T049/attempt-03/first-stable-window/1-T049-crop.png'),
'T054':('visible_content_insufficient','已查看紫色实例叠图：画面左边缘仅有小片基座，主体在画外。连续稳定窗口与原图逐像素一致；不自动删除标签。','replay/T054/attempt-02/receipt.json','replay/T054/attempt-02/first-stable-window/1-T054-crop.png'),
'T077':('evidence_insufficient','已查看辅助叠图：目标为后方窄带，右侧暗色片不能直接作为目标面板证据。原始采集程序异常退出，离线复核不能替代完整运行回执；三次尝试已耗尽。','postfailure-check/T077/inspection.json','postfailure-check/T077/first-stable-window/1-T077-crop.png'),
'T095':('evidence_insufficient','已查看辅助叠图：目标仅露出前景顶面和立柱后方的窄带。运行退出报 mutex lock failed，离线像素吻合不能替代完整运行回执；三次尝试已耗尽。','postfailure-check/T095/inspection.json','postfailure-check/T095/first-stable-window/1-T095-crop.png'),
}

def main():
    p=prepare();previous=read(FOLLOWUP/'review.json');prior={r['review_id']:r for r in previous['decisions']}
    manifest=read(FOLLOWUP/'manifest.json');lookup={r['review_id']:r for r in manifest['items']};decisions=[];frames=[]
    for frame in p['frames']:
        frame_status='expansion_blocked_pilot_incomplete';reason='Pilot T077/T095 exhausted attempt budget with abnormal capture exit; no new replay attempted.'
        for rid in frame['review_ids']:
            row=lookup[rid];d=dict(review_id=rid,status='evidence_insufficient',original_frame_certified=False,visible_pixel_count=None,visible_bbox_xyxy=None,
                component_evidence='unknown',review_nature='AI-assisted',reviewed_at=datetime.now(timezone.utc).isoformat(),
                reason=prior[rid]['reason']+' 本轮已核对来源；四帧试验未全部通过，未展开本帧重放，继续待定。',
                inputs={str(x):file_sha256(x) for x in (Path(row['image_path']),Path(row['label_path']),Path(row['followup_evidence_path']),FOLLOWUP/'review.json',OUT/'protocol.json')})
            if rid in OBSERVATIONS:
                status,reason_text,receipt,evidence=OBSERVATIONS[rid];result=read(OUT/receipt)
                target=next(t for t in result['records'][0]['targets'] if t['review_id']==rid)
                d.update(status=status,reason=reason_text,original_frame_certified=result['status']=='original_pixel_evidence_certified',
                         visible_pixel_count=target['visible_pixel_count'] if status!='evidence_insufficient' else None,
                         visible_bbox_xyxy=target['visible_bbox_xyxy'] if status!='evidence_insufficient' else None,
                         replay_supplemental_pixel_count=target['visible_pixel_count'])
                d['inputs'].update({str(OUT/x):file_sha256(OUT/x) for x in (receipt,evidence)})
                frame_status='pixel_aligned_reviewed' if d['original_frame_certified'] else 'runtime_failed_budget_exhausted';reason=reason_text
            decisions.append(d)
        frames.append(dict(member_id=frame['member_id'],review_ids=frame['review_ids'],source_status=frame['status'],status=frame_status,reason=reason))
    validate_reviews(p,decisions)
    inputs={str(x):file_sha256(x) for x in (Path(__file__),OUT/'protocol.json',FOLLOWUP/'review.json')}
    for d in decisions:inputs.update(d['inputs'])
    save(OUT/'review.json',dict(status='explicit_AI_assisted_decisions_with_holds',decisions=decisions,inputs=inputs))
    save(OUT/'frame-dispositions.json',dict(frames=frames,inputs={str(OUT/'review.json'):file_sha256(OUT/'review.json')}))

if __name__=='__main__':main()
