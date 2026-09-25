"""Explicit observations after viewing all current FP/loss and clear-fit panels."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.prepare_brightness_transfer import OUT,read,verify,frozen,file_sha256
from scripts.vision.audit_brightness_transfer import validate_review

OBS={
'FP01':[('cabinet_like_block','右图缘灰色块体的大矩形面板、侧边与基座，框中无独立杆体。')],
'FP02':[('cabinet_like_block','灰色块体的宽侧面、面板及完整基座，结构独立且较大。')],
'FP03':[('ground_shadow_mixed_edge','框大部为地面格线和杆影，左缘叠入青色块体及基座，顶部有细杆下段；不能按整帧主题记成柜体。')],
'FP04':[('cabinet_like_block','远处灰色块体宽侧面和右侧面板、基座，预测没有覆盖前景高杆。')],
'FP05':[('mixed_structure','灰色块体与前景深色竖杆、右侧青色小块体相互重叠。'),('mixed_structure','另一 seed 的相近框也覆盖灰块、竖杆与青色块体，独立检查局部后确认同一混合结构。')],
'FP06':[('cabinet_like_block','左侧青色块体的深色矩形面板和基座，非杆体。'),('cabinet_like_block','后方灰色块体正面面板和底边；与同图另一预测不是同一个结构。')],
'FP07':[('boundary_block_fragment','右图缘截断的灰色面板边缘与基座，框包含相邻地面。')],
'FP08':[('ground_shadow','图底部主要是灰色阴影和地面格线，紧靠块体基座下方；没有完整设备主体。')],
'LOSS01':[('partially_occluded_block','后方矩形主体大侧面和顶面可见，左下被青色设备遮挡。')],
'LOSS02':[('foreground_occluded','只露出圆柱上段，下部被前景设备遮住；不能宣称完整主体可见。')],
'LOSS03':[('foreground_occluded','后方矩形目标上段可见，下部被前景设备顶面、接线柱和青色结构遮挡。')],
'LOSS04':[('boundary_truncated','右图缘截断圆柱，可见连续曲面及部分基座；不是仅基座。')],
'LOSS05':[('foreground_occluded','远处矩形主体只露上段，框内混有前景设备顶面和柱附件。')],
'LOSS06':[('partially_occluded_block','矩形主体宽侧面、顶面和底边大部可见，左下角被青色前景遮挡。')],
}
FITOBS={
'27:C04-L3':'远处独立矩形块体的顶面、侧面和基座可辨，无遮挡但外形区分类别的特征少；不据预测改来源类别。',
'27:C10-L4':'独立矩形块体的两侧面、顶面与底座完整，目标没有被前景覆盖。',
'27:C17-L2':'中央独立矩形块体顶面、两侧面与基座清晰，前方杆和大型设备未遮住主体。',
'27:C31-L0':'完整圆柱曲面轮廓及基座清楚，非图缘截断或仅基座，原始图本身不是被增强后无法辨认的图。',
}

def main():
    path=OUT/'error-review/evidence.json';e=read(path);verify(e);now=datetime.now(timezone.utc).isoformat();ds=[]
    if set(OBS)!={r['event_id'] for r in e['events']}:raise ValueError('Observation coverage mismatch')
    for r in e['events']:
        notes=OBS[r['event_id']];n=len(r['predictions']) if r['kind']=='FP' else 1
        if len(notes)!=n:raise ValueError('Per-box coverage mismatch')
        for i,(content,reason) in enumerate(notes):
            d=dict(review_id=f"{r['event_id']}:{i}",evidence_sha256=r['evidence_sha256'],image_sha256=r['source']['image_sha256'],crop_sha256=r['crops'][i]['sha256'],visual_content=content,reason=reason,review_nature='AI辅助审核',reviewed_at=now,pixel_visibility_certified=False,training_approved=False)
            if r['kind']=='FP':d.update(prediction=r['predictions'][i],asset_identity='unknown_not_inferred_from_shape')
            else:d.update(truth=r['truth'],diagnostic_events=r['events'])
            ds.append(d)
    validate_review(e,ds)
    inputs={str(x):file_sha256(x) for x in [path,Path(__file__),Path('scripts/vision/audit_brightness_transfer.py').resolve()]}
    frozen(OUT/'error-review/review.json',dict(status='explicit_current_error_review_complete',decisions=ds,repeat_relations=[dict(event_id='FP05',relation='Two seeds on same image and overlapping mixed structure, not independent samples'),dict(event_id='FP06',relation='Two seeds on same image but different physical blocks')],inputs=inputs))
    path=OUT/'fit-review/evidence.json';e=read(path);verify(e)
    if set(FITOBS)!={r['review_id'] for r in e['events']}:raise ValueError('Fit observation coverage mismatch')
    ds=[dict(review_id=r['review_id'],evidence_sha256=r['evidence_sha256'],image_sha256=r['source']['image_sha256'],truth=r['truth'],reason=FITOBS[r['review_id']],review_nature='AI辅助审核',reviewed_at=now,pixel_visibility_certified=False,training_approved=False) for r in e['events']]
    frozen(OUT/'fit-review/review.json',dict(status='explicit_clear_fit_error_review_complete',decisions=ds,inputs={str(x):file_sha256(x) for x in [path,Path(__file__)]}))
    print('EXPLICIT_DEVELOPMENT_DECISIONS',sum(len(x) for x in OBS.values()),'CLEAR_FIT_DECISIONS',len(ds))
if __name__=='__main__':main()
