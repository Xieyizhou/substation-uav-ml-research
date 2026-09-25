"""Import explicit decisions made after viewing these exact current evidence images."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.brightness_lr_retention import OUT,read,verify,frozen,file_sha256
from scripts.vision.audit_brightness_transfer import validate_review

EVIDENCE='0330d62f53fc1215cf8bd01a2db24036abbe5a90aceb7e40b17a2a2c64dd2fee'
FIT_EVIDENCE='e54e202d0f06b463c3d8a48975a2465e908e1a45a198a7877df4d7b54a5b5c0b'
# Each entry was judged from its current full overlay and individual crop, not a frame theme.
NOTES={
 'FP01:0':('cabinet_like_block','青色矩形主体、深灰前面板、侧面/顶部及黑色底座清楚可见；框覆盖柜状块体，不是杆体。仅 seed 27；外形不等于资产身份。'),
 'FP02:0':('cabinet_like_block','近处青色块体前面板、左侧、顶部和底座构成预测框主体；后方杆和灰块不作为本框对象。仅 seed 27，视角与 FP01 不同。'),
 'FP03:0':('mixed_structure','框跨越前景深色竖杆、后方灰色块体和右下青色块体边缘，属于混合结构；不将整框解释为完整电抗器。seed 7 与 seed 27 对同图近同一区域重复出现。'),
 'FP03:1':('mixed_structure','单独查看本框裁剪：中心粗杆遮住后方灰块，右侧还包含青色块体片段及另一细杆；与 seed 7 对应同一混合结构，非新增独立样本。'),
 'FP04:0':('cabinet_like_block','冷暗条件下左侧青色矩形块体、深色面板和基座；贴近图缘，按可见内容记录，不将框外杆体当成误检主体。seed 27 本图另一个框是不同块体。'),
 'FP04:1':('cabinet_like_block','单独裁剪显示中央灰色方块的暗面板、灰边和薄黑基座，无中心杆体；被预测为电抗器。与同图第一个青色块体框是不同结构。'),
 'FP05:0':('cabinet_like_block','灰色块体的大面积暗前面板、外框及黑色基座构成误检内容；seed 7 将其判为电容器组。三个 seed 同图同一结构重复出现。'),
 'FP05:1':('cabinet_like_block','单独核对 seed 17 裁剪，仍为同一灰块面板及薄基座，框较 seed 7 略收缩，不是第二个独立对象。'),
 'FP05:2':('cabinet_like_block','单独核对 seed 27 裁剪，仍是灰色块体前面板和基座，不包含清晰电容器细结构；与其他两 seed 是同图同结构重复。'),
 'LOSS01:0':('partially_occluded_target','光照图中后方深青色目标的顶部、上部和右侧主体仍可见，下前部被较近青色块体遮挡。seed 7 出现与目标 IoU≈0.602 的 transformer 框，confidence≈0.617，按冻结规则归错类；不认证原帧像素级可见性。'),
 'LOSS02:0':('visible_block_target','原始图中的大面积青灰矩形目标及黑基座可见，左旁竖杆接近边界，主体并非仅基座或极窄片段。seed 27 的 transformer 框 IoU≈0.905、confidence≈0.866，按冻结规则归错类；不凭视觉更改已有实例类别。'),
}


def main():
    ep=OUT/'error-review/evidence.json';e=read(ep);verify(e)
    if e['identity']!=EVIDENCE:raise ValueError('These visual decisions bind only the viewed evidence identity')
    now=datetime.now(timezone.utc).isoformat();decisions=[]
    for r in e['events']:
        for i in range(len(r['predictions']) if r['kind']=='FP' else 1):
            rid=f"{r['event_id']}:{i}";content,reason=NOTES[rid]
            d=dict(review_id=rid,review_nature='AI辅助审核',reviewed_at=now,reason=reason,
                decision='reviewed_current_prediction_not_training_admission',visual_content=content,
                simulation_asset_identity='unknown_not_inferred_from_visual_shape',pixel_visibility_certified=False,
                evidence_sha256=r['evidence_sha256'],image_sha256=r['source']['image_sha256'],crop_sha256=r['crops'][i]['sha256'])
            if r['kind']=='FP':d['prediction']=r['predictions'][i]
            else:d['truth']=r['truth'];d['operational_events']=r['events']
            decisions.append(d)
    if {d['review_id'] for d in decisions}!=set(NOTES):raise ValueError('Unexpected review coverage')
    validate_review(e,decisions)
    path=OUT/'error-review/review.json'
    if path.exists():verify(read(path))
    else:frozen(path,dict(status='explicit_current_visual_review_complete',decisions=decisions,inputs={str(x):file_sha256(x) for x in [ep,Path(__file__)]}))
    fp=OUT/'fit-review/evidence.json';f=read(fp);verify(f)
    if f['identity']!=FIT_EVIDENCE or f['events']:raise ValueError('Clear miss absence changed; requires new review')
    path=OUT/'fit-review/review.json'
    if path.exists():verify(read(path))
    else:frozen(path,dict(status='no_current_clear_misses_no_visual_approvals_created',decisions=[],reviewed_at=now,
        reason='核对冻结清晰目标列表和三个当前预测单元后，未出现待审清晰漏检；空决定不表示训练图自动准入。',
        inputs={str(x):file_sha256(x) for x in [fp,Path(__file__)]}))
    print('EXPLICIT_VISUAL_DECISIONS',len(decisions),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--import-reviewed',action='store_true');a=ap.parse_args()
    if a.import_reviewed:main()
    else:print('READ_ONLY_NO_AUTO_APPROVAL')
