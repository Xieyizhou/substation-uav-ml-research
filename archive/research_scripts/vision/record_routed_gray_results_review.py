"""Explicit observations from this arm's full images and per-target crops."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.routed_gray_transfer_control import OUT as TRAIN, checked
from scripts.vision.record_contrast_cpu4_review import validate_positive
from scripts.vision.audit_reviewed_order_results import validate_review
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

OUT = TRAIN / 'audit-v1'
NOTES = {
0:{1:'青灰块体大正面及完整基座清楚，未见主要遮挡，斜线是落在主体上的阴影。'},
1:{1:'后变压器三柱、顶面与主体上部可见，下部被近变压器遮挡；框内近柱不是后目标部件。',2:'近变压器三柱、顶面及两个主体面清楚，下图缘截断。',3:'后电容器顶面与主体上部可见，下部被青色块体和近变压器遮挡；前景柱不属于该实例。'},
2:{2:'左缘柜体大侧背面、顶面及基座可见，左图缘截断，无正面面板。',3:'后圆柱顶面及主体上部可见，下部被变压器遮挡，框内小柱属于前景。'},
3:{7:'右图缘圆柱顶面、主体和基座部分可见，右缘截断。'},
4:{1:'独立圆柱主体及完整基座清楚，无主要前景遮挡。'},
5:{1:'中柜顶面及面板上缘细线可见，下部被近柜遮挡。',2:'近柜大顶面、主体与面板上部清楚，下图缘截断。'},
6:{3:'近柜面板、侧面、顶面与基座清楚，未见主要遮挡。'},
7:{0:'后柜顶部窄带可见，下部被同类队列遮挡，框内含前景柜体。',2:'近柜大顶面、主体与面板上部清楚，下图缘截断。'},
8:{2:'队列后柜顶面与侧面窄带可见，下部被近柜遮挡。',8:'远柜右侧主体、顶面及基座可见，左部被变压器遮挡，尺度较小。'},
9:{1:'青灰块体正面及完整基座清楚，主体上有斜向阴影，无主要前景遮挡。'},
10:{3:'后圆柱顶面及上部可见，下部被变压器遮挡，框内小柱属于前景变压器。'},
11:{1:'后变压器三柱、顶面和主体上部可见，下部被近变压器遮挡。'},
12:{0:'近左柜体大顶面与侧面可见，左下图缘截断。',1:'独立圆柱主体及完整基座清楚，未见前景遮挡。'},
13:{2:'队列后柜顶面与窄带可见，下部遮挡。',9:'后电容器顶面及主体上部可见，下部被变压器遮挡；框内三柱属于前景。'},
14:{7:'右缘圆柱顶面、主体与基座部分可见，右图缘截断。'},
15:{1:'后电容器顶面、大主体上部及右侧基座可见，下部被青色近块体遮挡。'},
16:{1:'右缘柜体侧面与基座局部可见，右图缘截断且被前柜遮挡。',8:'近圆柱顶面、主体与完整基座清楚，无遮挡。',10:'远电容器顶面、主体上部可见，粗杆及前变压器遮挡；不能把前景部件归给它。'},
17:{5:'后变压器三柱、顶面、主体上部及右基座局部可见，下部被近变压器遮挡。',8:'近圆柱顶面、主体与完整基座清楚，无主要遮挡。'},
}
MATERIAL = {
0:{1:'后灰柜顶面、大侧面与面板部分可见，左前被近柜遮挡。',2:'灰柜顶面、侧面、面板轮廓及基座清楚。',5:'后灰柜顶面与面板上部可见，下部被变压器遮挡。'},
1:{4:'灰柜面板、顶面、侧面及右基座清楚，左下局部被近柜遮挡。'},
2:{0:'右灰变压器三柱、顶面、大主体及基座清楚，右图缘截断。'},
3:{1:'灰圆柱主体与完整基座清楚，未见主要前景遮挡；不是仅基座片段。'},
4:{4:'左灰变压器顶柱、顶面、主体与基座可见，粗杆遮挡且左图缘截断。',6:'中灰柜顶面、侧面上部及基座部分可见，下部被近变压器遮挡。'},
5:{3:'右侧队列最远柜体顶部及窄带可见，前柜遮挡并受画面边缘限制。',5:'远灰变压器三柱、顶面、大主体及基座可见，前景杆体遮挡右部。'},
6:{0:'灰变压器三柱、顶面、主体及基座可见，粗杆遮挡中部。',2:'后灰电容器顶面、大侧面与右基座可见，左下被青色块体遮挡。'},
}
FP = {
0:('pole_body','框覆盖竖杆下段及杆脚、地面阴影，不包括顶端横件。'),
1:('gray_block_body','灰块体大侧背面、右侧暗面板及基座，右框缘靠近杆体，未包含大部分青色前景。'),
2:('mixed_structure','冷暗灰块体大背面与基座、前景粗杆及右下青色块体共同入框。'),
3:('gray_block_body','灰块体大侧背面及基座，右缘含少量青色前景；竖杆主体在预测框外。'),
4:('mixed_structure','冷暗灰块体大侧背面和基座，右侧面板叠有杆体，右下包含青色前景局部。'),
5:('gray_block_body','灰块体正面大暗面板、灰边框及基座。'),
6:('gray_block_body','冷暗灰块体正面大暗面板、灰边框及基座。'),
7:('teal_block_body','青色块体暗前面板、顶面、侧面与基座清楚，无主要遮挡。'),
8:('mixed_structure','灰块体大背侧面、前景粗杆及右下青色块体共同入框。'),
9:('mixed_structure','冷暗灰块体大背面与基座、前景粗杆及右下青色块体共同入框。'),
10:('teal_block_body','青色块体大侧面、顶面、右暗面板及基座，不是后方灰块体。'),
11:('gray_block_body','灰块体正面大暗面板、灰边框及基座，与本图其他seed预测指向同一结构。'),
12:('gray_block_body','右图缘严重截断的灰块体面板、边框及基座局部；左侧含围墙地面边界，不是中间青色块体。'),
13:('pole_body','竖杆中下段、杆脚及地面，杆顶横件不在框内，不是右侧灰块体。'),
14:('mixed_structure','冷暗灰块体大背侧面、面板及基座，右侧包含杆体和青色前景局部。'),
15:('mixed_structure','灰块体大背侧面、前景粗杆及右下青色块体共同入框。'),
16:('mixed_structure','冷暗灰块体大背面与基座、前景粗杆及右下青色块体共同入框。'),
17:('gray_block_body','灰块体正面暗面板、灰边框与基座，与同seed另一类别框覆盖同一结构，不是新增独立对象。'),
18:('gray_block_body','灰块体正面暗面板、灰边框与基座，与同seed另一类别框覆盖同一结构，不是新增独立对象。'),
}

def positive(e, notes, stamp):
    result=[]
    for f in e['frames']:
        for x in f['events']:
            result.append(dict(frame_id=f['frame_id'],seed=x['seed'],truth_index=x['truth_index'],truth=x['truth'],diagnosis=x['diagnosis'],loss_against=x['loss_against'],
                reason=notes[int(f['frame_id'].split('-')[-1])][x['truth_index']],review_nature='AI辅助审核',reviewed_at=stamp,
                image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],pixel_visibility_certified=False,decision='content_observed_not_label_admission'))
    validate_positive(e,result)
    return result

def run():
    ep=OUT/'evidence.json';mp=OUT/'material-evidence.json';np=TRAIN/'evaluation-v1/error-review-v1/evidence.json'
    e,m,n=map(checked,(ep,mp,np));stamp=datetime.now(timezone.utc).isoformat()
    positives=positive(e,NOTES,stamp);materials=positive(m,MATERIAL,stamp);negatives=[]
    for f in n['frames']:
        for x in f['events']:
            category,reason=FP[int(x['event_id'].split('-')[-1])]
            negatives.append(dict(event_id=x['event_id'],seed=x['seed'],prediction=x['prediction'],content_category=category,reason=reason,
                review_nature='AI辅助审核',reviewed_at=stamp,image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],asset_identity='unknown_visual_shape_not_asset_identity'))
    validate_review(n,negatives)
    deps={str(p.resolve()):file_sha256(p) for p in (ep,mp,np,OUT/'evidence-build-receipt.json',Path(__file__))}
    result=dict(status='current_arm_error_content_reviewed_not_candidate_passed',positive_decisions=positives,material_decisions=materials,negative_decisions=negatives,
        counts=dict(original_lighting_events=len(positives),material_events=len(materials),negative_events=len(negatives),negative_unique_frames=len(n['frames'])),
        limits='Visual content only; no instance-mask certification, no asset identity inferred from shape. Gains/persistent states retained in numerical comparisons. No label edits.',
        training_admitted=False,promotable=False,inputs=deps)
    return write_record(OUT/'review.json',result)

if __name__=='__main__':print(run()['counts'])
