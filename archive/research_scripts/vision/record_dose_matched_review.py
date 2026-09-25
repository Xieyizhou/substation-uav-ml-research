"""Current dose-arm observations after viewing all 25 full/crop pages."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.inspect_dose_matched_results import OUT,TRAIN,checked
from scripts.vision.record_contrast_cpu4_review import validate_positive
from scripts.vision.audit_reviewed_order_results import validate_review
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

NOTES={
0:{2:'左图缘柜体大侧面、顶面和基座可见，左边截断，无正面面板。',3:'圆柱顶面和上部主体可见，下部被变压器遮挡；框内小柱属于前景。'},
1:{4:'左侧变压器主体、顶面及两柱可见，左图缘截断且粗杆遮挡主体。',7:'右缘圆柱顶面、主体和基座部分可见，右图缘截断。'},
2:{0:'独立柜体完整顶面、两个大侧面及基座清楚，无遮挡，此视角无面板。'},
3:{1:'圆柱主体和基座清楚，无遮挡，不是仅基座或窄片段。'},
4:{0:'后柜仅顶面和上部窄带可见，前方同色柜遮挡。',1:'中柜顶面和面板上缘细线可见，下部被近柜遮挡。',4:'右柜顶面、大面板和基座清楚，左下小部分被近柜遮挡。'},
5:{4:'近变压器大顶面、主体与两柱可见，右下图缘严重截断。',5:'后柜顶面和面板上部可见，下部被近变压器遮挡。'},
6:{0:'后柜顶面及部分面板侧面可见，同类前柜及变压器局部遮挡。',3:'近柜顶面、面板、侧面和基座完整清楚，无遮挡。',4:'近变压器顶面、大主体与两柱可见，右侧下侧图缘截断。'},
7:{0:'后柜顶面与上部细条可见，下部被同色队列遮挡。',1:'中柜顶面和面板上缘可见，下部被近柜遮挡。',2:'近柜大顶面、主体及面板上部可见，下图缘截断。',4:'右柜顶面、面板和基座清楚，左下局部被近柜遮挡。'},
8:{8:'远柜右侧主体、顶面和基座可见，左部被变压器遮挡，尺度较小。'},
9:{0:'右侧变压器大主体、顶面柱及基座可见，右图缘截断。',2:'近变压器大主体、三柱、顶面及基座清楚，下图缘截断。'},
10:{0:'左近柜大侧面与顶面可见，左下图缘截断，无面板。',1:'独立圆柱主体和基座清楚，无遮挡。'},
11:{2:'后柜顶部及侧面窄带可见，被同色前柜遮挡。',4:'变压器三柱、顶面、主体及基座清楚，未见主要前景遮挡。',8:'远柜顶面、右侧面及基座可见，左部被变压器遮挡。',9:'后电容器顶面和上部主体可见，下部被变压器遮挡；框内三柱属于前景。'},
12:{7:'圆柱顶面、主体和基座部分可见，右图缘截断。'},
13:{1:'后电容器顶面、大主体上部可见，左下被前柜遮挡。'},
14:{3:'远柜顶面和侧面细带可见，队列遮挡且位于右图缘。',10:'后电容器顶面及主体上部可见，前景杆与变压器遮挡，不将框内前景当作目标部件。'},
15:{5:'后变压器三柱、顶面和主体上部可见，下部被前变压器遮挡。',8:'近圆柱完整顶面、大主体与基座清楚，无遮挡无截断。'},
}
FP={
0:('gray_block_body','冷暗灰块体大侧背面、侧面暗面板及基座，右边贴近杆体。'),
1:('mixed_structure','灰块体和前景粗杆重叠，右下青色块体进入框内。'),
2:('mixed_structure','冷暗灰块体、前景粗杆及右下青色块体共同占框。'),
3:('mixed_structure','灰块体大侧面及基座为主，右部含竖杆和前景青色块体。'),
4:('gray_block_body','灰块体正面暗矩形面板、灰边框及基座。'),
5:('mixed_structure','灰块体大面与前景粗杆、右下青色块体共同入框。'),
6:('gray_block_body','正面灰块体的暗面板、灰边框和底座。'),
7:('gray_block_body','冷暗正面灰块体、暗面板、灰边框和基座。'),
8:('teal_block_body','青色块体背侧两个大面、顶面及基座，无可见正面面板，非旁边杆体。'),
9:('gray_block_body','灰块体侧背大面、侧面暗面板及基座。'),
10:('gray_block_body','冷暗灰块体大侧背面、侧面暗面板及底座，右缘贴近杆体。'),
11:('mixed_structure','灰块体侧背面及基座，右侧粗杆和青色前景共同入框。'),
12:('gray_block_body','灰块体正面暗面板与灰色边框、基座。'),
13:('gray_block_body','右图缘灰块体面板、边框和基座局部，严重右缘截断，框左半还有墙地边界。'),
}

def run():
    ep=OUT/'evidence.json';np=TRAIN/'evaluation-v1/error-review-v1/evidence.json';e,n=checked(ep),checked(np)
    positive=[];negative=[];stamp=datetime.now(timezone.utc).isoformat()
    for f in e['frames']:
        for x in f['events']:
            positive.append(dict(frame_id=f['frame_id'],seed=x['seed'],truth_index=x['truth_index'],truth=x['truth'],diagnosis=x['diagnosis'],loss_against=x['loss_against'],
                reason=NOTES[int(f['frame_id'].split('-')[-1])][x['truth_index']],review_nature='AI辅助审核',reviewed_at=stamp,image_sha256=f['image_sha256'],
                evidence_sha256=f['evidence_sha256'],pixel_visibility_certified=False,decision='content_observed_not_label_admission'))
    for f in n['frames']:
        for x in f['events']:
            category,reason=FP[int(x['event_id'].split('-')[-1])]
            negative.append(dict(event_id=x['event_id'],seed=x['seed'],prediction=x['prediction'],content_category=category,reason=reason,review_nature='AI辅助审核',reviewed_at=stamp,
                image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],asset_identity='unknown_visual_shape_not_asset_identity'))
    validate_positive(e,positive);validate_review(n,negative)
    return write_record(OUT/'review.json',dict(status='current_errors_reviewed_not_candidate_passed',positive_decisions=positive,negative_decisions=negative,
        limits='AI visual observations only; no mask certification, historical label revision, source-independence or training admission. All transitions retained.',
        training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in (ep,np,Path(__file__))}))

if __name__=='__main__':print(run()['status'])
