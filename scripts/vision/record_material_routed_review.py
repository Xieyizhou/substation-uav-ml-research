"""Explicit current-arm visual observations; no label or admission changes."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.inspect_material_routed_results import OUT,TRAIN,checked
from scripts.vision.record_contrast_cpu4_review import validate_positive
from scripts.vision.audit_reviewed_order_results import validate_review
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

NOTES={
0:{0:'右侧变压器大主体、基座和顶部柱可见，右图缘截断。',1:'后方变压器顶面、三柱及主体上部可见，下部被近处变压器遮挡。',2:'近处变压器主体、顶面三柱及基座清楚，下图缘截断。',3:'后方电容器顶部及主体上部可见，下部被前景柜体和变压器遮挡；框内圆柱属于前景。'},
1:{2:'左缘柜体顶面、大侧面和基座可见，左边截断，无正面面板。',3:'圆柱顶面及上部主体可见，下部被变压器遮挡；小柱为前景变压器结构。'},
2:{1:'圆柱主体和基座清楚，未见主要前景遮挡，不是仅基座。'},
3:{0:'独立柜体顶面、两个大侧面及基座清楚，无遮挡无截断，此视角不见面板。'},
4:{0:'后柜仅顶部和上部窄带可见，下部被同色前柜遮挡。',1:'中柜顶面及面板上缘细线可见，下部被近柜遮挡。'},
5:{1:'后方电容器顶面和大主体上部可见，左下被前景柜体遮挡。'},
6:{0:'远柜顶面及部分面板侧面可见，被同类前柜和变压器结构遮挡。',4:'近处变压器大主体、顶面和两柱可见，右侧及下侧图缘截断。'},
7:{1:'中柜顶面及面板上缘可见，下部被近柜遮挡。',2:'近柜大顶面、主体与面板上部可见，下图缘截断。'},
8:{3:'圆柱顶面和上部主体可见，下部被变压器遮挡；不能将框内前景小柱归于电抗器。'},
9:{1:'后变压器三柱、顶面及上部主体可见，下部被近处变压器遮挡。',2:'近变压器大主体、顶面三柱及基座可见，下图缘截断。'},
10:{0:'左近柜大顶面和侧面可见，左侧及下侧图缘截断，此视角无面板。'},
11:{2:'队列中后柜顶面和侧面窄带可见，下部被同色前柜遮挡。',4:'变压器三柱、顶面、主体和基座清楚，未见主要前景遮挡。',9:'后电容器顶部和主体上部可见，下部被变压器遮挡；框内三柱属于前景。'},
12:{7:'右缘圆柱顶面、主体与基座部分可见，右图缘截断。'},
13:{1:'后电容器顶面及主体上部可见，左下被前景柜体遮挡。'},
14:{1:'右缘中柜侧面及基座部分可见，图缘截断且与前柜重叠。',3:'远柜顶面和侧面窄条可见，被队列遮挡并受右图缘限制。'},
15:{3:'远柜顶面和侧面窄带可见，被同色前柜遮挡，位于右图缘。',8:'近处圆柱顶面、大主体和基座完整清楚，无遮挡无截断。'},
}
FP={
0:('gray_block_body','灰块体大侧背面、暗侧面板和基座，右缘贴近杆体及少量青色局部。'),
1:('gray_block_body','灰块体正面暗矩形面板、灰边框和基座。'),
2:('gray_block_body','冷暗条件下灰块体正面暗面板、灰边框和基座。'),
3:('mixed_structure','灰块体大面与前景粗竖杆共同占框，右下还有青色块体。'),
4:('gray_block_body','灰块体正面暗面板、边框和下缘基座，不是附近独立杆体。'),
5:('gray_block_body','灰块体侧背面、暗侧面板及基座，右边贴近杆体。'),
6:('mixed_structure','灰块体与前景粗杆及右下青色块体共同入框，左侧还有地面和墙边。'),
7:('mixed_structure','冷暗灰块体与前景粗杆重叠，右下青色结构进入框内。'),
8:('gray_block_body','灰块体正面暗矩形面板及边框、基座。'),
}

def run():
    ep=OUT/'evidence.json';np=TRAIN/'evaluation-v1/error-review-v1/evidence.json'
    e,n=checked(ep),checked(np);positive=[];negative=[];stamp=datetime.now(timezone.utc).isoformat()
    for f in e['frames']:
        for x in f['events']:
            positive.append(dict(frame_id=f['frame_id'],seed=x['seed'],truth_index=x['truth_index'],truth=x['truth'],diagnosis=x['diagnosis'],
                reason=NOTES[int(f['frame_id'].split('-')[-1])][x['truth_index']],review_nature='AI辅助审核',reviewed_at=stamp,
                image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],pixel_visibility_certified=False,
                loss_against=x['loss_against'],seed17_original_planned_miss=x['seed17_original_planned_miss'],decision='content_observed_not_label_admission'))
    for f in n['frames']:
        for x in f['events']:
            category,reason=FP[int(x['event_id'].split('-')[-1])]
            negative.append(dict(event_id=x['event_id'],seed=x['seed'],prediction=x['prediction'],content_category=category,reason=reason,
                review_nature='AI辅助审核',reviewed_at=stamp,image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],asset_identity='unknown_visual_shape_not_asset_identity'))
    validate_positive(e,positive);validate_review(n,negative)
    return write_record(OUT/'review.json',dict(status='current_errors_reviewed_not_model_passed',positive_decisions=positive,negative_decisions=negative,
        positive_frames=len(e['frames']),positive_events=len(positive),negative_frames=len(n['frames']),negative_events=len(negative),
        limits='No instance-mask visibility certification; no historical labels changed; repeated seeds are not independent images. Full transitions retained.',
        training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in (ep,np,Path(__file__))}))

if __name__=='__main__':print(run()['status'])
