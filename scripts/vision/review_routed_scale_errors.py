"""Explicit current-scale observations; all 18 loss and six FP pages viewed."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.inspect_routed_scale_results import OUT,arm
from scripts.vision.record_contrast_cpu4_review import validate_positive
from scripts.vision.audit_reviewed_order_results import validate_review
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

NOTES={
0:{0:'右缘变压器三柱、顶面、主体及基座可见，右缘截断。',1:'后变压器三柱和主体上部可见，下部被近变压器遮挡；前景柱不属于后实例。',2:'近变压器三柱、大顶面和两侧主体清楚，下图缘截断。',3:'后电容器顶面和主体上部可见，下部被青块和变压器遮挡。'},
1:{2:'左缘柜体大侧背面、顶面和基座可见，左缘截断，无面板正面。',3:'后圆柱顶面和主体上部可见，下部被变压器遮挡；小柱属于前景。'},
2:{4:'左变压器顶柱、主体和基座可见，粗杆遮挡中部且左缘截断。',7:'右缘圆柱顶面、主体及基座部分可见，右图缘截断。'},
3:{1:'独立圆柱主体及完整基座清楚，无主要前景遮挡，低视角下顶面不明显。'},
4:{0:'队列后柜仅顶面及窄带可见，大部分被近柜遮挡，框内含前景。',2:'近柜大顶面、主体和面板上部可见，下缘截断。',5:'后圆柱只有主体上部可见，下部被柜体遮挡。'},
5:{3:'近柜面板、大侧面、顶面及基座清楚，无主要遮挡。',5:'后柜顶面、面板上部可见，下部被变压器遮挡。'},
6:{4:'近变压器顶柱、顶面及主体部分清楚，右下图缘截断。',5:'后柜顶面与面板上部可见，下部被变压器遮挡。'},
7:{0:'后柜顶部窄带可见，下部被同类队列遮挡。',2:'近柜大顶面和主体上部可见，下图缘截断。',4:'右柜面板、顶面及右基座可见，左下被近柜遮挡。',5:'后圆柱上部窄带可见，其余被柜体遮挡。'},
8:{2:'右队列后柜顶面和窄带可见，下部被近柜遮挡。',8:'远柜右侧主体、顶面及基座可见，左部被变压器遮挡，尺度较小。'},
9:{1:'电容器大正面及完整基座清楚，斜线为主体上的阴影，无主要前景遮挡。'},
10:{2:'左缘柜体大侧背面、顶面及基座可见，左缘截断。'},
11:{1:'后变压器三柱、顶面和主体上部可见，下部被近变压器遮挡。'},
12:{0:'近左柜大顶面和侧面清楚，左下图缘截断。',1:'圆柱主体及完整基座清楚，无主要前景遮挡。'},
13:{2:'队列后柜顶面及窄带可见，下部遮挡。',4:'中远变压器三柱、顶面、主体和基座清楚。',8:'远柜右主体、顶面和右基座可见，左部被变压器遮挡。',9:'后电容器顶面和主体上部可见，下部遮挡，框内三柱属于前景。'},
14:{7:'右缘圆柱顶面、主体与基座部分可见，右缘截断。'},
15:{1:'后电容器顶面和大主体上部及右基座可见，左下被青块遮挡。'},
16:{1:'右缘柜体侧面及基座局部可见，右缘截断并被近柜遮挡。',4:'远变压器三柱、顶面和主体可见，下部局部被柜体遮挡。',8:'近圆柱顶面、主体和完整基座清楚，无遮挡。',10:'后电容器顶面和主体上部可见，粗杆及近变压器遮挡。'},
17:{5:'后变压器三柱、顶面和主体上部可见，下部被近变压器遮挡。',8:'近圆柱顶面、主体和完整基座清楚，无主要遮挡。'},
}
FP={
0:('mixed_structure','灰块体大背侧面及基座、粗杆和青块局部共同入框。'),
1:('mixed_structure','灰块体大侧面及暗面板、杆体和青块局部入框。'),
2:('gray_block_body','灰块体正面暗面板、边框和底部基座。'),
3:('gray_block_body','冷暗灰块体正面暗面板、边框和基座。'),
4:('ground_shadow','框内为地面网格及杆体投影，竖杆实体在框外。'),
5:('mixed_structure','灰块体大背侧面、粗杆及右下青块局部入框。'),
6:('gray_block_body','灰块体正面暗面板、边框与基座；与另两个seed覆盖同一结构。'),
7:('gray_block_body','冷暗灰块体大侧背面、右暗面板及基座；右框缘含杆体窄带。'),
8:('mixed_structure','灰块体大背侧面、粗杆和右下青块局部，跨seed重复同一结构。'),
9:('gray_block_body','灰块体正面暗面板、边框和基座，跨seed重复同一结构。'),
}

def run():
    ep=OUT/'evidence.json';np=arm.OUT/'evaluation-v1/error-review-v1/evidence.json'
    e,n=map(arm.checked,(ep,np));stamp=datetime.now(timezone.utc).isoformat();positive=[];negative=[]
    for f in e['frames']:
        for x in f['events']:
            positive.append(dict(frame_id=f['frame_id'],seed=x['seed'],truth_index=x['truth_index'],truth=x['truth'],diagnosis=x['diagnosis'],loss_against=x['loss_against'],reason=NOTES[int(f['frame_id'].split('-')[-1])][x['truth_index']],review_nature='AI辅助审核',reviewed_at=stamp,image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],pixel_visibility_certified=False,decision='content_observed_not_label_admission'))
    for f in n['frames']:
        for x in f['events']:
            cat,reason=FP[int(x['event_id'].split('-')[-1])]
            negative.append(dict(event_id=x['event_id'],seed=x['seed'],prediction=x['prediction'],content_category=cat,reason=reason,review_nature='AI辅助审核',reviewed_at=stamp,image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],asset_identity='unknown_visual_shape_not_asset_identity'))
    validate_positive(e,positive);validate_review(n,negative)
    dest=OUT/'other-errors-review.json'
    if dest.exists():return arm.checked(dest)
    return write_record(dest,dict(status='error_content_reviewed_not_candidate_passed',positive_decisions=positive,negative_decisions=negative,training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in (ep,np,Path(__file__))}))

if __name__=='__main__':
    r=run();print(len(r['positive_decisions']),len(r['negative_decisions']))
