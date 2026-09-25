"""Explicit observations of the current backbone arm; no automated approval."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.routed_backbone_control import OUT, checked
from scripts.vision.record_routed_gray_results_review import positive
from scripts.vision.audit_reviewed_order_results import validate_review
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

NOTES = {
0:{0:'右变压器三柱、顶面、大主体与基座可见，右图缘截断。',2:'近变压器三柱、顶面、两个大主体面和基座部分可见，下图缘截断。',3:'后电容器顶面及主体上部可见，下部被青色块体和近变压器遮挡；前景柱不属于该目标。'},
1:{1:'近左变压器三柱、大顶面及主体可见，左下图缘截断。',2:'后电容器顶面、大主体和右基座可见，左下被青色块体遮挡。'},
2:{2:'左缘柜体大侧背面、顶面及基座可见，左图缘截断，无正面面板。',3:'后圆柱顶面及上部可见，下部被变压器遮挡；前景柱不属于圆柱。'},
3:{0:'独立柜体大顶面、两个主体面及完整基座清楚，无正面面板，未见主要遮挡或截断。'},
4:{0:'队列后柜顶面和极窄主体带可见，下部被前柜遮挡；框内包含前景柜体。',1:'中柜顶面及面板上缘细线可见，下部被近柜遮挡。',2:'近柜大顶面、主体及前面板上部可见，下图缘截断。',4:'右柜顶面、暗面板、侧面及基座清楚，左下角被近柜遮挡。'},
5:{3:'近柜顶面、前面板、侧面及完整基座清楚。',4:'近变压器大顶面、主体和两根顶柱可见，右下图缘截断。',5:'后柜顶面及前面板上部可见，下部被变压器遮挡；前景柱不是柜体部件。'},
6:{0:'右变压器三柱、顶面、两个大主体面及基座清楚，右图缘截断。'},
7:{3:'本光照图近柜顶面、前面板、侧面及完整基座清楚。',4:'本光照图近变压器大顶面、主体及两根顶柱可见，右下图缘截断。',5:'本光照图后柜顶面及面板上部可见，下部被变压器遮挡，前景柱不是柜体部件。'},
8:{0:'本光照图队列后柜顶面和极窄主体带可见，下部被前柜遮挡。',1:'本光照图中柜顶面及面板上缘细线可见，下部被近柜遮挡。',2:'本光照图近柜大顶面、主体及面板上部可见，下图缘截断。',4:'本光照图右柜顶面、暗面板、侧面与基座清楚，左下角被近柜遮挡。'},
9:{1:'队列柜大顶面及主体带可见，下部被前柜遮挡。',2:'更后柜顶面及薄主体带可见，下部被前柜遮挡。',7:'中央独立柜顶面、两个主体面和完整基座清楚，无正面面板。'},
10:{2:'近变压器三柱、大顶面及两个主体面可见，下图缘截断。'},
11:{0:'近左柜大顶面、侧面和部分基座可见，左下图缘截断，无正面面板。'},
12:{1:'队列柜顶面及主体带可见，下部被前柜遮挡。',2:'更后柜顶面及极窄主体带可见，下部被前柜遮挡。',4:'后变压器三柱、顶面、大主体与基座清楚，无主要遮挡。',7:'中央柜顶面、两个主体面及完整基座清楚，无正面面板。',9:'后电容器顶面与主体上部可见，下部被近变压器遮挡；三柱属于前景。'},
13:{1:'后电容器顶面、大主体上部与右基座可见，下部被近青色块体遮挡。'},
14:{1:'右缘柜侧面及部分基座可见，右图缘截断并被近柜遮挡。',7:'右柜顶面、大侧背面及完整基座清楚，无正面面板。',10:'远电容器顶面与上部可见，左粗杆和近变压器遮挡，不能将前景部件归属该目标。'},
15:{7:'本光照图右柜顶面、大侧背面与完整基座清楚，无正面面板，未见主要遮挡。'},
}
MATERIAL = {
0:{1:'后灰柜顶面、大侧面及部分前面板可见，左前近柜遮挡。',2:'灰柜顶面、侧面、面板轮廓及基座清楚。',5:'后灰柜顶面及面板上部可见，下部被近变压器遮挡；前景柱不是柜体部件。'},
1:{2:'左缘灰柜大侧背面、顶面及基座可见，左图缘截断，无正面面板。'},
2:{3:'右灰变压器三柱、顶面、两个主体面及基座可见，右图缘截断。',4:'灰柜顶面、面板轮廓、侧面与右基座可见，左下被近柜遮挡。'},
3:{0:'右灰变压器三柱、顶面、大主体和基座可见，右图缘截断。',2:'近灰变压器三柱、大顶面、两个主体面与基座部分清楚，下图缘截断。'},
4:{0:'右灰变压器三柱、大顶面、两个主体面与基座清楚，右图缘截断。'},
5:{4:'远灰变压器三柱、顶面、大主体及基座可见，右下局部被前柜遮挡，主体并非仅窄条。'},
6:{1:'灰圆柱大主体与完整基座清楚，无主要前景遮挡；顶面未见，不能据此认证像素级可见性。'},
7:{1:'后灰柜顶面、大侧背面及左基座可见，右下被前柜遮挡。',4:'左灰变压器顶柱、顶面、两个主体面与基座可见，前景粗杆遮挡且左图缘截断。',5:'近灰变压器三柱、大顶面与主体上部可见，下图缘严重截断。',6:'中灰柜顶面及主体上部可见，下部被近变压器遮挡，左基座局部露出。'},
8:{1:'右队列柜顶面和侧面带可见，前柜遮挡下部且右图缘截断。',3:'右侧更后柜仅顶面及窄带可见，下部被前柜遮挡且受右图缘限制。',4:'中远灰变压器三柱、顶面、两个主体面及基座清楚，未见主要前景遮挡。'},
9:{1:'近左灰变压器三柱、大顶面及主体上部可见，左下图缘截断。',2:'后灰电容器顶面、大主体及右基座可见，左下被青色块体遮挡。'},
}

def run():
    audit=OUT/'audit-v1'
    paths=[audit/'evidence.json',audit/'material-evidence.json',OUT/'evaluation-v1/error-review-v1/evidence.json',OUT/'evaluation-v1/error-review-v1/content-review.json']
    e,m,n,nr=map(checked,paths)
    stamp=datetime.now(timezone.utc).isoformat()
    p=positive(e,NOTES,stamp);q=positive(m,MATERIAL,stamp)
    validate_review(n,nr['decisions'])
    deps=paths+[audit/'evidence-build-receipt.json',Path(__file__),Path('scripts/vision/record_routed_gray_results_review.py')]
    return write_record(audit/'review.json',dict(status='current_arm_error_content_reviewed_not_candidate_passed',positive_decisions=p,material_decisions=q,negative_decisions=nr['decisions'],counts=dict(original_lighting_events=len(p),material_events=len(q),negative_events=len(nr['decisions']),negative_unique_frames=len(n['frames'])),limits='AI visual review only; no mask certification or label admission; training-fit residual review remains separate.',training_admitted=False,promotable=False,inputs={str(x.resolve()):file_sha256(x) for x in deps}))

if __name__=='__main__':print(run()['counts'])
