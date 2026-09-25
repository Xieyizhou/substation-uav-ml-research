"""Explicit AI observations of the 22 current fixed-BN loss pages."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision import routed_fixed_bn_control as arm
from scripts.vision.record_contrast_cpu4_review import validate_positive
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

NOTES = {
0:{1:'电容器大正面和完整基座清楚，面上斜影不是前景遮挡。'},
1:{0:'右缘变压器三柱、顶面、主体及基座可见，右缘截断。',2:'近变压器三柱、大顶面及两侧主体清楚，下缘截断。',3:'后电容器顶面及主体上部可见，下部被青块及近变压器遮挡，前景柱不属于它。'},
2:{0:'中右变压器三柱、顶面、主体及基座可见，粗前景杆遮挡中部。'},
3:{0:'中右变压器三柱、主体及基座可见，粗杆遮挡。',1:'近左变压器三柱、顶面及大主体可见，左下缘截断。',2:'后电容器顶面、主体和右基座可见，左下部被青块遮挡。'},
4:{3:'后圆柱顶面和上部主体可见，下部被变压器遮挡，框内小柱属于前景。'},
5:{4:'左变压器主体、顶面、柱和基座可见，粗杆遮挡且左缘截断。',5:'近变压器三柱、大顶面及上部主体可见，下缘截断。',7:'右圆柱顶面、主体和部分基座可见，右缘截断。'},
6:{1:'独立圆柱主体和完整基座清楚，无主要前景遮挡，低视角顶面不明显。'},
7:{0:'后柜仅顶面及窄带可见，大部被前方柜列遮挡。',2:'近柜大顶面、上部主体及面板上缘可见，下缘截断。',4:'右柜面板、顶面及右基座可见，左下被近柜遮挡。',5:'后圆柱仅上部窄带可见，其余被柜体遮挡。'},
8:{3:'近柜面板、顶面、侧面及基座清楚，无主要遮挡。',4:'近变压器三柱、大顶面及主体可见，右下缘截断。',5:'后柜顶面及面板上部可见，下部被变压器遮挡。'},
9:{0:'右变压器三柱、顶面、大主体及基座清楚，右缘截断。'},
10:{0:'远柜顶面、右侧及面板窄带可见，左下被前柜及前景柱遮挡。',3:'近柜面板、顶面、侧面及基座清楚。',4:'近变压器三柱、大顶面及主体部分可见，右下缘截断。'},
11:{5:'后圆柱仅主体上部窄带可见，其余被柜体遮挡。'},
12:{2:'右列后柜顶面及窄带可见，下部被前柜遮挡。',4:'中部变压器三柱、顶面、主体及基座清楚。',6:'近左变压器三柱、顶面、主体及基座可见，左缘截断。',8:'远柜右主体、顶面及右基座可见，左部被变压器遮挡，尺度较小。'},
13:{1:'电容器大正面和完整基座清楚，面上阴影不属于前景遮挡。'},
14:{1:'变压器三柱、大顶面、主体两面及完整基座清楚。',3:'后圆柱顶面和上部可见，下部被变压器遮挡，框内小柱为前景。'},
15:{0:'右缘变压器三柱、顶面、主体及基座可见，右缘截断。',2:'近变压器三柱、大顶面和两侧主体可见，下缘截断。'},
16:{0:'近左柜大顶面及侧背面可见，左下缘截断，无正面面板。',1:'圆柱主体和完整基座清楚，无主要遮挡，低视角顶面不明显。'},
17:{2:'右列后柜顶部及窄带可见，下部被近柜遮挡，框内含前景柜。',4:'中远变压器三柱、顶面、主体和基座清楚。',5:'远变压器三柱、顶面及主体可见，粗杆遮挡右部。',6:'近左变压器三柱、顶面、主体及基座可见，左缘截断。',7:'近柜大顶面、侧背面及完整基座清楚，无面板正面。',8:'远柜右主体、顶面和右基座可见，左部被变压器遮挡。',9:'后电容器顶面及上部主体可见，下部遮挡，框内三柱属于前景。'},
18:{4:'左变压器柱、顶面、主体及基座可见，粗杆遮挡且左缘截断。',7:'右圆柱顶面、主体及部分基座可见，右缘截断。'},
19:{0:'右变压器三柱、大顶面、主体及基座清楚，右缘截断。'},
20:{1:'右缘柜侧面及基座局部可见，右缘截断并被近柜遮挡。',4:'远变压器三柱、顶面及主体可见，下部局部被柜遮挡。',10:'后电容器顶面及上部主体可见，粗杆与近变压器遮挡。'},
21:{8:'近圆柱顶面、主体及完整基座清楚，无主要前景遮挡。'},
}

def run():
    ep=arm.OUT/'audit-v1/evidence.json'
    evidence=arm.checked(ep)
    expected={(f['frame_id'],e['truth_index']) for f in evidence['frames'] for e in f['events']}
    assert expected=={(f'loss-{i:02}',j) for i,notes in NOTES.items() for j in notes}
    stamp=datetime.now(timezone.utc).isoformat()
    decisions=[]
    for f in evidence['frames']:
        for e in f['events']:
            decisions.append(dict(frame_id=f['frame_id'],seed=e['seed'],truth_index=e['truth_index'],truth=e['truth'],diagnosis=e['diagnosis'],loss_against=e['loss_against'],reason=NOTES[int(f['frame_id'].split('-')[-1])][e['truth_index']],review_nature='AI辅助审核',reviewed_at=stamp,image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],pixel_visibility_certified=False,decision='content_observed_not_label_admission'))
    validate_positive(evidence,decisions)
    dest=arm.OUT/'audit-v1/loss-review.json'
    if dest.exists(): return arm.checked(dest)
    return write_record(dest,dict(status='positive_losses_reviewed_negative_review_pending',positive_decisions=decisions,training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in (ep,Path(__file__))}))

if __name__=='__main__':
    print(len(run()['positive_decisions']))
