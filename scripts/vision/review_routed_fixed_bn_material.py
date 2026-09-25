"""Explicit observations of current fixed-BN material-loss pages."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.inspect_routed_fixed_bn_results import OUT,arm
from scripts.vision.record_contrast_cpu4_review import validate_positive
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

NOTES={
0:{1:'后灰柜顶面、大侧面及部分面板可见，左下被近柜遮挡。',2:'灰柜顶面、侧面、面板轮廓和基座清楚。'},
1:{2:'左图缘灰柜大侧背面、顶面和基座可见，左缘截断，无正面面板。'},
2:{3:'右灰变压器三柱、顶面、两侧主体及基座清楚，右图缘截断。'},
3:{0:'右缘灰变压器三柱、顶面、主体和基座可见，右缘截断。',2:'近灰变压器三柱、大顶面和两侧主体清楚，下图缘截断。'},
4:{0:'右灰变压器三柱、顶面、大主体和基座可见，右缘截断。'},
5:{4:'远灰变压器三柱、顶面及主体清楚，下部局部被前柜遮挡。'},
6:{1:'灰圆柱主体及完整基座清楚，无遮挡；低视角下顶面不明显。'},
7:{4:'左灰变压器顶柱、大主体与基座可见，粗杆遮挡且左缘截断。',5:'近灰变压器三柱、大顶面和主体上部可见，下图缘截断。',6:'中灰柜顶面和主体上部及左基座可见，下部被近变压器遮挡。'},
8:{1:'近队列柜体大顶面和上部窄带可见，下部被更近柜体遮挡。',3:'最远队列柜顶面和窄带可见，下部被前柜遮挡，右缘受图边限制。',4:'中灰变压器三柱、顶面、主体和基座清楚，无主要遮挡。',5:'远灰变压器三柱、顶面和主体及基座可见，杆体遮挡右部。',6:'近左灰变压器三柱、顶面、主体及基座可见，左缘截断。'},
9:{0:'右灰变压器三柱、顶面、主体和基座可见，粗杆遮挡中部。',1:'近左灰变压器三柱、顶面和大主体可见，左下图缘截断。',2:'后灰电容器顶面、大侧面及右基座可见，左下被青块遮挡。'},
}

def run():
    ep=OUT/'material-evidence.json';e=arm.checked(ep);decisions=[];stamp=datetime.now(timezone.utc).isoformat()
    expected={(f['frame_id'],x['truth_index']) for f in e['frames'] for x in f['events']}
    if expected!={(f'material-{n:02}',i) for n,rows in NOTES.items() for i in rows}:raise ValueError('Observation coverage mismatch')
    for f in e['frames']:
        for x in f['events']:
            decisions.append(dict(frame_id=f['frame_id'],seed=x['seed'],truth_index=x['truth_index'],truth=x['truth'],diagnosis=x['diagnosis'],loss_against=x['loss_against'],reason=NOTES[int(f['frame_id'].split('-')[-1])][x['truth_index']],review_nature='AI辅助审核',reviewed_at=stamp,image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],pixel_visibility_certified=False,decision='content_observed_not_label_admission'))
    validate_positive(e,decisions)
    dest=OUT/'material-review.json'
    if dest.exists():return arm.checked(dest)
    return write_record(dest,dict(status='material_loss_content_reviewed_other_errors_pending',positive_decisions=decisions,training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in (ep,Path(__file__))}))

if __name__=='__main__':print(len(run()['positive_decisions']))
