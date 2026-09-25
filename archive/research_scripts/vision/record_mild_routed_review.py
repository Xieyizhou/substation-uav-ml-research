"""Explicit current-frame observations, not automatic approval."""
from pathlib import Path
from datetime import datetime, timezone
from scripts.vision import record_dose_matched_review as writer
from scripts.vision.mild_routed_contrast_control import OUT as TRAIN, checked
from scripts.vision.record_contrast_cpu4_review import validate_positive
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
OUT=TRAIN/'audit-v1'
NOTES={
0:{0:'右侧变压器主体、顶柱和基座清楚，右图缘截断。',1:'后变压器三柱、顶面和主体上部可见，下部被近变压器遮挡，框内近柱不是其部件。',2:'近变压器三柱、顶面和两个大主体面清楚，下图缘截断。'},
1:{2:'左缘柜体顶面、大侧背面和基座可见，左图缘截断，无正面面板。'},
2:{7:'右缘圆柱顶面、主体和基座可见，右图缘截断。'},
3:{0:'后柜顶面及上部窄带可见，下部被同类队列遮挡。'},
4:{3:'近柜顶面、面板、侧面及基座清楚，无遮挡。',5:'后柜顶面及面板上部可见，下部被近变压器遮挡。'},
5:{0:'右侧变压器三柱、顶面、大主体及基座清楚，右图缘截断。'},
6:{3:'近柜面板、顶面、侧面和基座清楚。',4:'近变压器三柱与大顶面主体可见，右下图缘截断。'},
7:{0:'后柜顶面和细带可见，被同类队列遮挡。',1:'中柜顶面、面板上缘细线可见，下部被前柜遮挡。',4:'右柜面板、侧面、顶面和基座清楚，左下小部被近柜遮挡。'},
8:{2:'队列后柜顶面与侧面窄带可见，下部被近柜遮挡。'},
9:{3:'后圆柱顶面和主体上部可见，下部被变压器遮挡；框内小柱属于前景。'},
10:{2:'队列后柜顶面与窄带可见，下部遮挡。',8:'远柜右侧面、顶面和基座可见，左部被变压器遮挡，尺度小。',9:'后电容器顶面及主体上部可见，下部被变压器遮挡，框内三柱属于前景。'},
11:{4:'左变压器顶柱、主体和基座可见，粗杆遮挡且左图缘截断。',7:'右缘圆柱主体、顶面和基座部分可见，右图缘截断。'},
12:{1:'右侧柜体侧面与基座局部可见，图缘截断并受前柜遮挡。'},
13:{8:'近圆柱顶面、完整主体和基座清楚，无遮挡。'},
}
MATERIAL={
0:{1:'后灰柜顶面、大侧面及面板部分可见，左前有柜体遮挡。',2:'灰柜顶面、侧面、面板轮廓及基座清楚。',5:'后灰柜顶部和面板上半可见，下半被变压器遮挡。'},
1:{0:'右缘灰变压器三柱、顶面、大主体和底座可见，右图缘截断。',2:'近灰变压器三柱、顶面、大主体及底座清楚，下图缘截断。'},
2:{0:'灰变压器三柱、顶面、主体和底座清楚，右图缘截断。'},
3:{1:'后灰柜顶面与大侧面可见，右下被近柜遮挡。',2:'远灰柜顶面、大侧背面和底座清楚，无面板。',4:'左灰变压器三柱、顶面及主体可见，粗杆遮挡且左缘截断。',5:'近灰变压器三柱、大顶面与主体上部可见，下图缘截断。',6:'中灰柜顶面、侧面上部和部分底座可见，下部被近变压器遮挡。'},
4:{4:'远灰变压器三柱、顶面、主体和底座清楚，未见主要遮挡。'},
5:{0:'灰变压器三柱、顶面、主体和底座可见，前景粗杆遮挡中部。'},
}
FP={
0:('mixed_structure','灰块体大侧面、侧面面板及基座，右侧包含杆体和青色前景块体。'),
1:('gray_block_body','灰块体正面暗面板、边框及基座。'),
2:('mixed_structure','灰块体大面、前景粗杆及右下青色块体共同入框。'),
3:('gray_block_body','灰块体正面暗面板、边框及底座。'),
4:('gray_block_body','冷暗灰块体正面面板、灰边框及基座。'),
5:('mixed_structure','冷暗灰块体大面、前景杆体及青色块体局部。'),
6:('mixed_structure','灰块体大侧面、侧面面板、前景杆体及青色块体。'),
7:('gray_block_body','灰块体正面暗面板、灰边框及底座。'),
8:('gray_block_body','冷暗灰块体大侧背面、侧面暗面板及基座，框右缘贴近杆体。'),
}
def run():
    writer.TRAIN=TRAIN;writer.OUT=OUT;writer.NOTES=NOTES;writer.FP=FP
    rp=OUT/'review.json'
    if not rp.exists():writer.run()
    ep=OUT/'material-evidence.json';e=checked(ep);decisions=[]
    for f in e['frames']:
        for x in f['events']:
            decisions.append(dict(frame_id=f['frame_id'],seed=x['seed'],truth_index=x['truth_index'],truth=x['truth'],diagnosis=x['diagnosis'],loss_against=x['loss_against'],
                reason=MATERIAL[int(f['frame_id'].split('-')[-1])][x['truth_index']],review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),
                image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],pixel_visibility_certified=False,decision='content_observed_not_label_admission'))
    validate_positive(e,decisions)
    return write_record(OUT/'material-review.json',dict(status='material_loss_reviewed_not_candidate_passed',positive_decisions=decisions,
        training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in (ep,rp,Path(__file__))}))
if __name__=='__main__':print(run()['status'])
