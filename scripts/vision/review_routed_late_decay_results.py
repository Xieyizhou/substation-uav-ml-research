"""Explicit AI observations of current late-decay pages, not inherited reviews."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision import routed_late_decay_control as arm
from scripts.vision.record_contrast_cpu4_review import validate_positive
from scripts.vision.audit_reviewed_order_results import validate_review
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

LOSS={
0:{1:'后变压器三柱、顶面和上部主体可见，下部被近变压器遮挡，框内近柱不是后实例组件。'},
1:{2:'左缘柜体顶面、侧背面和基座可见，左缘截断，无面板正面。',3:'后圆柱顶面和主体上部可见，下部被变压器遮挡，框内小柱为前景。'},
2:{7:'右缘圆柱顶面、主体和部分基座可见，右缘截断。'},
3:{1:'圆柱主体和完整基座清楚，无主要前景遮挡，低视角顶面不明显。'},
4:{0:'后柜顶部窄带可见，大部被同类柜列遮挡。',1:'中间柜顶面、主体上带和面板窄条可见，下部被近柜遮挡。'},
5:{4:'近变压器大顶面、柱和部分主体清楚，右下图缘截断。'},
6:{0:'后柜顶面、右侧和面板局部可见，下部被柜列和前景变压器遮挡。',3:'近柜面板、顶面、大侧面及基座清楚。',4:'近变压器大顶面、柱和主体部分可见，右下缘截断。'},
7:{0:'后柜仅顶面和窄带可见，大部柜列遮挡。',1:'中间柜顶面和面板窄条可见，下部近柜遮挡。',2:'近柜大顶面、主体上部及面板上部可见，下缘截断。'},
8:{0:'左近柜大顶面和侧背面可见，左下缘截断。',1:'圆柱主体和完整基座清楚，无主要遮挡。'},
9:{2:'右后柜顶面和窄带可见，下部被近柜遮挡。',9:'后电容器顶面和上部主体可见，下部被近变压器遮挡，三柱属于前景。'},
10:{7:'右缘圆柱顶面、主体及基座部分可见，右缘截断。'},
11:{1:'右缘柜侧面和基座局部可见，图缘截断并有近柜遮挡。'},
12:{8:'近圆柱顶面、主体及完整基座清楚，无主要前景遮挡。'},
}
MATERIAL={
0:{1:'灰柜顶面、侧面、面板和基座部分可见，左部被近柜遮挡。',2:'灰柜顶面、面板、大侧面和基座清楚，无主要遮挡。'},
1:{4:'灰柜顶面、面板和右基座可见，左下被近柜遮挡。'},
2:{0:'右缘灰变压器三柱、顶面、主体和基座可见，右缘截断。'},
3:{0:'灰变压器三柱、大顶面、主体及基座清楚，右缘截断。'},
4:{4:'远灰变压器三柱、顶面、主体可见，下部部分被近柜遮挡。'},
5:{1:'灰圆柱主体和完整基座清楚，无主要遮挡，低视角顶面不明显。'},
6:{4:'左灰变压器柱、顶面、主体和基座可见，粗杆遮挡且左缘截断。'},
7:{5:'远灰变压器三柱、顶面、主体和基座可见，粗杆遮挡右侧。'},
}
FP={
0:('gray_block_body','冷暗灰块体大侧背面、右暗面板及基座，右缘含杆体窄带。'),
1:('mixed_structure','灰块体侧背面、暗面板、杆体和右下青块共同入框。'),
2:('gray_block_body','灰块体正面暗面板、边框及底部，同图跨seed重复结构。'),
3:('gray_block_body','冷暗灰块体正面暗面板、边框和基座。'),
4:('mixed_structure','远灰块体侧背面、粗杆和青块局部共同入框。'),
5:('gray_block_body','灰块体正面暗面板、边框及底部，同图跨seed重复结构。'),
6:('gray_block_body','灰块体大侧背面、右暗面板及基座清楚。'),
7:('mixed_structure','远灰块体侧背面、粗杆和青块局部，重复同图混合结构但预测类别不同。'),
8:('mixed_structure','冷暗远灰块体侧背面、粗杆及青块局部共同入框。'),
9:('gray_block_body','灰块体正面暗面板、边框及基座，同图三个seed均误检。'),
}

def positives(e,notes,stamp):
    observed={(f['frame_id'],x['truth_index']) for f in e['frames'] for x in f['events']}
    prefix=e['frames'][0]['frame_id'].rsplit('-',1)[0]
    if observed!={(f'{prefix}-{i:02}',j) for i,d in notes.items() for j in d}:raise ValueError('Observation coverage mismatch')
    result=[]
    for f in e['frames']:
        for x in f['events']:
            result.append(dict(frame_id=f['frame_id'],seed=x['seed'],truth_index=x['truth_index'],truth=x['truth'],diagnosis=x['diagnosis'],loss_against=x['loss_against'],reason=notes[int(f['frame_id'].split('-')[-1])][x['truth_index']],review_nature='AI辅助审核',reviewed_at=stamp,image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],pixel_visibility_certified=False,decision='content_observed_not_label_admission'))
    validate_positive(e,result);return result

def run():
    out=arm.OUT/'audit-v1';paths=[out/'evidence.json',out/'material-evidence.json',arm.OUT/'evaluation-v1/error-review-v1/evidence.json']
    e,m,n=map(arm.checked,paths);stamp=datetime.now(timezone.utc).isoformat()
    pos=positives(e,LOSS,stamp);mat=positives(m,MATERIAL,stamp);neg=[]
    assert {x['event_id'] for f in n['frames'] for x in f['events']}=={f'fp-{i:02}' for i in FP}
    for f in n['frames']:
        for x in f['events']:
            category,reason=FP[int(x['event_id'].split('-')[-1])]
            neg.append(dict(event_id=x['event_id'],seed=x['seed'],prediction=x['prediction'],content_category=category,reason=reason,review_nature='AI辅助审核',reviewed_at=stamp,image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],asset_identity='unknown_visual_shape_not_asset_identity'))
    validate_review(n,neg);paths.append(Path(__file__))
    dest=out/'review.json'
    if dest.exists():return arm.checked(dest)
    return write_record(dest,dict(status='current_error_content_reviewed_not_candidate_passed',positive_decisions=pos,material_decisions=mat,negative_decisions=neg,training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in paths}))

if __name__=='__main__':print(run()['status'])
