"""Explicit AI observations after viewing all nine current material evidence pages."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.inspect_routed_scale_results import OUT,arm
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

NOTES = {
0:{1:'后灰柜顶面、大侧面及部分面板可见，左下被近柜遮挡。',2:'灰柜顶面、大侧面、面板轮廓及基座清楚。',5:'后灰柜顶面和面板上部可见，下部被前景变压器遮挡。'},
1:{2:'左图缘灰柜大侧背面、顶面和基座可见，左缘截断，正面面板不可见。'},
2:{4:'灰柜面板轮廓、顶面、侧面及右基座可见，左下被近柜遮挡。'},
3:{0:'右缘灰变压器三柱、顶面、主体及基座可见，右缘截断。',2:'近灰变压器三柱、大顶面和两侧主体清楚，下图缘截断。'},
4:{4:'远灰变压器三柱、顶面及主体可见，右下被前景柜体遮挡。'},
5:{1:'灰圆柱主体及完整基座清楚，无主要前景遮挡；低视角下顶面不可辨，不是只有基座。'},
6:{1:'远灰柜顶面、主体上部及左基座可见，右下被前景块体遮挡。',4:'左灰变压器顶柱、顶面、主体和基座可见，粗杆遮挡中部且左缘截断。',5:'近灰变压器三柱、大顶面及主体上部可见，下图缘截断。'},
7:{5:'远灰变压器三柱、顶面、主体及基座可见，前景杆遮挡右部。'},
8:{1:'近左灰变压器三柱、顶面及主体可见，左下图缘截断。'},
}

def run():
    p=OUT/'material-evidence.json';e=arm.checked(p);decisions=[]
    stamp=datetime.now(timezone.utc).isoformat()
    observed={(f['frame_id'],x['truth_index']) for f in e['frames'] for x in f['events']}
    authored={(f'material-{n:02}',i) for n,rows in NOTES.items() for i in rows}
    if observed!=authored:raise ValueError('Explicit observation coverage mismatch')
    for f in e['frames']:
        for x in f['events']:
            decisions.append(dict(frame_id=f['frame_id'],seed=x['seed'],truth_index=x['truth_index'],
                truth=x['truth'],diagnosis=x['diagnosis'],reason=NOTES[int(f['frame_id'].split('-')[-1])][x['truth_index']],
                image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],
                review_nature='AI辅助审核',reviewed_at=stamp,pixel_visibility_certified=False,
                decision='content_observed_not_label_admission'))
    dest=OUT/'material-review.json'
    if dest.exists():return arm.checked(dest)
    return write_record(dest,dict(status='material_losses_reviewed_other_errors_pending',decisions=decisions,
        training_admitted=False,promotable=False,
        inputs={str(q.resolve()):file_sha256(q) for q in (p,Path(__file__))}))

if __name__=='__main__':print(len(run()['decisions']))
