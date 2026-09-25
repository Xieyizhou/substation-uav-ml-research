"""Explicit observations authored after viewing all 22 full-frame evidence pages."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.inspect_contrast_cpu4_results import ROOT,OUT,checked
from scripts.vision.audit_reviewed_order_results import validate_review
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

NOTES={
 'loss-00':{2:'左图缘截断柜体；顶面、侧面和基座较大部分可见，此背侧视角无面板。'},
 'loss-01':{0:'独立柜体顶面、两侧和基座完整清楚，无遮挡；此视角无面板。'},
 'loss-02':{1:'独立圆柱主体及基座清楚，未见前景遮挡，不属于仅细片段。'},
 'loss-03':{0:'独立柜体完整顶面、两侧与基座，无遮挡无图缘截断；无前面板。'},
 'loss-04':{0:'后柜只露顶面和上部窄带，前方同色柜体遮挡。',1:'中柜顶面和面板上缘细线可见，下部被前柜遮挡。',2:'近柜大顶面、主体与面板上部可见，下图缘截断。',5:'圆柱上部可见，下部被前方柜体遮挡；无法据框面积认证可见比例。'},
 'loss-05':{4:'近处变压器大主体、顶面与两柱可见，右侧和下侧严重图缘截断。'},
 'loss-06':{0:'后柜上部及面板局部可见，下部被前柜和变压器结构遮挡。',3:'近柜主体、面板、顶面与基座完整清楚，未见主要前景遮挡。'},
 'loss-07':{1:'中柜顶面及面板上缘可见，下部被近柜遮挡。',2:'近柜大顶面、主体与面板上部清楚，下图缘截断。',4:'右柜顶面、面板和基座清楚，左下局部被近柜遮挡。'},
 'loss-08':{8:'远柜右侧主体、顶面和基座可见，左部被变压器遮挡，尺度较小。'},
 'loss-09':{2:'左图缘柜体顶面、大侧面及基座可见，左边截断；无面板。',3:'圆柱顶面及上部可见，下部被变压器遮挡，框内柱属于前景变压器。'},
 'loss-10':{0:'近左柜体大顶面与侧面可见，左下图缘截断。',1:'独立圆柱主体及基座清楚，不是仅基座或极窄片段。'},
 'loss-11':{2:'队列中后柜顶部和侧面窄带可见，被前柜遮挡；框内含前景同色面。',9:'后方电容器主体上部与顶面可见，下部被变压器遮挡；框内三柱属于前景变压器，不能据此改类别。'},
 'loss-12':{7:'右图缘圆柱顶面、主体和基座部分可见，右边截断；左下与变压器相邻。'},
 'loss-13':{1:'右缘柜体侧面和基座局部可见，右边截断且与前柜重叠。',3:'最远柜体顶部与侧面窄条可见，前面队列遮挡；画面右缘限制内容。'},
 'loss-14':{3:'远柜顶部侧面窄条可见，同色队列遮挡，位于右图缘。',5:'后变压器三柱、顶面及主体上部可见，下部被前变压器遮挡。',8:'近处圆柱顶面、主体与完整基座清楚，无遮挡无截断。'},
}
FP={
 0:('gray_block_body','冷暗灰块体的主体、侧面大暗面板和基座，不是单独阴影。'),
 1:('gray_block_body','冷暗灰块体大侧面、暗面板与基座，右框缘贴近杆体。'),
 2:('gray_block_body','灰块体正面暗矩形面板、边框和下部基座。'),
 3:('gray_block_body','冷暗灰块体正面大暗面板与边框、基座。'),
 4:('mixed_structure','灰块体大面和基座，右部同时含竖杆及青色前景局部。'),
 5:('gray_block_body','正面灰块体暗面板及灰边框、下缘基座，非整帧其他杆体。'),
 6:('gray_block_body','灰块体侧背大面、侧面暗面板及基座。'),
 7:('gray_block_body','冷暗灰块体主体、侧面面板和基座，右边贴近杆体及少量青色局部。'),
 8:('mixed_structure','灰块体与前景粗杆共同占框，右下包括青色块体。'),
 9:('mixed_structure','灰块体大面与基座，右部有粗杆及青色前景遮挡。'),
 10:('gray_block_body','灰块体正面暗面板和边框，底缘贴近基座。'),
 11:('gray_block_body','冷暗灰块体正面暗矩形面板、边框与基座。'),
}

def validate_positive(e,decisions):
    expected={(f['frame_id'],x['seed'],x['truth_index']):(f,x) for f in e['frames'] for x in f['events']}
    keys=[(d['frame_id'],d['seed'],d['truth_index']) for d in decisions]
    if len(keys)!=len(set(keys)) or set(keys)!=set(expected):raise ValueError('Missing/duplicate positive review')
    for d in decisions:
        f,x=expected[d['frame_id'],d['seed'],d['truth_index']]
        if d['truth']!=x['truth'] or d['diagnosis']!=x['diagnosis'] or not d['reason'] or d['review_nature']!='AI辅助审核':raise ValueError('Review identity drift')
        for path,h in (('image_path','image_sha256'),('evidence_path','evidence_sha256')):
            if d[h]!=f[h] or file_sha256(f[path])!=d[h]:raise ValueError('Stale reviewed evidence')

def run():
    ep=OUT/'evidence.json';e=checked(ep);np=ROOT/'contrast/evaluation-v1/error-review-v1/evidence.json';n=checked(np)
    stamp=datetime.now(timezone.utc).isoformat();positive=[];negative=[]
    for f in e['frames']:
        for x in f['events']:
            positive.append(dict(frame_id=f['frame_id'],seed=x['seed'],truth_index=x['truth_index'],truth=x['truth'],diagnosis=x['diagnosis'],
                reason=NOTES[f['frame_id']][x['truth_index']],review_nature='AI辅助审核',reviewed_at=stamp,image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],
                decision='observed_content_and_limits_not_label_admission',pixel_visibility_certified=False))
    for f in n['frames']:
        for x in f['events']:
            category,reason=FP[int(x['event_id'].split('-')[-1])]
            negative.append(dict(event_id=x['event_id'],seed=x['seed'],prediction=x['prediction'],content_category=category,reason=reason,review_nature='AI辅助审核',reviewed_at=stamp,
                image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],asset_identity='unknown_visual_shape_not_asset_identity'))
    validate_positive(e,positive);validate_review(n,negative)
    return write_record(OUT/'review.json',dict(status='all_new_negative_fp_and_original_lighting_loss_events_reviewed',positive_decisions=positive,negative_decisions=negative,
        unique_positive_frames=15,unique_positive_truths=27,unique_negative_frames=7,loss_events=40,negative_events=12,
        limits='AI visual characterizations, no instance-mask certification or independent-scene claim. All gains/persistent hits/misses retained in evidence. No historical labels altered.',
        training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in (ep,np,Path(__file__))}))

if __name__=='__main__':print(run()['status'])
