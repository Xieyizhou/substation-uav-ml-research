"""Explicit observations from twelve material pages and seven FP pages."""
from datetime import datetime,timezone
from pathlib import Path
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision.clear_context_lr_control import OUT,checked

NOTES=[
 ['三柱、主体及基座清楚，右图缘截断部分侧面。','灰色电容器主体顶面可见，下部被青色前景部分遮挡。'],
 ['近柜侧面顶面清楚，左侧及下侧图缘截断，无面板。','圆柱和基座完整，顶面轮廓可辨。'],
 ['后柜仅顶部窄带，同色前柜遮挡。','中柜上部与面板细线可见，下部被前柜遮挡。','近柜大顶面与面板细线可辨，下图缘截断。','变压器主体三柱及基座可辨，右缘截断。','右柜面板轮廓很淡，主体侧面顶面可辨，左下遮挡。','圆柱上部露出，底部被灰柜遮挡。'],
 ['远处三柱及主体可辨，右图缘截断。','后变压器三柱顶面可辨，前变压器遮挡下部。','近变压器主体三柱和基座清楚。','后灰块体顶部大面可辨，下部被青块体与近变压器遮挡。'],
 ['后柜面板及顶面局部，前柜和柱体遮挡。','中柜主体侧面顶面与淡面板可辨，左部遮挡。','柜体大侧面顶面、淡面板及基座可辨。','近柜主体顶面基座清楚，灰面板边界细淡。','变压器顶面和柱可见，右下严重截断。','右柜面板上部和顶面可见，下部被变压器遮挡。'],
 ['后柜上部局部可见，前方同色结构遮挡。','后柜侧面顶面可辨，右下被前柜遮挡。','后柜主体和基座完整，灰色侧面无面板。','后柜主体顶面可辨，左下被前柜遮挡。','变压器三柱和主体可辨，杆体遮挡且左图缘截断。','近变压器三柱顶面可辨，底部图缘截断。','中柜顶部及主体上部可辨，下部被近变压器遮挡。','右边圆柱局部、顶面和基座可见，右图缘截断。','左图缘柜体极窄片段，无足够独立形状。'],
 ['右图缘极窄柜体侧面和基座片段。','右图缘柜体侧面及基座局部，有前景遮挡。','后柜侧面顶面可见，前柜遮挡且右缘截断。','远柜顶部侧面局部，下部前景重叠。','远变压器三柱和主体可见，下部有前景灰块体遮挡。','后变压器三柱可辨，下部被前变压器遮挡。','中变压器三柱顶面可辨，下部被圆柱遮挡。','中右柜体主体顶面基座完整，正面无面板。','近电抗器圆柱、顶面和基座完整清楚。','最右图缘柜体窄灰片段，内容不足。','远电容器上部窄带，被杆体及变压器遮挡。'],
 ['左图缘只有变压器窄条，内容不足。','居中电容器组大主体和基座无遮挡，灰色面上阴影边界可见。'],
 ['近右柜顶部和主体大面可辨，下部图缘截断。','队列柜体顶面和上部可见，下部被前柜遮挡。','后柜仅顶面与窄侧带可见，同色前柜遮挡。','最远柜体顶面与窄侧带可见。','远变压器三柱主体基座可辨。','中后变压器三柱主体可辨，粗杆遮挡右部。','左近变压器三柱主体可辨，左图缘截断。','近中柜完整顶面大侧面及基座，无面板。','远柜侧面顶面局部，被变压器遮挡。','远电容器顶面上部可见，下部被变压器遮挡。'],
 ['完整独立灰柜：顶面、两侧面和基座清楚，无遮挡无截断；此视角看不到面板。'],
 ['右变压器三柱顶面主体可辨，粗杆遮挡中央。','左近变压器顶部三柱可辨，左下严重截断。','远电容器主体和顶面可辨，左下青色前景部分遮挡。'],
 ['左图缘后柜上部片段，被前柜遮挡。','变压器三柱主体基座清楚。','左柜大侧面顶面基座可辨，左图缘截断。','电抗器圆柱上部可见，下部被变压器遮挡。'],
]
FP_NOTES={
0:'灰块体大面、侧面暗面板和基座，右缘贴近杆体。',1:'冷暗灰块体大面与侧面暗面板、基座。',
2:'灰块体被前景粗杆遮挡，右下含青色块体局部。',3:'灰块体大面和基座，右下青色遮挡且右部竖杆。',
4:'灰块体正面大暗矩形面板和外框；几乎不含其他物体。',5:'冷暗灰块体正面暗面板和边框及基座。',
6:'冷暗灰块体正面暗面板、大侧面和基座，非单纯阴影。',7:'灰块体与前景粗杆、右下青色局部重叠。',
8:'灰色块体大暗面板、边框和少量底座。',9:'灰块体主体、侧面暗面板和底座，右缘接近杆体。',
10:'冷暗灰块体主体、侧面暗面板和底座。',11:'灰块体与粗杆及右下青色局部共同入框。',
12:'灰块体大面和基座，右侧含粗杆及青色前景局部。',13:'灰块体正面暗矩形面板、外框和底座。',
}


def validate_material(evidence,decisions):
    expected={(f['frame_id'],i):(f,t) for f in evidence['frames'] for i,t in enumerate(f['truth'])}
    if len(decisions)!=len(expected) or {(d['frame_id'],d['truth_index']) for d in decisions}!=set(expected):raise ValueError('Missing/duplicate decisions')
    for d in decisions:
        f,t=expected[d['frame_id'],d['truth_index']]
        if d['truth']!=t or not d['reason'] or d['review_nature']!='AI辅助审核':raise ValueError('Invalid decision')
        for path,key in [('image_path','image_sha256'),('evidence_path','evidence_sha256')]:
            if d[key]!=f[key] or file_sha256(f[path])!=d[key]:raise ValueError('Stale review')


def run():
    ep=OUT/'verification-v1/evidence.json';e=checked(ep);fp_path=OUT/'evaluation-v1/error-review-v1/evidence.json';fp=checked(fp_path)
    stamp=datetime.now(timezone.utc).isoformat();decisions=[];fps=[]
    for i,f in enumerate(e['frames']):
        if len(NOTES[i])!=len(f['truth']):raise ValueError('Authored observation coverage')
        for idx,t in enumerate(f['truth']):
            decisions.append(dict(frame_id=f['frame_id'],truth_index=idx,truth=t,pair_id=f['pair_id'],reason=NOTES[i][idx],
                review_nature='AI辅助审核',reviewed_at=stamp,image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],
                pixel_visibility_certified=False,decision='visual_characterization_not_training_admission'))
    validate_material(e,decisions)
    for f in fp['frames']:
        for event in f['events']:
            fps.append(dict(event_id=event['event_id'],prediction=event['prediction'],image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],
                seed=event['seed'],reason=FP_NOTES[int(event['event_id'].split('-')[-1])],review_nature='AI辅助审核',reviewed_at=stamp,
                asset_identity='unknown_visual_shape_not_asset_identity'))
    from scripts.vision.audit_reviewed_order_results import validate_review
    validate_review(fp,fps)
    target=OUT/'verification-v1/review.json'
    if target.exists():
        existing=checked(target);validate_material(e,existing['material_decisions']);validate_review(fp,existing['fp_decisions']);return existing
    return write_record(target,dict(status='material_all_truth_and_negative_fp_visual_review_complete',material_decisions=decisions,fp_decisions=fps,
        limits='60 material-frame truths in 12 poses; seed repetition is not independence. No mask or asset certification, no label edits, no new training.',
        training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in (ep,fp_path,Path(__file__))}))


if __name__=='__main__':print(run()['status'])
