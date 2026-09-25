"""Explicit visual observations; never certify masks or change labels."""
from datetime import datetime,timezone
from pathlib import Path
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record
from scripts.vision.freeze_clear_context_training import OUT
from scripts.vision.prepare_clear_context_increment import checked

OBS={
 (0,2):'大型变压器顶部三柱和主体两面清楚，底座接近下图缘。',
 (1,4):'左边变压器被图缘截断，粗杆挡住部分主体；仍有顶部柱和主体大面。',
 (2,0):'后排柜体仅上部窄条及顶部可辨，前方同色柜体遮挡，不能认证独立像素归属。',
 (2,1):'中排柜体上部、窄面板线可见，下部被前排柜体遮挡；框内大量为遮挡物。',
 (2,4):'右侧柜体面板、顶部和侧面清楚，左下被前景同色结构部分遮挡。',
 (2,5):'后方电抗器只有圆柱上部可见，底部被柜体遮挡。',
 (3,4):'近处变压器顶部和柱体清楚，右侧及下侧严重图缘截断。',
 (4,0):'后排柜体面板及顶部局部可见，前方柜体与柱体遮挡部分轮廓。',
 (4,4):'变压器顶部及柱体可辨，主体右侧和下侧超出图缘。',
 (5,0):'后排柜体仅上部窄条、顶部局部；同色前景遮挡，不能认证独立像素归属。',
 (5,1):'中排柜体上部和面板细线可见，下部被前排同色柜体覆盖。',
 (5,4):'右侧柜体完整面板和大部分主体可辨，左下被前景遮挡。',
 (5,5):'电抗器圆柱上部露出，底部被柜体遮挡，无完整圆柱证据。',
 (6,1):'右侧队列柜体上部可见，下部被前方柜体遮挡，右边贴近场景边缘。',
 (6,2):'队列中后部柜体仅顶部和窄侧面带，框内包含前方柜体。',
 (6,3):'最远柜体顶部及窄侧带可见，与同色前方柜体重叠。',
 (6,8):'远处柜体侧面和顶面可辨，左边被变压器遮挡。',
 (7,2):'左侧柜体无面板侧面、顶部及底座可见，左边被图缘截断。',
 (7,3):'电抗器顶面和上部圆柱可见，下部被变压器遮挡。',
 (8,0):'近处柜体顶面和无面板大侧面清楚，左侧及下侧被图缘截断。',
 (9,1):'前后排列柜体的中间上部可见，下部被同色前景柜体遮挡。',
 (9,2):'后部柜体顶部与窄侧面带可见，框内含前方同色结构。',
 (9,3):'远处柜体顶部及窄侧面可见，独立主体被遮挡。',
 (9,8):'远处柜体侧面和顶部局部可辨，左边被变压器遮挡。',
 (9,9):'后方电容器组顶面和上部大面可见，下部被变压器遮挡。',
 (10,4):'左边变压器图缘截断且被粗杆遮挡，顶部柱及大侧面仍可辨。',
 (11,1):'右图缘柜体侧面与底座局部，右侧截断并与前景柜体重叠。',
 (11,2):'右后柜体顶部和侧面带可见，图缘截断且有同色前景遮挡。',
 (11,6):'后方变压器顶部三柱可辨，下部大范围被圆柱电抗器遮挡。',
 (11,8):'近处电抗器完整顶面、圆柱主体与基座清楚，无明显主体遮挡。',
 (11,10):'远处电容器组顶面及上部窄带，左侧有杆体且下部被变压器遮挡。',
 (12,1):'右图缘柜体侧面与基座局部，右侧截断并有前景遮挡。',
 (12,5):'后方变压器仅顶部三柱及窄主体上部可辨，前景变压器遮挡下部。',
 (12,6):'变压器顶部三柱清楚，主体下部被近处圆柱遮挡。',
 (12,8):'电抗器完整圆柱、顶面和基座清楚，光照偏冷暗但轮廓可辨。',
}


def run():
    root=OUT/'evaluation-v1/positive-review-v1';ep=root/'evidence.json';e=checked(ep)
    expected={(int(f['frame_id'].split('-')[-1]),x['truth_index']) for f in e['frames'] for x in f['events']}
    if expected!=set(OBS):raise ValueError('Visual observation coverage mismatch')
    decisions=[];stamp=datetime.now(timezone.utc).isoformat()
    for f in e['frames']:
        if file_sha256(f['image_path'])!=f['image_sha256'] or file_sha256(f['evidence_path'])!=f['evidence_sha256']:raise ValueError('Stale evidence')
        for event in f['events']:
            key=(int(f['frame_id'].split('-')[-1]),event['truth_index'])
            decisions.append(dict(frame_id=f['frame_id'],seed=event['seed'],truth_index=event['truth_index'],truth=event['truth'],
                diagnosis=event['diagnosis'],observation=OBS[key],review_nature='AI辅助审核',reviewed_at=stamp,
                image_sha256=f['image_sha256'],evidence_sha256=f['evidence_sha256'],
                pixel_visibility_certified=False,decision='visual_error_characterization_not_label_admission'))
    target=root/'review.json'
    if target.exists():return checked(target)
    return write_record(target,dict(status='all_original_lighting_new_loss_events_characterized',decisions=decisions,
        unique_images=len(e['frames']),unique_image_truth_pairs=len(expected),
        limits='No instance masks. Frame-local truth identities only; no cross-world runtime ID inference. Partial/occluded evidence is not declared clear. All gains and persistent states retained in evidence.',
        training_admitted=False,promotable=False,inputs={str(p.resolve()):file_sha256(p) for p in (ep,Path(__file__))}))


if __name__=='__main__':print(len(run()['decisions']))
