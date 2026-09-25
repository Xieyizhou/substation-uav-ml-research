"""Transcribe the explicitly inspected 31 evidence pages; never infer approval."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from scripts.vision.train_reviewed_hold import OUT, prior
from scripts.vision.finalize_reviewed_hold import validate_errors

EVIDENCE_ID = 'c55590c6cf47e6071730a180fe2925650f16f7dc70f16722b8c9ed12531f32cb'
NOTES = {
 'FP01': ('柜体', '蓝青色普通柜状块体的宽侧背面、顶部和基座；所见面无面板。邻近杆体主要在框外。这里只描述视觉外形，不改写资产身份。'),
 'FP02': ('柜体', '两 seed 的框覆盖同一蓝色柜体：深色矩形前面板、顶部、侧面及底座清楚；属于同图同结构重复误检。'),
 'FP03': ('柜体', '主要为蓝色柜体前面板、右侧面和基座，右上少量背景建筑边缘不改变框内主体分类。'),
 'FP04': ('混合结构', '两 seed 框同时包含灰色建筑块体立面、前景深色竖杆以及右下部分蓝色柜体；不能仅按帧主题归为杆体或建筑。'),
 'FP05': ('柜体', '冷暗光照下左缘蓝色柜体，面板、顶部、主体及基座可见，有轻微图缘截断。'),
 'FP06': ('柜体', '正常光照下蓝色面板柜体、顶部及基座；与 FP05 为同 view 的光照变体，不是新增独立场景。'),
 'LOSS01': ('清晰主体', '圆柱曲面主体和基座面积较大，无明显前景遮挡。'),
 'LOSS02': ('局部遮挡', '中景柜体前面板、顶部和右侧可辨，左下被前景柜体遮挡。'),
 'LOSS03': ('边界截断', '近景柜体下缘被画面截断，宽顶部和部分前面板仍可辨；操作性诊断为定位不足。'),
 'LOSS04': ('内容有限', '远处柜体被近处柜体重叠遮挡，仅上部和窄条内容可见，实例内容可辨识程度有限；不据此修订标签。'),
 'LOSS05': ('局部遮挡', '原始光照的中景柜体前面板和顶部清楚，左下被前景柜体遮挡；各 seed 的操作性原因分别保留。'),
 'LOSS06': ('边界截断', '近景变压器宽主体、顶部和三个套管可辨，底部被图缘截断。'),
 'LOSS07': ('前景遮挡', '后方变压器上部和三个小套管可见，下部被大幅前景变压器遮挡；裁剪中的前景套管不归给后方目标。'),
 'LOSS08': ('前景遮挡', '远处柜体的局部面板、顶部和右侧可见，左侧及底部遮挡；裁剪包含的前景套管不是柜体组件。'),
 'LOSS09': ('遮挡及截断', '左缘变压器宽主体和顶部套管可见，前景竖杆横跨主体，左侧图缘截断。'),
 'LOSS10': ('边界截断', '右缘电抗器圆柱主体、部分顶面和基座可见，右侧截断；未发现合格保留预测不等于网络未产生候选。'),
 'LOSS11': ('边界截断', '右缘圆柱曲面和基座清楚，右侧截断；两个 seed 均为同类低置信度，不是两个独立实例。'),
 'LOSS12': ('清晰主体', '中景蓝色柜体宽侧背面、顶部和基座完整可见；当前所见面没有前面板。'),
 'LOSS13': ('局部遮挡', '右上柜体列重叠，目标上缘和侧面窄条可见，部分被更近柜体遮挡；不能把整列都当作目标。'),
 'LOSS14': ('局部遮挡', '原始条件右上柜体列重叠，目标顶部及侧面局部可见，近柜遮挡其余区域。'),
 'LOSS15': ('极端截断', '最右图缘仅极窄蓝色主体侧片与基座片段，裁剪还含相邻柜体；内容和组件归属可辨识程度有限，像素级归属未认证。'),
 'LOSS16': ('清晰主体', '前景电抗器完整顶面、圆柱主体和基座清晰，无明显遮挡或图缘截断；支持非全部错误都由内容不足解释。'),
 'LOSS17': ('清晰主体', '电容器组真值对应宽蓝青色块体表面和底座，面板在此视角不可见，斜阴影跨过主体；视觉外形不用于重定来源类别。'),
 'LOSS18': ('边界截断', '左缘变压器主体、顶部三个套管及底座可辨，左缘截断，未见主要前景遮挡。'),
 'LOSS19': ('清晰主体', '中央柜状目标的宽主体、顶面、侧面及底座清晰，面板未处于所见面，无明显遮挡。'),
 'LOSS20': ('前景遮挡', '远处目标左部被前景变压器遮挡，右部蓝色主体、顶部及基座片段可见；尺度较小。'),
 'LOSS21': ('边界截断', '右下近柜体顶部和宽主体可见，下缘被图像截断，裁剪也包含上方相邻柜体。'),
 'LOSS22': ('边界截断', '左下近景变压器顶部三个套管和宽主体清楚，左缘、底缘截断。'),
 'LOSS23': ('局部遮挡', '中远处电容器组真值对应宽块体、顶面和底座，左下被前方蓝色柜体遮挡，主要表面仍可见。'),
 'LOSS24': ('清晰主体', '中景变压器宽主体、顶部三个套管和基座清晰，无明显遮挡或截断；不是仅局部难辨内容。'),
 'LOSS25': ('遮挡及截断', '左缘后方柜体仅上部窄条和局部侧面可见，受左图缘截断及前柜遮挡；裁剪包含前柜内容，组件精确归属未认证。'),
}

def main():
    ep = OUT/'error-review/evidence.json'
    e = prior.read(ep); prior.verify(e)
    if e['identity'] != EVIDENCE_ID or set(NOTES) != {r['event_id'] for r in e['events']}:
        raise ValueError('Explicit notes do not cover this exact evidence')
    now = datetime.now(timezone.utc).isoformat()
    decisions=[]; inputs={str(ep):prior.file_sha256(ep), str(Path(__file__).resolve()):prior.file_sha256(__file__)}
    for event in e['events']:
        category, note=NOTES[event['event_id']]
        for path, sha in [(event['page'],event['page_sha256']), (event['source']['image_path'],event['source']['image_sha256'])]+[(c['path'],c['sha256']) for c in event['crops']]:
            if prior.file_sha256(path)!=sha: raise ValueError('Stale inspected image')
            inputs[path]=sha
        records=event['predictions'] if event['kind']=='FP' else event['loss']['seeds']
        for n, record in enumerate(records):
            reason=note
            if event['kind']=='LOSS':reason += ' 固定低阈值操作性分类：'+record['miss']['reason']+'；仅为图像内容审核，不认证原帧像素级实例可见性。'
            decisions.append(dict(decision_id=f'{event["event_id"]}-{n}', event_id=event['event_id'],
                kind=event['kind'], seed=record['seed'], source=event['source'],
                prediction_or_diagnostic=record,
                prediction_or_diagnostic_sha256=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest(),
                truth=event['loss']['truth'] if event['kind']=='LOSS' else None,
                page_sha256=event['page_sha256'], crops=event['crops'], content_category=category,
                review_nature='AI-assisted', reviewed_at=now, reason=reason, status='reviewed',
                pixel_visibility_certified=False, training_admitted=False,promotable=False))
    r=dict(evidence_identity=e['identity'],decisions=decisions,inputs=inputs,
        meaning='Reviewed means error evidence described, never training approval or visibility certification.',
        unknown_component_attribution_events=['LOSS15','LOSS25'],
        status='explicit_error_review_complete')
    validate_errors(e,r)
    prior.frozen(OUT/'error-review/decisions.json',r)
    print('Explicit decisions:',len(decisions))

if __name__=='__main__':main()
