"""Explicit current-scale new-miss observations; no training admission."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.freeze_reviewed_scale_control import OUT, prior
from scripts.vision.record_reviewed_endpoint_review import validate

NOTES = {
1: ('clear_body', '近处深青块体背面和完整基座清楚，斜向阴影可见，没有可见端子。'),
2: ('occluded', '后方深青块体顶部和上部可见，下方被蓝块及变压器遮挡；前景端子不能归给目标。'),
3: ('partially_occluded', '远处深青块体背面、顶面和右侧基座可见，左下被前景蓝块遮挡。'),
4: ('occluded', '后方圆柱顶面和上部主体可辨，下部被变压器遮挡，框内白色端子属于前景。'),
5: ('truncated', '左图缘蓝色柜体顶面、侧背面及基座可见，左侧截断，面板不可见。'),
6: ('truncated', '右图缘圆柱主体、部分椭圆顶面及基座可辨，右侧被画面截断。'),
7: ('clear_body', '完整柜体顶面、侧背面和基座清楚，没有明显前景遮挡，面板朝向不可见。'),
8: ('clear_body', '圆柱主体和方形基座清楚，无明显前景遮挡；顶面因视角近水平不明显。'),
9: ('truncated', '近处蓝柜在左下图缘截断，大片顶面和侧背面可见，底部不完整。'),
10: ('clear_body', '原始条件下完整柜体顶面、侧背面和基座可辨，无明显遮挡。'),
11: ('occluded_small', '远处圆柱只露上部窄段，下部被蓝色面板柜遮挡，框内含前景柜体顶部。'),
12: ('partially_occluded', '右侧柜体面板、顶部、侧面及基座清楚，左下被近柜顶面遮挡。'),
13: ('truncated', '下图缘近柜顶部和面板上部清楚，底部及面板下部越界，不是仅小片段。'),
14: ('clear_body', '左侧完整柜体顶面、大侧面、端部面板和基座可见，无明显前景遮挡。'),
15: ('truncated', '近处变压器大片顶面与白色端子可见，右侧和下方越出图缘。'),
16: ('occluded_small', '光照条件下后方圆柱仅露上部暗色片段，下部被蓝色柜体遮挡。'),
17: ('partially_occluded', '光照图右侧面板柜顶部和面板仍可辨，左下前景近柜遮住一角。'),
18: ('truncated', '光照图下图缘近柜顶部与面板上部可见，下部截断。'),
19: ('clear_body', '中近处柜体顶部、两侧背面及基座清楚，邻柜未明显遮住主体。'),
20: ('clear_body', '光照图深青块体背面与完整黑色基座清楚，表面斜影未遮去整体轮廓。'),
21: ('occluded', '原始图圆柱顶面及上部可辨，下部被变压器挡住，框内有前景端子。'),
22: ('clear_body', '原始图圆柱主体与方形底座轮廓清晰，没有明显前景遮挡，仍发生错类。'),
23: ('truncated', '原始图近柜被左下图缘截断，大侧背面和顶面仍可见。'),
24: ('truncated', '原始图右缘圆柱主体、顶面片段和底座可见，右侧截断。'),
25: ('partially_occluded', '后方深青目标顶面与背面清楚，左下部被近处蓝柜遮挡，右侧底座可见。'),
26: ('truncated_occluded', '右图缘多排同色柜体侧面和顶边重叠，目标局部可见；保留匹配竞争标记。'),
27: ('occluded', '后方变压器端子和顶部可辨，下半被近处大圆柱遮挡，框内混入前景主体。'),
28: ('partially_occluded', '远处变压器顶部及端子可见，下部被前排变压器遮挡，两个实例内容叠加。'),
29: ('truncated_occluded', '光照图右缘柜体群侧面、顶边重叠且截断，不能将正式分配竞争当成无候选。'),
30: ('partially_occluded', '光照图远变压器顶面及端子可见，下部与前排变压器重叠。'),
}

def run():
    ep = OUT/'evaluation/endpoint-review-v1/evidence.json'
    e = prior.read(ep)
    prior.verify(e)
    if {int(t['review_id'][1:]) for t in e['targets']} != set(NOTES):
        raise ValueError('Review population changed')
    decisions = []
    for t in e['targets']:
        state, reason = NOTES[int(t['review_id'][1:])]
        paths = [ep, Path(t['page']), Path(t['source']['image_path'])]
        decisions.append(dict(review_id=t['review_id'], visual_state=state, reason=reason,
            truth=t['truth'], review_nature='AI辅助审核', reviewed_at=datetime.now(timezone.utc).isoformat(),
            pixel_visibility_certified=False, training_admitted=False, promotable=False,
            instance_identity_scope=e['identity_scope'],
            evidence_hashes={str(p):prior.file_sha256(p) for p in paths}))
    expected = [t['review_id'] for t in e['targets']]
    validate(decisions, expected, 'review_id')
    dest = ep.parent/'explicit-review.json'
    if dest.exists():
        r=prior.read(dest); prior.verify(r); validate(r['decisions'],expected,'review_id'); return r
    transitions = {}
    for cell in sorted({t['cell'] for t in e['transitions']}):
        transitions[cell] = {}
        for variant in ('original', 'lighting'):
            transitions[cell][variant] = {c:dict(Counter(t['state'] for t in e['transitions']
                if t['cell']==cell and t['variant']==variant and t['truth']['class_name']==c))
                for c in ('transformer','switchgear','capacitor_bank','reactor')}
    return prior.frozen(dest, dict(status='new_miss_visual_review_complete_with_visibility_limits',
        decisions=decisions, new_miss_events=e['new_miss_events'], unique_targets=e['unique_targets'],
        miss_types=dict(Counter(x['miss']['reason'] for t in e['targets'] for x in t['events'])),
        matching_competition_events=sum(bool(x['miss']['formal_matching_competition']) for t in e['targets'] for x in t['events']),
        all_transition_counts=e['transition_counts'], transitions_by_cell_variant_class=transitions,
        selected_candidate=None, inputs={str(p):prior.file_sha256(p) for p in
            (ep, Path(__file__).resolve(), Path(__file__).with_name('record_reviewed_endpoint_review.py'))}))

if __name__=='__main__':
    print(run()['status'])
