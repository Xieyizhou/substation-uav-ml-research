"""Explicit decisions following visual inspection of N01-N24 and R01-R10."""
from datetime import datetime,timezone
from collections import Counter,defaultdict
from pathlib import Path
from scripts.vision.exposure_order_retention import OUT,read,save,file_sha256,verify,frozen

# Each entry is an explicit observation; no frame-theme inference or inherited review.
NEG={
'N01':('混合结构','左上天空占大部，底部围墙、右缘粗杆，非柜体。','left_boundary_a'),
'N02':('混合结构','右上天空、底部围墙地面及左缘杆体，未见柜状主体。','right_boundary_a'),
'N03':('混合结构','左上天空与底部围墙交界，框右缘贴近粗杆；与 N01 重叠区域。','left_boundary_a'),
'N04':('混合结构','天空、底部围墙和右缘杆体组合；不是画面中间蓝柜。','left_boundary_b'),
'N05':('混合结构','低光照下天空、围墙和右侧粗杆宽片段，无柜体。','left_boundary_a'),
'N06':('混合结构','低光照下框内天空、围墙、粗杆；与 N05 同一边界区域。','left_boundary_a'),
'N07':('混合结构','天空和斜向延伸的围墙及邻接地面网格，不含完整柜体。','diagonal_wall'),
'N08':('柜体','图像左缘截断的蓝绿色柜体背侧、顶部及黑色底座。','left_edge_blue'),
'N09':('混合结构','大块天空、左侧杆上部、围墙和右下灰柜局部；无法归因于单一结构。','right_boundary_c'),
'N10':('柜体','蓝绿色柜体正面深灰面板、右侧面及底座完整可见。','blue_front_c'),
'N11':('柜体','左侧较远蓝柜正面面板、右侧面及基座，框内主体明确。','blue_front_d'),
'N12':('杆体','主要可见两根竖杆及顶部横件，底部有围墙和蓝柜边缘；主要内容为杆体。','two_poles_d'),
'N13':('柜体','蓝柜宽侧面、窄正面面板、顶面及底座清楚可见。','blue_side_e'),
'N14':('混合结构','天空、右侧斜杆上部及底部围墙的组合，不是中间蓝柜。','left_boundary_e'),
'N15':('柜体','与 N10 同图同一蓝柜正面及右侧，面板和基座可见，分别核对了框。','blue_front_c'),
'N16':('柜体','与 N11 同图左侧蓝柜，面板、侧面和底座可见。','blue_front_d'),
'N17':('柜体','低光照中的灰柜正面深色面板及黑色基座；非杆体。','gray_front_d'),
'N18':('杆体','极窄右缘框主要覆盖截断斜杆、上部横件及天空，非柜体。','right_edge_pole_f'),
'N19':('杆体','同图右缘斜杆上部窄框，主要是杆体与天空，已单独检查。','right_edge_pole_f'),
'N20':('杆体','同图右缘细长框覆盖斜杆及天空，底部少量地面，非柜体。','right_edge_pole_f'),
'N21':('柜体','右缘截断灰柜无面板宽背侧及黑色基座，框下含地面。','gray_back_c'),
'N22':('柜体','中远处灰柜宽侧面及右侧深面板、基座，相邻杆仅在边缘。','gray_side_g'),
'N23':('柜体','与 N10/N15 同一图蓝柜正面面板、右侧与基座，seed27 框单独查看。','blue_front_c'),
'N24':('柜体','与 N13 同图蓝柜宽侧面及窄面板、顶面底座，seed27 框单独查看。','blue_side_e'),
}
REACT={
'R01':('none_apparent','not_truncated','圆柱主体和方形基座清晰可见，无明显前景遮挡。'),
'R02':('none_apparent','not_truncated','光照变体中圆柱和基座仍清楚；持续漏检不能解释为完全不可见。'),
'R03':('foreground_occlusion','not_truncated','远处圆柱上部露出，下部及基座被前景柜体遮挡，细节有限。'),
'R04':('foreground_occlusion','not_truncated','光照条件下仍为远处圆柱上部，前景柜遮下部；与原图分别查看。'),
'R05':('none_apparent','right_edge_truncated','右缘截断圆柱，可见宽弧面、顶面及部分基座；不是仅有窄边。'),
'R06':('none_apparent','right_edge_truncated','光照下右缘圆柱宽弧面和基座仍可见；右侧被画面截断。'),
'R07':('none_apparent','not_truncated','近景大圆柱完整主体、顶面及方形基座清楚，无严重遮挡。'),
'R08':('none_apparent','not_truncated','光照条件下近景圆柱完整顶面、主体和基座明显；深色外观但非不可见。'),
'R09':('foreground_occlusion','not_truncated','后方圆柱上部及顶面可见，下部被前景变压器遮挡，另有浅色附件投影在框内。'),
'R10':('foreground_occlusion','not_truncated','光照下后方圆柱上部可辨，下部和基座被前景设备遮挡，类别细节有限。'),
}

def validate_decisions(evidence,decisions):
    expected={e['event_id']:e for e in evidence['negative']+evidence['reactors']}
    ids=[d['event_id'] for d in decisions]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):raise ValueError('Missing / duplicate / unexpected review')
    for d in decisions:
        e=expected[d['event_id']]
        if not d.get('reason') or d.get('review_nature')!='AI辅助审核' or not d.get('reviewed_at'):raise ValueError('Review not explicit')
        if d.get('status')!='diagnosed' or d.get('content_category')=='无法确认':raise ValueError('Unresolved review')
        for kind in ('image','evidence'):
            if d.get(kind+'_sha256')!=e[kind+'_sha256'] or file_sha256(e[kind+'_path'])!=e[kind+'_sha256']:raise ValueError('Stale review')
        if d.get('source_event_identity')!=__import__('src.ml.artifacts',fromlist=['object_sha256']).object_sha256(e):raise ValueError('Prediction/evidence changed')

def main():
    path=OUT/'review.json'
    if path.exists():verify(read(path));return
    from src.ml.artifacts import object_sha256
    ep=OUT/'evidence.json';e=read(ep);verify(e);decisions=[]
    for event in e['negative']+e['reactors']:
        eid=event['event_id'];base=dict(event_id=eid,source_event_identity=object_sha256(event),
            image_sha256=event['image_sha256'],evidence_sha256=event['evidence_sha256'],
            review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),status='diagnosed')
        if eid in NEG:
            category,reason,group=NEG[eid];base.update(content_category=category,reason=reason,visual_structure_group=group)
        else:
            occlusion,truncation,reason=REACT[eid];base.update(content_category='圆柱及基座或受遮挡圆柱',reason=reason,
                occlusion=occlusion,truncation=truncation,component_identity='unknown',pixel_visibility_certified=False)
        decisions.append(base)
    validate_decisions(e,decisions)
    groups=defaultdict(list);by_image=defaultdict(list)
    for event in e['negative']:
        groups[NEG[event['event_id']][2]].append(event['event_id']);by_image[event['image_sha256']].append(event['event_id'])
    changes=Counter()
    for r in e['reactors']:
        for c in r['comparisons']:
            a,b=c['reference']['hit'],c['appearance']['hit']
            changes['retained' if a and b else 'gain' if b else 'loss' if a else 'persistent_miss']+=1
    frozen(path,dict(status='review_complete_no_identity_conflict_detected',decisions=decisions,
        negative_content_counts=dict(Counter(NEG[eid][0] for eid in NEG)),same_image_events=dict(by_image),
        same_visual_structure_events=dict(groups),structure_grouping_is_visual_not_simulator_certification=True,
        reactor_changes=dict(changes),scope='24 FP events/12 unique images; all 10 reactor image-instance contexts/30 seed comparisons',
        inputs={str(ep):file_sha256(ep),str(Path(__file__)):file_sha256(Path(__file__))}))
    print('REVIEW_COMPLETE',dict(changes),flush=True)

if __name__=='__main__':main()
