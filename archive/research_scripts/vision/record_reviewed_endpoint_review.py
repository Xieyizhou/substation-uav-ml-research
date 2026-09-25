"""Explicit observations authored after inspecting every current review page."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.build_reviewed_endpoint_errors import OUT,DEST,prior

# Each numbered observation refers only to its own displayed full image and crop.
NOTES={
1:('occluded','后方深青块体上部可见；前方蓝块及变压器顶面遮住下部，框内白色端子不能全归给目标。'),
2:('occluded','圆柱上部及椭圆顶面可辨，下部被前方变压器遮挡，框含前景端子。'),
3:('truncated_occluded','左图缘仅见后排蓝色顶部和侧面片段，前排同色柜体遮挡，类别内容有限。'),
4:('truncated','左图缘截断柜体，顶部、大片无面板侧背面和基座可见，面板不可见。'),
5:('clear_body','主体、顶面、侧背面及基座完整清楚，当前视角不见面板。'),
6:('clear_body','灰色圆柱主体及基座轮廓清楚，无明显前景遮挡，不能将错类归咎于仅剩基座。'),
7:('truncated','左下图缘截断近处柜体，可见大块顶面与侧背面，底部不完整。'),
8:('clear_body','原始条件下完整蓝色柜体顶面、侧背面及基座清楚，面板朝向不可见。'),
9:('partially_occluded','右侧目标面板、顶面和侧面可辨，左下被近处柜体遮挡。'),
10:('truncated','近处柜体下部越出图缘，顶面与部分面板仍清楚，不是小目标。'),
11:('occluded','中排柜体主要见顶面、窄前面带；近柜顶面占框下半，内容归属有局部歧义。'),
12:('occluded','远排柜体顶部与窄侧面可见，多个同色柜体在框内叠加，完整轮廓不清。'),
13:('partially_occluded','后方柜体顶面和深色面板可辨，下部被变压器顶面遮挡，框内含前景端子。'),
14:('clear_body','左侧柜体主体、侧面、端部面板与基座完整可辨，无明显遮挡。'),
15:('partially_occluded','深青目标背面及顶部可见，下部左侧被蓝色近柜遮挡，没有可见端子。'),
16:('clear_body','光照条件下左侧柜体侧面、端部面板和基座仍清楚，无明显遮挡。'),
17:('partially_occluded','右侧深色面板及顶部可辨，左下角被前景柜体遮挡。'),
18:('truncated','光照图近处柜体底部越界，顶部和面板上部清楚。'),
19:('occluded','中间柜体被近柜遮挡，大部分框内是近柜顶面，仅余远柜上部条带。'),
20:('occluded','远排蓝柜与相邻同色柜体重叠，可见顶部条带，独立外形有限。'),
21:('truncated','右下近柜同时接触右、下图缘，仍见较大顶部及侧背面。'),
22:('occluded_small','远处柜体左部被变压器挡住，仅见右侧蓝色块体、顶部和部分基座。'),
23:('clear_body','近处电容器标签对应的大块背面与基座清楚；无端子可见，表面有斜向阴影。'),
24:('truncated_occluded','原始图左缘后排柜体只有顶面与侧面窄片段，并被前排柜体遮挡。'),
25:('truncated','原始图左缘柜体顶面、大片侧背面及基座可见，左侧越界。'),
26:('clear_body','原始条件圆柱及基座清楚，主体未被前景遮挡，存在清晰目标退化。'),
27:('truncated','左下近处柜体顶面与侧背面清楚，但整体被左下图缘截断。'),
28:('occluded_small','远处电容器目标只见后方深色块体上部；框内前景变压器端子和顶面占比较大。'),
29:('truncated','原始图右下近柜顶面和侧面较大，右下边界截断其整体。'),
30:('occluded_small','原始图远柜受变压器左侧遮挡，剩余蓝色侧面、顶部和小段底座。'),
31:('truncated','右图缘圆柱主体、部分椭圆顶面和底座可辨，右侧明显截断。'),
32:('partially_occluded','中部柜体顶面与部分侧面可见，下方被近处变压器遮挡，后方另有同色柜。'),
33:('occluded_small','远处深色块体受杆体及前方变压器遮挡，框内不能把前景内容算作目标结构。'),
34:('clear_body','近处完整圆柱、椭圆顶面和方形基座清楚，无遮挡，两个单元降为低置信度。'),
35:('truncated_occluded','右缘远排多个柜体侧面重叠，目标边界不直观；保留已检测到的匹配竞争。'),
36:('truncated_occluded','右缘中排柜体顶边、侧面和基座片段可见，重叠同类结构导致分配竞争。'),
37:('truncated','右缘近柜可见较大侧面及基座，右侧越界且邻近同色柜体。'),
38:('occluded','后方变压器顶部、端子可辨，下半被近处圆柱大幅遮挡，定位需区分前景。'),
39:('partially_occluded','远处变压器顶面及端子可辨，前方变压器遮挡下部，框含两实例交叠。'),
40:('truncated_occluded','光照图右缘远排柜体轮廓重叠且截断，正式匹配竞争应与保留预测不足并列。'),
41:('truncated_occluded','光照图右缘中排柜体被邻柜重叠，侧面与窄基座片段可见，多事件有匹配竞争。'),
42:('truncated','光照图右缘近柜侧面及基座仍可见，右侧图缘截断；不能声称完全不可见。'),
43:('occluded','光照图变压器顶部和端子可辨，下部被圆柱遮挡，框内前景显著。'),
44:('partially_occluded','光照图远变压器顶面与端子可见，下半与前方变压器重叠。')}

FP={
'F01': [('mixed_structure','大框覆盖天空、两根杆体及灰蓝块体边缘，非单一设备轮廓。'),('block_with_panel','灰色块体背侧面为主，叠加杆体及近处蓝块。'),('cabinet_like','蓝色柜状体顶面、侧面、深色面板与基座清晰。')],
'F02': [('mixed_structure','灰色块体被前景粗杆遮挡，框内含蓝色块体一角。')]*4,
'F03': [('block_with_panel','灰色块体正面具有深色矩形面板，下方基座可见；同一外形被预测为不同设备类。')]*4,
'F04': [('block_with_panel','灰色大块背侧面，右侧被杆体和蓝块叠加。'),('cabinet_like','蓝色柜状体带深色侧面板及黑色基座，轮廓清晰。')],
'F05': [('mixed_structure','右图缘灰色面板块体与杆体、蓝色前景块体重叠，且被图缘截断。')],
'F06': [('mixed_structure','灰色块体和前景粗杆重叠，框内含蓝色块体边角。'),('mixed_structure','灰色块体与前景杆体、蓝块边角形成复合轮廓。'),('mixed_structure','较大框同时覆盖灰块、蓝块及粗杆，不是单个清晰圆柱。'),('mixed_structure','灰块主体被粗杆部分遮挡，前景蓝块占框右下。')],
'F07': [('block_with_panel','灰色块体大侧背面与侧方深色面板可见，底部有基座，边缘有杆体。')]*2,
'F08': [('block_with_panel','灰色块体正面深色大面板与窄边框，三事件定位于同一面板结构。')]*3,
'F09': [('block_with_panel','灰色块体大侧面、右侧深色面板和基座清楚，三事件类别不一致。')]*3}

def validate(decisions,expected,key):
    ids=[d[key] for d in decisions]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):raise ValueError('Missing or duplicate review')
    for d in decisions:
        if not d['reason'] or d['review_nature']!='AI辅助审核':raise ValueError('Invalid decision')
        for p,h in d['evidence_hashes'].items():
            if prior.file_sha256(p)!=h:raise ValueError('Stale evidence')

def run():
    ep=OUT/'evidence.json';fp=DEST/'evaluation/false-positive-review-v1/evidence.json'
    e,f=[prior.read(p) for p in (ep,fp)]
    for r in (e,f):prior.verify(r)
    now=datetime.now(timezone.utc).isoformat();decisions=[];false=[]
    if len(e['targets'])!=44 or len(NOTES)!=44:raise ValueError('Review population changed')
    for t in e['targets']:
        n=int(t['review_id'][1:]);state,reason=NOTES[n];paths=[ep,Path(t['page']),Path(t['source']['image_path'])]
        decisions.append(dict(review_id=t['review_id'],visual_state=state,reason=reason,review_nature='AI辅助审核',reviewed_at=now,
            pixel_visibility_certified=False,training_admitted=False,promotable=False,
            instance_identity_scope=e['identity_scope'],evidence_hashes={str(p):prior.file_sha256(p) for p in paths}))
    for g in f['images']:
        notes=FP[g['image_id']]
        if len(notes)!=len(g['events']):raise ValueError('FP population changed')
        for event,(state,reason) in zip(g['events'],notes):
            paths=[fp,Path(event['page_path']),Path(g['source']['image_path'])]
            false.append(dict(event_id=event['event_id'],image_id=g['image_id'],visual_structure=state,reason=reason,
                simulation_asset_identity='unknown_not_resolved_by_visual_review',review_nature='AI辅助审核',reviewed_at=now,
                same_image_seeds=sorted({x['seed'] for x in g['events']}),
                evidence_hashes={str(p):prior.file_sha256(p) for p in paths}))
    validate(decisions,[t['review_id'] for t in e['targets']],'review_id')
    validate(false,[x['event_id'] for g in f['images'] for x in g['events']],'event_id')
    dest=OUT/'review.json'
    if dest.exists():prior.verify(prior.read(dest));return prior.read(dest)
    return prior.frozen(dest,dict(status='explicit_error_visual_review_recorded_with_limits',new_miss_reviews=decisions,false_positive_reviews=false,
        new_miss_events=e['new_miss_events'],unique_target_frames=e['unique_targets'],false_positive_events=len(false),unique_negative_images=f['unique_images'],
        miss_types=dict(Counter(x['miss']['reason'] for t in e['targets'] for x in t['events'])),
        matching_competition_events=sum(bool(x['miss']['formal_matching_competition']) for t in e['targets'] for x in t['events']),
        new_miss_classes=dict(Counter(x['truth']['class_name'] for t in e['targets'] for x in t['events'])),
        visual_target_states=dict(Counter(d['visual_state'] for d in decisions)),false_positive_structures=dict(Counter(d['visual_structure'] for d in false)),
        transition_counts=e['transition_counts'],limits=['Visual review is not mask certification or simulation asset identification.',
            'All gains and persistent states retained in evidence; only new misses individually reviewed here.',
            'Multiple models/seeds and paired conditions do not increase independent scenes.'],
        candidate_passed=False,inputs={str(p):prior.file_sha256(p) for p in (ep,fp,Path(__file__).resolve())}))

if __name__=='__main__':
    r=run();print({k:r[k] for k in ('status','miss_types','matching_competition_events','new_miss_classes','visual_target_states','false_positive_structures')})
