"""Explicit observations of all 293 switchgear training boxes; preserve unknowns."""
import sys
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.vision.build_switchgear_condition_review import OUT,read,save,file_sha256,verify_tree
from scripts.vision.build_switchgear_full_context import IDS
# Each code is an explicit visual observation in manifest order, not inferred from labels.
OBS_ROWS=[
    "AN AN AN AN AN AN AP AN AP AN AN AN",
    "AP AP AN AP AN AN AN AN AP AN AP AP",
    "AP AP AN AN AN AN AN AN AN AN AN AP",
    "AP AN AP AN AP AN AN AP AN AP AH AP",
    "UU DN AP AP AH UU UU AP AN AH DN AP",
    "DP AN DP DP AN DN DN AN AN AN AP AH",
    "AH AN AH AH UP AH AH AN AH AH AH AN",
    "DP DP AH DN AH AN AP AN AP AN UH AH",
    "DN AN AN AP AP AN AN AP AH AP AN AH",
    "AP AH UH AP DN AH AH UH UH AP AN AP",
    "AH AP AN UU DP AN DN DP DN DN UH AN",
    "DP AP AH AN AN AH AP AN AH AN AP DP",
    "DN AN AN UU DP DP DP DH AN DP AP AN",
    "AN AN UU AN AN AH AP AH AP AP AH AP",
    "AP AH DP AN DN AN UH AH DP UP AP AN",
    "AP AP AH DP AP DP UH AH AP AP AN DP",
    "AN DN DN DN LN LN DN DN DP DN DN LP",
    "LN LN DP DN DN AP AH AH AP AH AH AP",
    "AH AH AP AH AN AP AH AP AH AN AP AH",
    "AP AH AN AP AH AP AH UH AH AH AP AH",
    "UH AH AH AP AH UH AH AH AP AP AP DP",
    "AN DN LP AN LN DP AN DN AP AP AP AP",
    "AP AP AP AH UH UH AH AP AH UH UH AH",
    "AP AH UH UH AH DP DN UH DP LP LN UH",
    "LP DP DN UH DP"
]
PANEL={'A':'not_visible_on_exposed_faces','D':'visible_distinct_contrast','L':'visible_low_contrast','U':'unknown'}
OCCLUSION={'N':'no_obvious_foreground_occlusion','P':'partial_foreground_occlusion','H':'heavy_foreground_occlusion','U':'unknown'}
REASONS_P={'A':'可见箱体平面或顶部，未见可辨识的前面板；不推断被遮挡或背向区域的面板状态。','D':'可见深色矩形面板，与周围外壳有清楚视觉对比。','L':'可见面板边界或轮廓，但面板与外壳呈接近灰色，视觉对比较弱。','U':'当前可见部分不足以可靠判断前面板状态，保留unknown。'}
REASONS_O={'N':'未见明显前景遮挡；图像边缘截断另行记录。','P':'框内目标被前景柜体、立杆或其他设备部分遮挡。','H':'前景结构遮住目标较大部分，仅能查看其剩余区域；不报告精确遮挡比例。','U':'极窄边缘或内容不足，无法可靠确认目标可见区域和遮挡关系。'}
SPECIAL={
49:'全图复核：最右图像边缘框主要呈地面/边界细条，无法可靠确认设备可见像素；需核对实例投影与截断。',
54:'全图复核：最左边缘框主要呈地面与边界，无法可靠确认设备；需核对实例投影与截断。',
148:'全图复核：最右边缘框主要呈地面，无法可靠确认设备；需核对实例投影与截断。',
55:'全图复核：最右边界的设备区域仅剩窄条，无法可靠归属其面板。',
124:'全图复核：最右边界紧邻变压器的窄框，设备可见部分不足。',
159:'全图复核：最左边缘仅有设备片段，无法可靠判断对应面板。',
59:'补查全图后确认近景柜体侧面及左侧面板属于同一可见目标。',
65:'补查全图后，目标可见部分为左边缘蓝色平面，邻近暗面板不能归给该框。',
94:'补查全图后，目标为右下截断近景箱体，可见顶部/平面，面板未显露。',
157:'补查全图后，可见目标位于画面底部，主要留下近景顶部；不是完整设备视图。',
160:'补查全图后，可见左下边缘近景箱体平面，面板未显露。',
161:'补查全图后，目标位于右边缘，可见蓝色侧面片段，面板未显露。'
}

def validate(manifest,decisions,require_resolved=False):
    expected={r['review_id']:r for r in manifest['items']}
    if len(decisions)!=len(expected) or {d['review_id'] for d in decisions}!=set(expected):raise ValueError('Missing or duplicate condition review')
    for d in decisions:
        row=expected[d['review_id']]
        if d['review_nature']!='AI-assisted' or not d['reason'] or not d['reviewed_at']:raise ValueError('Invalid review nature')
        if d['bbox_xyxy']!=row['bbox_xyxy']:raise ValueError('Changed target box')
        unknown=d['panel_condition']=='unknown' or d['occlusion_condition']=='unknown'
        if d['decision']!=('held_condition_unknown' if unknown else 'condition_recorded'):raise ValueError('Unknown silently accepted')
        if require_resolved and unknown:raise ValueError('Condition audit unresolved')
        for kind in ('image','label','evidence'):
            if d[kind+'_sha256']!=row[kind+'_sha256'] or file_sha256(row[kind+'_path'])!=d[kind+'_sha256']:raise ValueError('Stale evidence')

def main():
    mp=OUT/'manifest.json';verify_tree(mp);verify_tree(OUT/'full-context.json');m=read(mp)
    codes=' '.join(OBS_ROWS).split()
    if len(codes)!=len(m['items']):raise ValueError('Explicit observations incomplete')
    decisions=[]
    for row,code in zip(m['items'],codes):
        pc,oc=code;i=int(row['review_id'][1:]);x1,y1,x2,y2=row['bbox_xyxy'];w,h=row['original_image_size']
        edge=x1<=1 or y1<=1 or x2>=w-1 or y2>=h-1
        decisions.append(dict(review_id=row['review_id'],member_id=row['member_id'],lineage_id=row['lineage_id'],label_line_index=row['label_line_index'],subset=row['subset'],size_bin=row['size_bin'],image_exposures=row['image_exposures'],bbox_xyxy=row['bbox_xyxy'],
            panel_condition=PANEL[pc],occlusion_condition=OCCLUSION[oc],view_descriptor='panel_bearing_face_visible' if pc in 'DL' else 'plain_faces_or_top_only' if pc=='A' else 'unknown',
            image_edge_contact=edge,edge_contact_semantics='bbox within one original pixel of image boundary; not an exact truncation fraction',
            decision='held_condition_unknown' if 'U' in code else 'condition_recorded',reason=REASONS_P[pc]+REASONS_O[oc]+SPECIAL.get(i,''),
            full_context_checked=i in IDS,review_nature='AI-assisted',reviewed_at=datetime.now(timezone.utc).isoformat(),
            image_sha256=row['image_sha256'],label_sha256=row['label_sha256'],evidence_sha256=row['evidence_sha256']))
    validate(m,decisions)
    stats={}
    for subset in ('all','base','regular','bridge_positive'):
        rows=[d for d in decisions if subset=='all' or d['subset']==subset]
        stats[subset]=dict(boxes=len(rows),unique_frames=len({d['member_id'] for d in rows}),lineages=len({d['lineage_id'] for d in rows}),panel=dict(Counter(d['panel_condition'] for d in rows)),occlusion=dict(Counter(d['occlusion_condition'] for d in rows)),
            panel_exposures={k:sum(d['image_exposures'] for d in rows if d['panel_condition']==k) for k in PANEL.values()})
    save(OUT/'review.json',dict(status='all_boxes_inspected_with_held_conditions',decisions=decisions,counts=stats,held_review_ids=[d['review_id'] for d in decisions if d['decision']=='held_condition_unknown'],
        priority_instance_visibility_recheck=['T049','T054','T148'],all_conditions_resolved=False,original_training_admission_changed=False,
        scope='Additional AI-assisted condition audit, not a fresh all-label admission. Plain/no-visible-panel is not low panel contrast. Counts are frame-label events and exposure counts, not independent assets.',
        inputs={str(q):file_sha256(q) for q in (mp,OUT/'full-context.json',Path(__file__))}))
    print('STATS',stats);print('HELD',[d['review_id'] for d in decisions if d['decision']=='held_condition_unknown'])
if __name__=='__main__':main()

