"""Explicit closure of previously viewed frames and named background follow-ups."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT,prior

NO_EXTRA={'U03','U04','U07','U11','U13','U16','U17','U20','U23','U24','U30','U34','U35'}
NAMED={
 'U01':['cabinet_3'],'U02':['cabinet_1'],'U06':['cabinet_2','cabinet_1'],'U08':['cabinet_3','cabinet_2'],
 'U09':['cabinet_2'],'U10':['cabinet_3','cabinet_2','cabinet_1'],'U12':['cabinet_2','cabinet_3'],
 'U14':['cabinet_2','cabinet_1'],'U15':['cabinet_2'],'U18':['cabinet_2','cabinet_1'],'U19':['cabinet_2'],
 'U21':['cabinet_2'],'U25':['cabinet_2'],'U26':['cabinet_2'],'U27':['cabinet_2'],'U32':['cabinet_2'],
 'U33':['cabinet_1','cabinet_3'],
 'U37':['cabinet_center','control_building'],'U38':['cabinet_center','control_building'],
 'U39':['cabinet_center','control_building'],'U40':['cabinet_center','control_building'],
 'U41':['cabinet_center','control_building'],'U42':['cabinet_center','control_building'],
 'U51':['cabinet_center','control_building'],
}


def run():
    dest=OUT/'remaining-fullframe-closure.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    names=['full-frame-evidence-index','remaining-review/review','authored-quality-dispositions-01',
           'authored-quality-dispositions-02','u33-fragment-followup','closeout-inventory-v2']
    paths=[OUT/(n+'.json') for n in names];index,review,first,second,follow,q=[prior.read(p) for p in paths]
    for r in (index,review,first,second,follow,q):prior.verify(r)
    keep={m['member_id'] for m in q['members'] if m['stage']!='whole_frame_held'}
    frames=[m for m in index['members'] if m['member_id'] in keep]
    if {m['review_id'] for m in frames}!=NO_EXTRA|set(NAMED):raise ValueError('Authored full-frame selection drift')
    ds={d['member_id']:d for d in review['decisions']}
    quality={d['member_id']:d for r in (first,second) for d in r['decisions']};cache={};decisions=[]
    allowed={'named_source_background_association_supported','named_structure_is_source_defined_non_target_cabinet',
        'named_unboxed_structure_is_source_defined_non_target_cabinet','named_background_source_association_supported'}
    for m in frames:
        mid=m['member_id'];rid=m['review_id'];d=ds[mid];qual=quality[mid]
        if qual['quality_status']!='supported_for_bounded_research_with_recorded_limits':raise ValueError('Quality held')
        if qual['image_sha256']!=d['image_sha256'] or qual['label_sha256']!=d['label_sha256']:raise ValueError('Decision drift')
        named=m['named_source_decisions']
        if {x['object_id'] for x in named}!=set(NAMED.get(rid,[])):raise ValueError('Missing named background resolution')
        reasons=[]
        for link in named:
            path=Path(link['path'])
            if path not in cache:cache[path]=prior.read(path);prior.verify(cache[path]);paths.append(path)
            r=cache[path];matches=[x for x in r['decisions'] if x.get('member_id',r.get('member_id'))==mid and x['object_id']==link['object_id']]
            if len(matches)!=1 or matches[0]['status'] not in allowed:raise ValueError('Unresolved source association')
            reasons.append(dict(object_id=link['object_id'],source=str(path),decision=matches[0]))
        reason='复核已有全图观察及逐标签质量决定：未见额外明确目标；保持该帧遮挡、截断和封闭外壳限制。'
        if named:reason='复核已有全图观察后，具名未框结构已分别由保存世界与逐图空间观察支持为普通柜体／建筑；不依颜色判断类别，保留非像素认证限制。'
        if rid=='U33':
            if follow['member_id']!=mid or follow['status']!='ordinary_cabinet_base_association_supported_with_limits':raise ValueError('U33 fragment unresolved')
            reason+=' 左下片段另有 cabinet_2 基座后续判断；投影较大，不宣称精确像素归属。'
        decisions.append(dict(member_id=mid,review_id=rid,status='existing_fullframe_questions_closed_with_recorded_limits',
            reason=reason,original_full_frame_observation=d['full_frame_observation'],quality_reason=qual['reason'],
            named_source_decisions=reasons,image_sha256=d['image_sha256'],label_sha256=d['label_sha256'],page_sha256=d['page_sha256'],
            review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),
            evidence_basis='Reconciliation of existing viewed-frame decisions; no new automatic visual pass.',pixel_visibility_certified=False,training_eligible=False))
    paths.append(Path(__file__).resolve())
    return prior.frozen(dest,dict(status='37_existing_fullframe_reviews_closed',decisions=decisions,
        inputs={str(path):prior.file_sha256(path) for path in paths}))


if __name__=='__main__':print(run()['status'])
