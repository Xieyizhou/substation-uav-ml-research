"""Persist previously individually viewed spatial associations with source checks."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior

NOTES = [
    ('U09','cabinet_2','前景左下蓝色块体与cabinet_2位置和主体轮廓相符。'),
    ('U10','cabinet_3','左下蓝色柜体与cabinet_3对应。'),
    ('U10','cabinet_2','中后方蓝色柜体与cabinet_2对应。'),
    ('U10','cabinet_1','右下蓝色柜体与cabinet_1对应。'),
    ('U12','cabinet_2','中央后方未框柜体与cabinet_2对应。'),
    ('U12','cabinet_3','左缘未框柜体与cabinet_3对应，画缘限制保留。'),
    ('U14','cabinet_2','前景下部未框蓝色柜体与cabinet_2对应。'),
    ('U14','cabinet_1','右缘未框蓝色柜体与cabinet_1对应。'),
    ('U15','cabinet_2','左后方蓝色柜体与cabinet_2对应。'),
    ('U18','cabinet_2','中左蓝色柜体与cabinet_2对应。'),
    ('U18','cabinet_1','前景下部蓝色柜体与cabinet_1对应；不认证出画部分。'),
    ('U19','cabinet_2','前景右侧蓝色柜体与cabinet_2对应。'),
    ('U21','cabinet_2','前景下缘蓝色顶面三角片段与cabinet_2对应，只有局部内容，不声称整柜可见。'),
    ('U36','cabinet_west','中央后方未框蓝色柜体与cabinet_west对应，与前方两台变压器不同。'),
    ('U37','cabinet_center','左侧变压器后露出的窄蓝色片段与cabinet_center对应。'),
    ('U37','control_building','电容器后灰色立面与control_building对应；重叠的transformer_mid投影不代表其像素可见。'),
]


def run():
    ep=OUT/'source-spatial-context/evidence.json';tp=OUT/'legacy-world-source-trace.json'
    e,t=prior.read(ep),prior.read(tp);prior.verify(e);prior.verify(t)
    deps=[ep,tp,Path(__file__).resolve()];decisions=[]
    for rid,obj,reason in NOTES:
        row=next(x for x in e['rows'] if x['review_id']==rid)
        source=next(x for x in t['members'] if x['member_id']==row['member_id'])
        pp=Path(source['source_plan']);op=pp.parent/'obstacles.json';page=Path(row['page_path'])
        if prior.read(pp)['files']['obstacles.json']!=prior.file_sha256(op):raise ValueError('Source taxonomy drift')
        assets=[x for x in prior.read(op)['obstacles'] if x['name']==obj]
        projected=[x for x in row['projected'] if x['object_id']==obj]
        category='control_building' if obj=='control_building' else 'cabinet'
        if len(assets)!=1 or assets[0]['visual_category']!=category or len(projected)!=1 or projected[0]['saved_labels']:
            raise ValueError('Source association conflict: '+rid+':'+obj)
        decisions.append(dict(review_id=rid,member_id=row['member_id'],object_id=obj,reason=reason,
            source_category=category,status='named_source_background_association_supported',
            review_nature='AI辅助审核',recorded_at=datetime.now(timezone.utc).isoformat(),
            timing_note='Observation persisted after prior individual page inspection; timestamp is recording time.',
            page_sha256=prior.file_sha256(page),pixel_visibility_certified=False,training_eligible=False))
        deps.extend([pp,op,page])
    dest=OUT/'source-spatial-context/background-review-05.json'
    if dest.exists():
        r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='sixteen_previously_viewed_named_associations_recorded',decisions=decisions,
        scope='Named associations only, not whole-frame completeness or dataset admission.',
        inputs={str(p):prior.file_sha256(p) for p in deps}))


if __name__=='__main__':print(run()['status'])
