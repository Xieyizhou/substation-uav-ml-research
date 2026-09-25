"""Authored spatial associations for three individually inspected frames."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior

NOTES = [('U05','cabinet_2','左后蓝色柜体被粗杆遮挡，露出两侧轮廓与cabinet_2对应。'),
         ('U06','cabinet_2','左缘蓝色柜体顶面及宽侧面与cabinet_2对应。'),
         ('U06','cabinet_1','右下近处蓝色柜体被图缘截断，与cabinet_1对应。'),
         ('U08','cabinet_3','左缘带面板柜体与cabinet_3对应，与右侧目标开关柜不同。'),
         ('U08','cabinet_2','中央后方蓝色小柜体与cabinet_2对应。')]


def run():
    ep = OUT/'source-spatial-context/evidence.json'; tp = OUT/'legacy-world-source-trace.json'
    e,t = prior.read(ep),prior.read(tp)
    prior.verify(e);prior.verify(t)
    deps=[ep,tp,Path(__file__).resolve()]; decisions=[]
    for rid,obj,reason in NOTES:
        row=next(x for x in e['rows'] if x['review_id']==rid)
        source=next(x for x in t['members'] if x['member_id']==row['member_id'])
        pp=Path(source['source_plan']);op=pp.parent/'obstacles.json';page=Path(row['page_path'])
        if prior.read(pp)['files']['obstacles.json']!=prior.file_sha256(op):raise ValueError('Source taxonomy drift')
        assets=[x for x in prior.read(op)['obstacles'] if x['name']==obj]
        projections=[x for x in row['projected'] if x['object_id']==obj]
        if len(assets)!=1 or assets[0]['visual_category']!='cabinet' or len(projections)!=1 or projections[0]['saved_labels']:
            raise ValueError('Non-target source association conflict')
        decisions.append(dict(review_id=rid,member_id=row['member_id'],object_id=obj,
            status='named_source_background_association_supported',reason=reason,
            review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),
            page_sha256=prior.file_sha256(page),pixel_visibility_certified=False,training_eligible=False))
        deps.extend([pp,op,page])
    dest=OUT/'source-spatial-context/background-review-04.json'
    if dest.exists():
        r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='five_named_background_associations_reviewed',decisions=decisions,
        scope='Named structures only; full-frame eligibility remains separate.',inputs={str(p):prior.file_sha256(p) for p in deps}))


if __name__=='__main__':print(run()['status'])
