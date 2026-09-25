"""Explicit reviewed spatial associations, not auto-decisions from projection."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior


def run():
    dest=OUT/'source-spatial-context/background-review-01.json'
    if dest.exists():
        r=prior.read(dest);prior.verify(r);return r
    ep=OUT/'source-spatial-context/evidence.json';tp=OUT/'legacy-world-source-trace.json'
    rp=OUT/'remaining-review/partial-review-01.json'
    bp=prior.ROOT/'config/perception/visual_experiment_baseline_v1.json'
    e,t,r=[prior.read(x) for x in (ep,tp,rp)]
    for item in (e,t,r):prior.verify(item)
    if prior.read(bp)['ordinary_cabinet_is_target'] is not False:raise ValueError('Taxonomy changed')
    deps=[ep,tp,rp,bp,Path(__file__).resolve()];decisions=[]
    for rid,obj,reason in [
        ('U01','cabinet_3','左侧未框柜体与cabinet_3主体投影位置、宽高和朝向相符；保存来源将其定义为普通柜体。'),
        ('U02','cabinet_1','右后未框柜体与cabinet_1主体投影位置和轮廓相符，与前方已有标签的switchgear_south是不同资产。')]:
        row=next(x for x in e['rows'] if x['review_id']==rid)
        source=next(x for x in t['members'] if x['member_id']==row['member_id'])
        pp=Path(source['source_plan']);op=pp.parent/'obstacles.json'
        if prior.read(pp)['files']['obstacles.json']!=prior.file_sha256(op):raise ValueError('Source taxonomy hash changed')
        matches=[x for x in prior.read(op)['obstacles'] if x['name']==obj]
        if len(matches)!=1 or matches[0]['visual_category']!='cabinet':raise ValueError('Not uniquely ordinary cabinet')
        association=next(x for x in row['projected'] if x['object_id']==obj)
        if association['saved_labels']:raise ValueError('Unexpected target labels')
        page=Path(row['page_path']);deps.extend([pp,op,page])
        decisions.append(dict(review_id=rid,member_id=row['member_id'],object_id=obj,
            status='named_unboxed_structure_is_source_defined_non_target_cabinet',reason=reason,
            review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),
            spatial_page_sha256=prior.file_sha256(page),pixel_visibility_certified=False,
            training_eligible=False,scope='Resolves this named unboxed structure only; not whole-frame or dataset admission.'))
    return prior.frozen(dest,dict(status='two_named_background_questions_resolved',decisions=decisions,
        history_modified=False,inputs={str(x):prior.file_sha256(x) for x in deps}))


if __name__=='__main__':print(run()['status'])
