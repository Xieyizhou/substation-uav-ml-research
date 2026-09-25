"""Explicit unknown outcome for the remaining lower-left fragment."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT, prior


def run():
    pp=OUT/'protocol.json';ep=OUT/'remaining-review/evidence.json';sp=OUT/'source-spatial-context/evidence.json'
    p,e,s=[prior.read(x) for x in (pp,ep,sp)]
    for r in (p,e,s):prior.verify(r)
    event=next(x for x in e['events'] if x['review_id']=='U33')
    member=next(x for x in p['members'] if x['member_id']==event['member_id'])
    image=Path(member['image_path'])
    if prior.file_sha256(image)!=member['image_sha256']:raise ValueError('Image changed')
    dest=OUT/'u33-fragment-review.json'
    if dest.exists():
        r=prior.read(dest);prior.verify(r);return r
    deps=[pp,ep,sp,image,Path(member['label_path']),Path(__file__).resolve()]
    return prior.frozen(dest,dict(status='hold_pending_unresolved_edge_fragment',member_id=member['member_id'],
        review_id='U33',review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),
        reason='原尺寸图左缘约y=636至667像素处仍有窄深色结构片段及邻近阴影；现有主体投影没有覆盖该片段，不能从右侧两台普通柜体的来源判断外推其归属。',
        object_id='unknown',pixel_visibility_certified=False,training_eligible=False,
        interpretation='Named evidence gap, not a confirmed missing target label. Whole frame remains pending; labels unchanged.',
        inputs={str(x):prior.file_sha256(x) for x in deps}))


if __name__=='__main__':print(run()['status'])
