"""Reuse strict full-label legacy decisions, retaining all adverse outcomes."""
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT,freeze,prior
from scripts.vision.revision_compensation_feasibility import reviewed_members,CONTROL,SECOND


def run():
    p=freeze();accepted,deps=reviewed_members(p['members']);links=[]
    for root in (CONTROL,SECOND):
        e=prior.read(root/'evidence.json');r=prior.read(root/'review.json')
        ds={d['event_id']:d for d in r['decisions']}
        for f in e['events']:
            mid=f['member']['member_id']
            if mid not in accepted:continue
            links.append(dict(member_id=mid,review_path=str(root/'review.json'),
                image_sha256=f['member']['image_sha256'],label_sha256=f['member']['label_sha256'],
                decisions=[ds[l['event_id']] for l in f['labels']],
                status='strict_existing_full_label_review_validated_limits_retained'))
    dest=OUT/'legacy-review-links.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    deps += [OUT/'protocol.json',Path(__file__).resolve(),Path(__file__).with_name('revision_compensation_feasibility.py')]
    return prior.frozen(dest,dict(status='legacy_full_label_bindings_verified_not_whole_pool_admission',
        accepted_member_ids=sorted(accepted),members=links,inputs={str(x):prior.file_sha256(x) for x in deps}))


if __name__=='__main__':
    r=run();print(len(r['accepted_member_ids']))
