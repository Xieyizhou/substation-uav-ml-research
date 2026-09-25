"""Bind older explicitly reviewed cohorts without rerunning historical writers."""
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT,freeze,prior
from scripts.vision.closed_source_training_quality import OUT as CLOSED
from scripts.vision.record_closed_exterior_review import validate
from scripts.vision.same_source_material_quality_v2 import OUT as SAME
from scripts.vision.freeze_visual_augmentation_240_v2 import pixel_hash


def run():
    p=freeze();idx={m['member_id']:m for m in p['members']};links=[]
    rp=CLOSED/'quality-review.json';mp=CLOSED/'export/manifest.json';qp=SAME/'quality-v2.json'
    r=prior.read(rp);manifest=prior.read(mp);q=prior.read(qp)
    for v in (r,manifest,q):prior.verify(v)
    deps=[OUT/'protocol.json',rp,mp,qp,Path(__file__).resolve()];evidence=[]
    for m in manifest['members']:
        ep=Path(m['evidence_path']);e=prior.read(ep);prior.verify(e);deps.append(ep);evidence.append(e)
        mid=m['member_id']
        if mid not in idx:continue
        if m['label_sha256']!=idx[mid]['label_sha256'] or pixel_hash(m['image_path'])!=pixel_hash(idx[mid]['image_path']):
            raise ValueError('Current cohort export differs')
        ds=[d for d in r['decisions'] if d['evidence_identity']==e['identity']]
        if len(ds)!=len(idx[mid]['truth']) or any(d['status']!='approved_for_bounded_research_cohort' for d in ds):
            raise ValueError('Incomplete or unapproved source review')
        links.append(dict(member_id=mid,review_path=str(rp),image_sha256=idx[mid]['image_sha256'],
            label_sha256=idx[mid]['label_sha256'],decisions=ds,status='existing_bounded_review_validated'))
    validate(evidence,r['decisions'])
    for m in q['members']:
        mid=m['member_id']
        if mid not in idx:continue
        current=idx[mid]
        if m['image_sha256']!=current['image_sha256'] or m['label_sha256']!=current['label_sha256']:
            raise ValueError('Older quality binding differs from current member')
        if m['disposition']!='existing_explicit_review_bound_not_new_admission' or len(m['reviews'])!=len(current['truth']):
            raise ValueError('Unresolved older review binding')
        for link in m['reviews']:
            path=Path(link['review_path']);prior.verify(prior.read(path));deps.append(path)
        links.append(dict(member_id=mid,image_sha256=current['image_sha256'],label_sha256=current['label_sha256'],
            review_path=str(qp),decisions=m['reviews'],status='explicit_review_bound_semantic_gate_still_required'))
    dest=OUT/'early-positive-review-links.json'
    if dest.exists():v=prior.read(dest);prior.verify(v);return v
    return prior.frozen(dest,dict(status='older_positive_review_links_complete_not_pool_approval',members=links,
        inputs={str(x):prior.file_sha256(x) for x in deps}))


if __name__=='__main__':
    r=run();print(r['status'],len(r['members']))
