"""Bind lossless exports to exact existing cohort decisions, not filename guesses."""
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT,freeze,prior
from scripts.vision.small_scale_review_gate import validate_decisions
from scripts.vision.freeze_visual_augmentation_240_v2 import pixel_hash


def run():
    p=freeze();idx={m['member_id']:m for m in p['members']};links=[];deps=[OUT/'protocol.json',Path(__file__).resolve()]
    from scripts.vision.small_scale_material_capture import OUT as SMALL
    from scripts.vision.neutral_gray_capture import OUT as GRAY
    from scripts.vision.cool_light_capture import OUT as COOL
    for root in (SMALL,GRAY,COOL):
        rp=root/'quality-review.json';mp=root/'export/manifest.json';cp=root/'capture-protocol.json'
        r=prior.read(rp);manifest=prior.read(mp);protocol=prior.read(cp)
        for v in (r,manifest,protocol):prior.verify(v)
        deps.extend([rp,mp,cp]);frames={}
        for u in protocol['units']:
            ep=root/'evidence'/u['unit_id']/'evidence.json';e=prior.read(ep);prior.verify(e);deps.append(ep)
            frames[u['unit_id']]=e
        if root==SMALL:validate_decisions(frames,r['decisions'],r['full_frames_viewed'])
        else:
            # Existing cohort-specific validators enforce exact inventory and signatures.
            if root==GRAY:
                from scripts.vision.neutral_gray_dataset import validate_review
            else:
                from scripts.vision.cool_light_dataset import validate_review
            validate_review(protocol,r)
        bypixel={}
        for uid,e in frames.items():
            px=pixel_hash(e['image_path'])
            if px in bypixel:raise ValueError('Ambiguous source frame')
            bypixel[px]=(uid,e)
        for m in manifest['members']:
            mid=m['member_id']
            if mid not in idx:continue
            active=idx[mid]
            if active['image_sha256']!=m['image_sha256'] or active['label_sha256']!=m['label_sha256']:
                raise ValueError('Export identity drift')
            uid,e=bypixel[pixel_hash(active['image_path'])]
            ids={x['event_id'] for x in e['events']};ds=[d for d in r['decisions'] if d['event_id'] in ids]
            if len(ds)!=len(active['truth']):raise ValueError('Incomplete full-label review link')
            links.append(dict(member_id=mid,review_path=str(rp),evidence_identity=e['identity'],
                source_unit=uid,decision_ids=sorted(ids),label_sha256=active['label_sha256'],image_sha256=active['image_sha256'],
                full_labels_reviewed=len(ds),status='existing_explicit_bounded_review_validated',
                limitation='Existing visual/instance evidence limits retained; not general training admission.'))
    if len({x['member_id'] for x in links})!=len(links):raise ValueError('Duplicate cohort link')
    dest=OUT/'cohort-review-links.json'
    if dest.exists():r=prior.read(dest);prior.verify(r);return r
    return prior.frozen(dest,dict(status='three_cohort_review_links_verified_remaining_pool_pending',members=links,
        inputs={str(x):prior.file_sha256(x) for x in deps}))


if __name__=='__main__':
    r=run();print(r['status'],len(r['members']))
