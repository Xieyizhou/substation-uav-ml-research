"""Resolve inherited parent paths separately from actual rendered RGB."""
from pathlib import Path
from scripts.vision.diagnose_small_scale_order_fit import OUT,freeze,prior
from scripts.vision.prepare_unified_lighting_training import DESIGN
from scripts.vision.record_unified_lighting_review import validate
from scripts.vision.freeze_visual_augmentation_240_v2 import pixel_hash


def run():
    p=freeze();cp=OUT/'source-census.json';c=prior.read(cp);prior.verify(c)
    ep=DESIGN/'light-review/evidence.json';rp=DESIGN/'light-review/review.json'
    e=prior.read(ep);r=prior.read(rp)
    prior.verify(e);prior.verify(r);validate(e,r['decisions'])
    deps=[cp,ep,rp,Path(__file__).resolve()];resolved=[];idx={m['member_id']:m for m in p['members']}
    for row in c['members']:
        if row['source_pixel_status']!='pixel_mismatch':continue
        m=idx[row['member_id']]
        if m['variant']!='physical-lighting':raise ValueError('Unexplained pixel change')
        events=[v for v in e['events'] if v['pair_id']==m['pair_id']]
        if len(events)!=1:raise ValueError('Nonunique lighting source')
        v=events[0];actual=Path(v['image']);receipt=Path(v['receipt'])
        prior.verify(prior.read(receipt));deps.extend([actual,receipt])
        if pixel_hash(actual)!=row['pixel_sha256']:raise ValueError('Rendered source differs')
        decisions=[x for x in r['decisions'] if x['pair_id']==m['pair_id']]
        if len(decisions)!=len(m['truth']):raise ValueError('Full review coverage differs')
        resolved.append(dict(member_id=m['member_id'],parent_image=row['source_image'],
            actual_render_image=str(actual),actual_render_sha256=prior.file_sha256(actual),
            receipt=str(receipt),review_decisions=decisions,
            status='rendered_rgb_pixel_equal_parent_is_derivation_not_same_image'))
    duplicates=[]
    for group in c['duplicate_pixel_groups']:
        rows=[idx[k] for k in group]
        labels=[Path(m['label_path']).read_bytes() for m in rows]
        if any(t!=labels[0] for t in labels):raise ValueError('Identical RGB with conflicting full labels')
        lineages={m['lineage_id'] for m in rows}
        if len(lineages)!=1:raise ValueError('Duplicate RGB crosses registered lineage')
        duplicates.append(dict(members=group,lineage_id=next(iter(lineages)),
            full_labels_identical=True,policy='one_pixel_equivalence_group_not_independent_samples; preserve historical identities'))
    dest=OUT/'source-resolution.json'
    if dest.exists():a=prior.read(dest);prior.verify(a);return a
    return prior.frozen(dest,dict(status='four_derived_sources_and_three_duplicate_groups_resolved',
        resolved_sources=resolved,duplicate_groups=duplicates,history_modified=False,
        remaining='Full pool review provenance and usable-set freeze still required.',
        inputs={str(x):prior.file_sha256(x) for x in deps}))


if __name__=='__main__':print(run()['status'])
