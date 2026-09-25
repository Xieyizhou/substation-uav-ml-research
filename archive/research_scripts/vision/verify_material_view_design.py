"""Audit existing candidates and freeze a blocked design, never train or collect."""
from pathlib import Path
from collections import Counter
import hashlib
from PIL import Image
from scripts.vision.material_control_feasibility import OUT as HISTORY,CANDIDATE,prior
from scripts.vision.record_closed_material_review import validate,frame_gates
from scripts.vision import endpoint_member_fit as fit
from scripts.vision import train_frozen_multiscale as fixed

OUT=HISTORY.parent/'material-view-diversity-design-v1'

def pixels(path):
    with Image.open(path) as im:
        im=im.convert('RGB')
        return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()

def blockers(rows):
    usable=[r for r in rows if r['status']=='reviewed_candidate_only' and not r['held_labels'] and not r['unboxed_visible_instances']]
    return dict(usable_variants=len(usable),usable_pose_groups=len({r['lineage_id'] for r in usable}),
        new_pose_groups=len({r['lineage_id'] for r in usable if not r['source_pose_already_exposed']}),
        altered_classes=sorted({r['altered_target_class'] for r in usable}),
        missing_altered_classes=sorted({'transformer','switchgear','capacitor_bank','reactor'}-{r['altered_target_class'] for r in usable}))

def main():
    paths=[HISTORY/'coverage-census.json',HISTORY/'risk-containment.json',CANDIDATE/'semantic-review.json',CANDIDATE/'review-evidence.json',CANDIDATE/'mask-coverage.json',CANDIDATE/'protocol.json',fit.OUT/'completion.json']
    objs=[prior.read(p) for p in paths]
    for r in objs:prior.verify(r)
    census,risk,review,evidence,mask,design,_=objs
    validate(evidence,review['decisions'])
    if frame_gates(evidence,review['decisions'],mask)!=review['frames']:raise ValueError('Changed review gate')
    p=fit.freeze();fit.check(p)
    _,source,_=fixed.contract('fixed-7')
    pool={m['member_id']:m for m in source['pool_rows']}
    seen_lineages={pool[mid]['lineage_id'] for model in p['models'].values() for mid,n in model['counts'].items() if n>0}
    pool_hashes={m['image_sha256'] for m in pool.values()};pool_pixels={pixels(m['image_path']) for m in pool.values()}
    dev=prior.read(HISTORY/'initial-gate.json');prior.verify(dev);paths.append(HISTORY/'initial-gate.json')
    devimages={r['image_path']:r['image_sha256'] for r in dev['corrected_target_records']}
    dev_pixels={pixels(path) for path in devimages}
    rows=[];keys=set()
    for candidate in census['candidates']:
        key=(candidate['source_review_id'],candidate['variant'])
        if key in keys:raise ValueError('Duplicate candidate')
        keys.add(key)
        es=[e for e in evidence['events'] if (e['source_review_id'],e['variant'])==key]
        frame=next(x for x in review['frames'] if (x['source_review_id'],x['variant'])==key)
        if not es or len({e['image_sha256'] for e in es})!=1:raise ValueError('Missing or conflicting image')
        image=es[0]['image_path'];pixel=pixels(image)
        classes=dict(Counter(e['source_truth']['class_name'] for e in es))
        if classes!=candidate['full_label_classes'] or classes!=pool[candidate['member_id']]['class_instances']:raise ValueError('Full class budget mismatch')
        src=pool[candidate['member_id']]
        paths += [Path(image),Path(src['image_path']),Path(src['label_path'])]
        for e in es:
            paths += [Path(e['receipt_path']),Path(e['crop_path']),Path(e['full_context_path'])]
            prior.verify(prior.read(e['receipt_path']))
        frame_source=[f for f in design['frames'] if f['member_id']==candidate['member_id']]
        if len(frame_source)!=1:raise ValueError('Nonunique original pose source')
        oldframe=frame_source[0]
        if candidate['member_id'] in risk['denied_members'] and frame['status']!='held_whole_image':raise ValueError('Held source silently restored')
        rows.append(dict(source_review_id=key[0],variant=key[1],source_member_id=candidate['member_id'],
            image_path=image,image_sha256=es[0]['image_sha256'],pixel_sha256=pixel,
            lineage_id=candidate['lineage_id'],source_pose_already_exposed=candidate['lineage_id'] in seen_lineages,
            actual_source_exposures={k:m['counts'].get(candidate['member_id'],0) for k,m in p['models'].items()},
            candidate_file_seen_in_pool=es[0]['image_sha256'] in pool_hashes,candidate_pixels_seen_in_pool=pixel in pool_pixels,
            overlaps_development_file=es[0]['image_sha256'] in devimages.values(),overlaps_development_pixels=pixel in dev_pixels,
            source_pose=oldframe['actual_pose']['position'],source_world=oldframe['source_world'],
            altered_target_class=candidate['altered_target_class'],altered_target_objects=candidate['altered_target_objects'],
            full_label_classes=classes,status=frame['status'],held_labels=frame['held_labels'],unboxed_visible_instances=frame['unboxed_visible_instances'],
            source_independent=False,scope_note='Same known complex layout/assets; byte/pixel exclusion does not certify independent scene',
            training_ready=False))
    coverage=blockers(rows)
    source_counts={str(seed):sum(r['actual_source_exposures'][f'fixed-{seed}'] for r in rows if r['status']=='reviewed_candidate_only' and r['variant']=='warm') for seed in (7,17,27)}
    OUT.mkdir(exist_ok=True);paths += [Path(__file__).resolve(),fit.OUT/'protocol.json']
    prior.frozen(OUT/'verification.json',dict(status='candidate_audit_complete_design_blocked',candidates=rows,coverage=coverage,
        same_pose_replacement_slots_per_seed=source_counts,
        candidate_variant_exposure_not_inferred_from_source_counts=True,
        training_ready=False,training_started=False,collection_started=False,selected_candidate=None,
        scope='Known 8 material candidates; unresolved other pool sources not assumed eligible',
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    print(coverage,'existing source slots',source_counts)

if __name__=='__main__':main()
