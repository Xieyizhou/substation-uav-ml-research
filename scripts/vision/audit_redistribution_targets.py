"""Twelve-target source attribution and explicit visual conclusions; no replay."""
from pathlib import Path
from datetime import datetime,timezone
from scripts.vision.prepare_redistribution_review import OUT as PRIOR,ROOT,read,verify,frozen,file_sha256
from scripts.vision.redistribution_risk_crops import TARGETS
from scripts.vision.review_redistribution_members import RISK
from scripts.vision.check_structure_fit_sources import equal_rgb,label_correspondence
from scripts.vision.instance_visibility_diagnosis import raw_box
from src.vision.canonical.gates import instance_mapping,annotation_mode_from_world
from scripts.vision import run_visibility_cleanup_validation as replay
from scripts.vision.resume_supervision_risk_pilot import OUT as PILOT
from scripts.vision.supervision_risk_revision import OUT as OLD_RISK
from src.ml.artifacts import object_sha256

OUT=PRIOR/'target-attribution-review-v1'
OCCLUDED={'P31','P44','P47','P53'}

def resolve(raw_boxes,truth,mapping):
    candidates=[b for b in raw_boxes if max(abs(x-y) for x,y in zip(raw_box(b),truth['bbox_xyxy']))<1e-4]
    if len(candidates)!=1:raise ValueError('Ambiguous raw target')
    label=int(candidates[0]['label'])
    if label not in mapping or mapping[label]['category']!=truth['class_name']:raise ValueError('Target category/instance unresolved')
    return label,mapping[label]

def build():
    e=read(PRIOR/'evidence.json');review=read(PRIOR/'review.json');crops=read(PRIOR/'risk-crops.json')
    for r in (e,review,crops):verify(r)
    cropmap={c['event_id']:c for c in crops['targets']}
    paths=[PRIOR/'evidence.json',PRIOR/'review.json',PRIOR/'risk-crops.json',PRIOR/'completion.json',Path(__file__),
        ROOT/'src/vision/canonical/gates.py',ROOT/'scripts/vision/instance_visibility_diagnosis.py']
    registries=[replay.OUT/'protocol.json',PILOT/'protocol.json',OLD_RISK/'protocol.json']
    known=[]
    for path in registries:
        doc=read(path);verify(doc);paths.append(path)
        known.extend((str(path),f) for f in doc['frames'])
    decisions=[];now=datetime.now(timezone.utc).isoformat()
    for r in e['events']:
        eid=r['event_id']
        if eid not in TARGETS:continue
        ip=Path(r['source']['source_image']);rp=ip.parents[1]/(ip.parent.name+'.json');pp=ip.parents[2]/'plan/plan.json';wp=pp.parent/'world.sdf'
        rec=read(rp);plan=read(pp);mapping=instance_mapping(plan);truth=r['truth'][TARGETS[eid]]
        if file_sha256(wp)!=plan['files']['world.sdf']:raise ValueError('Saved world hash differs')
        equal_rgb(ip,r['member']['image_path']);label_correspondence(r['truth'],rec['truth']['objects'])
        label,obj=resolve(rec['raw_truth']['annotatedBox'],truth,mapping)
        if len({x['object_id'] for x in mapping.values()})!=len(mapping):raise ValueError('Mapping collision')
        original=r['source']['annotations'][TARGETS[eid]]
        if str(label)!=str(original['runtime_label']) or original['object_id'] not in ('unknown',obj['object_id']):raise ValueError('Trace identity conflict')
        mode=annotation_mode_from_world(wp)
        if mode!='full_2d':raise ValueError('Unexpected actual world mode')
        candidates=[dict(registry=reg,review_ids=f['review_ids']) for reg,f in known if f['source_image']==str(ip)]
        if candidates:raise ValueError('Same-frame candidate exists; inspect it before concluding no mask')
        for path in (rp,pp,wp,ip,Path(r['member']['image_path']),Path(r['member']['label_path']),Path(cropmap[eid]['crop_path'])):paths.append(path)
        decisions.append(dict(event_id=eid,member_id=r['member']['member_id'],source_identity=object_sha256(r),
            truth=truth,runtime_label=label,object_id=obj['object_id'],category=obj['category'],world_mode=mode,
            mapping_basis='saved_plan_posthoc_not_historical_gate_certificate' if r['source']['gaps'] else 'saved_plan_agrees_with_receipt_trace',
            source_image=str(ip),source_receipt=str(rp),source_world=str(wp),actual_pose=rec['actual_pose'],
            original_RGB_and_full_labels_correspond=True,identity_conflict_found=False,
            status='occluded_target_content_unresolved' if eid in OCCLUDED else 'visually_insufficient_content_not_pixel_certified',
            reason=RISK[eid],review_nature='AI辅助审核',reviewed_at=now,prior_results_known_not_blind=True,
            image_sha256=r['member']['image_sha256'],evidence_sha256=r['evidence_sha256'],crop=cropmap[eid],
            component_identity='unknown',pixel_visibility_certified=False,existing_same_frame_candidate_count=0,
            no_mask_scope='Only the three explicitly listed registries were checked; no whole-machine absence claim.',
            recommendation='Do not increase this image exposure until target risk is resolved; not an instruction to delete the image or one label.',
            training_approved=False,exposures=r['exposures']))
    if len(decisions)!=12 or len({d['event_id'] for d in decisions})!=12:raise ValueError('Scope incomplete')
    OUT.mkdir(exist_ok=True)
    return frozen(OUT/'review.json',dict(status='twelve_targets_reviewed_quality_gate_still_blocked',decisions=decisions,
        insufficient_content_events=[d['event_id'] for d in decisions if d['event_id'] not in OCCLUDED],
        attribution_pending_events=sorted(OCCLUDED),checked_replay_registries=[str(x) for x in registries],
        new_replays_started=0,labels_modified=False,new_sampling_generated=False,training_started=False,
        inputs={str(x):file_sha256(x) for x in paths}))

if __name__=='__main__':
    r=build();print('REVIEWED12; CONTENT_INSUFFICIENT8; ATTRIBUTION_PENDING4; NO_TRAINING')
