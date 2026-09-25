"""Export only the twelve reviewed target-body variants, without admission."""
from pathlib import Path
from scripts.vision import import_transfer_pilot_review as pilot
from scripts.vision import import_transfer_expansion_review as expansion
from scripts.vision.freeze_condition_transfer_probe import OUT as DESIGN
from scripts.vision.audit_material_transfer_scope import CAND,resolve_truth
from scripts.vision.export_material_candidate_batch import export

prior=pilot.prior
OUT=DESIGN/'body-transfer-training-v1'


def main():
    dp=DESIGN/'protocol.json';d=prior.read(dp);prior.verify(d)
    sources={s['source_pose_id']:s for s in d['sources']}
    paths=[dp,Path(__file__).resolve(),DESIGN/'analysis-v1/summary.json',CAND/'dataset-completion.json',
           CAND/'known-source-pose-audit.json',CAND/'resolved-pose-gaps.json']
    for p in paths[2:]:prior.verify(prior.read(p))
    gaps=prior.read(CAND/'resolved-pose-gaps.json')
    if gaps['unresolved'] or gaps['pose_overlaps']:raise ValueError('Source exclusion gap')
    members=[]
    for module in (pilot,expansion):
        r=module.main();ep=module.DEST/'evidence.json';e=prior.read(ep);prior.verify(e)
        if not module.validate(e,r['decisions']):raise ValueError('Unknown review')
        paths += [ep,module.DEST/'label-review.json']
        for page in e['pages']:
            if page['condition']!='gray_target_body':continue
            sid=page['source_id'];source=sources[sid];f=source['source_frame']
            events=[x for x in e['events'] if x['source_id']==sid and x['condition']=='gray_target_body']
            tp=Path(f['source_receipt']);t=prior.read(tp)['truth'];resolved=resolve_truth(t,f['instance_mapping'])
            if [(x['truth'],x['object_id']) for x in events]!=[(a,b['object_id']) for a,b in zip(t['objects'],resolved,strict=True)]:
                raise ValueError('Full-label/instance disagreement')
            rp=Path(events[0]['replay_receipt']);replay=prior.read(rp);prior.verify(replay)
            if replay['status']!='candidate_rendered_review_pending' or not replay['process_cleanup_complete']:raise ValueError('Capture gap')
            members.append(dict(member_id=sid+'-gray_target_body',source_pose_id=sid,pair_id=source['pair_id'],
                planned_object_id=source['target_object_id'],variant='gray_target_body',full_truth=t,
                image_path=events[0]['image_path'],image_sha256=events[0]['image_sha256'],
                instance_mapping=f['instance_mapping'],actual_pose=f['actual_pose'],source_world=f['source_world'],
                source_receipt=str(tp),replay_receipt=str(rp),training_admitted=False,promotable=False))
            paths += [tp,rp]
    if len(members)!=12 or len({m['source_pose_id'] for m in members})!=12:raise ValueError('Incomplete target body set')
    OUT.mkdir(exist_ok=True);review_path=OUT/'reviewed-candidates.json'
    if review_path.exists():prior.verify(prior.read(review_path))
    else:prior.frozen(review_path,dict(status='twelve_reviewed_body_variants_pending_export',members=members,
        training_ready=False,training_started=False,training_admitted=False,promotable=False,
        scope='Same-source bounded development condition replacement, not independent scenes or whole-pool recertification.',
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    export(review_path,OUT/'candidate-export-v1')


if __name__=='__main__':main()
