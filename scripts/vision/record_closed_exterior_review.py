"""Bind explicit observations. Failed replay does not become approval."""
from datetime import datetime,timezone
from pathlib import Path
from scripts.vision.closed_exterior_material_capture import OUT,freeze,prior
from scripts.vision.closed_exterior_review_observations import OBS,HOLDS,VARIANT_NOTES
from src.ml.artifacts import object_sha256


def validate(evidence,decisions):
    expected={x['event_id']:(e,x) for e in evidence for x in e['events']}
    if len(expected)!=len(decisions) or len({d['event_id'] for d in decisions})!=len(decisions):raise ValueError('missing_duplicate_decisions')
    for d in decisions:
        e,x=expected[d['event_id']]
        if d['crop_sha256']!=x['crop_sha256'] or d['truth_identity']!=object_sha256(x['truth']):raise ValueError('stale_decision')
        if d['evidence_identity']!=e['identity'] or not d['reason']:raise ValueError('stale_evidence_or_missing_reason')


def main():
    p=freeze();ep=OUT/'evidence/completion.json';evidence_summary=prior.read(ep);prior.verify(evidence_summary)
    rp=OUT/'original-replays/completion.json';r=prior.read(rp);prior.verify(r)
    if len(r['results'])!=16:raise ValueError('incomplete_replay_inventory')
    observations=[];frames=[];es=[];paths=[ep,rp,Path(__file__),Path(__file__).with_name('closed_exterior_review_observations.py')]
    for u in p['units']:
        path=OUT/'evidence'/u['key']/'evidence.json';e=prior.read(path);prior.verify(e);es.append(e);paths.append(path)
        notes=OBS[u['pair_id']]
        if set(notes)!={x['object_id'] for x in e['events']}:raise ValueError('observation_instance_inventory_changed')
        for x in e['events']:
            observations.append(dict(event_id=x['event_id'],object_id=x['object_id'],reason=notes[x['object_id']]+' '+VARIANT_NOTES[u['variant']],
                status='held_content_risk' if u['pair_id'] in HOLDS else 'visual_content_reviewed_instance_confirmation_pending',
                review_nature='AI辅助审核',reviewed_at=datetime.now(timezone.utc).isoformat(),
                crop_sha256=x['crop_sha256'],truth_identity=object_sha256(x['truth']),evidence_identity=e['identity'],
                training_admitted=False,promotable=False))
        frames.append(dict(key=u['key'],pair_id=u['pair_id'],full_frame_inspected=True,
            status='held_whole_pair_content_risk' if u['pair_id'] in HOLDS else 'pending_instance_confirmation',
            pixel_visibility_certified=False,unboxed_instance_check='pending_replay',training_ready=False))
    validate(es,observations)
    path=OUT/'visual-review.json'
    if path.exists():prior.verify(prior.read(path));return
    prior.frozen(path,dict(status='64_frames_visually_reviewed_no_training_clearance',frames=frames,decisions=observations,
        held_pairs=sorted(HOLDS),unresolved_replay_pairs=[x['pair_id'] for x in r['results'] if x['status']!='original_pixel_evidence_certified'],
        training_ready=False,training_started=False,inputs={str(p):prior.file_sha256(p) for p in paths}))
    print('FRAMES',len(frames),'LABELS',len(observations),'HELD_PAIRS',len(HOLDS))


if __name__=='__main__':main()
