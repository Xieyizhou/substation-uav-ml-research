"""Import explicit observations only after all exact replay evidence is valid."""
from datetime import datetime, timezone
from pathlib import Path
from scripts.vision.physical_low_light_capture import OUT, freeze, prior
from scripts.vision.physical_low_light_observations import NOTES
from scripts.vision.physical_low_light_dataset import validate_review

def main():
    p=freeze(); dest=OUT/'quality-review.json'
    if dest.exists():
        r=prior.read(dest);validate_review(p,r);return r
    if set(NOTES)!={u['unit_id'] for u in p['units']}:raise ValueError('Missing explicit frame observation')
    deps=[OUT/'capture-protocol.json',Path(__file__).resolve(),Path(__file__).with_name('physical_low_light_observations.py'),Path(__file__).with_name('review_physical_low_light_pilot.py')]
    decisions=[]; coverage=[]
    for u in p['units']:
        uid=u['unit_id'];ep=OUT/'evidence'/uid/'evidence.json';cp=OUT/'replays'/uid/'completion.json'
        e,c=prior.read(ep),prior.read(cp);prior.verify(e);prior.verify(c)
        rp=Path(c['receipt_path']);r=prior.read(rp);prior.verify(r)
        if c['status']!='low_light_capture_exact_replay_verified' or r['status']!='original_pixel_evidence_certified':raise ValueError('Replay not certified')
        if len(r['full_mask_coverage'])!=3 or any(x['missing_targets'] for x in r['full_mask_coverage']):raise ValueError('Unboxed mapped target')
        if set(NOTES[uid])!={x['object_id'] for x in e['events']}:raise ValueError('Label review mismatch')
        deps.extend([ep,cp,rp,Path(u['plan_path']).parent/'world.sdf'])
        coverage.append(dict(unit_id=uid,stable_frames=3,all_mapped_visible_targets_boxed=True,receipt_path=str(rp)))
        for x in e['events']:
            decisions.append(dict(event_id=x['event_id'],reason=NOTES[uid][x['object_id']],review_nature='AI辅助审核',review_time=datetime.now(timezone.utc).isoformat(),
                evidence_identity=e['identity'],crop_sha256=x['crop_sha256'],status='content_sufficient_for_bounded_research',training_admitted=False,promotable=False))
    r=prior.frozen(dest,dict(status='32_frames_explicitly_reviewed_exact_replay_complete',decisions=decisions,full_frames_viewed=list(NOTES),coverage=coverage,
        background_investigation=dict(frames=['L21','L22','L23','L24'],observed='Right cyan block and upper gray block were independently questioned during full-frame review.',
            evidence='Saved world cabinet_center body diffuse 0.02 0.32 0.43 at (3,1), control_building at (8,6); both non-target assets. Conservative projections match right cyan block and upper gray building. Three exact masks per frame contain only mapped reactor target; no mapped unboxed target pixels.',
            cabinet_projection_L21=[1469.7141,191.3085,1936.9280,527.7589],building_projection_L21=[856.4431,-99.1107,1280.1729,248.9249],
            limitation='Projection is supporting source identification, not a pixel visibility certificate or a new label.'),
        limitations=['Neutral tops sometimes have weak contrast against floor; visible dark faces/posts remain identifiable. No new independent poses or assets.','Exact replay certifies these new low-light captures only, not equality to normal-light images.','No component-specific mask certification or whole historical pool quality claim.'],
        inputs={str(x):prior.file_sha256(x) for x in deps}))
    validate_review(p,r);return r

if __name__=='__main__':print(main()['status'])
