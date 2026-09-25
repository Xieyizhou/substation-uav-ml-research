"""Explicit diagnostic-only four-source pilot; never collects admitted training data."""
import argparse
import asyncio
import fcntl
from pathlib import Path
from scripts.vision.freeze_condition_transfer_probe import freeze,OUT as DESIGN
from scripts.vision import capture_designed_material_triplets as existing
from scripts.vision.audit_material_transfer_scope import prior

OUT=DESIGN/'pilot-v1'


def preflight():
    p=freeze();OUT.mkdir(exist_ok=True)
    paths=[DESIGN/'protocol.json',Path(__file__).resolve(),Path(existing.__file__),Path(p['helper'])]
    paths += [prior.ROOT/'scripts/vision'/f for f in ('run_visibility_cleanup_validation.py','expand_material_view_n05.py','build_material_view_world_drafts.py')]
    body=dict(status='diagnostic_pilot_frozen',helper=p['helper'],source_ids=p['pilot_source_ids'],
              training_ready=False,training_started=False,max_attempts=3,
              purpose='Same-pose intervention evidence only, no training dataset export or admission',
              inputs={str(x):prior.file_sha256(x) for x in paths})
    path=OUT/'protocol.json'
    if path.exists():
        r=prior.read(path);prior.verify(r)
        if any(r[k]!=v for k,v in body.items()):raise ValueError('Runner identity drift')
    else:prior.frozen(path,body)
    return p


async def run():
    p=preflight();dest=OUT/'completion.json'
    if dest.exists():prior.verify(prior.read(dest));return prior.read(dest)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=existing.OUT;existing.OUT=OUT;results=[];paths=[OUT/'protocol.json']
        try:
            for sid in p['pilot_source_ids']:
                source=next(s for s in p['sources'] if s['source_pose_id']==sid)
                frame=source['source_frame']
                control=await existing.unit(frame,sid+'/original_control')
                paths.append(Path(control['receipt']));results.append(dict(source_id=sid,condition='original_control',**control))
                if control['status']!='original_pixel_evidence_certified':break
                for u in [r for r in p['units'] if r['source_pose_id']==sid]:
                    result=await existing.unit(frame,sid+'/'+u['condition'],u['world_path'],Path(source['reference_mask']))
                    paths.append(Path(result['receipt']));results.append(dict(source_id=sid,condition=u['condition'],**result))
                    if result['status']!='candidate_rendered_review_pending':break
                else:continue
                break
        finally:existing.OUT=old
        return prior.frozen(dest,dict(status='pilot_captured_explicit_review_pending' if len(results)==20 and all(r['status'] in ('candidate_rendered_review_pending','original_pixel_evidence_certified') for r in results) else 'pilot_blocked_no_expansion',
            units=results,training_started=False,training_ready=False,expanded=False,
            inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
    if args.pilot:print(asyncio.run(run())['status'])
    else:preflight();print('PREFLIGHT_ONLY_NO_RENDER_NO_TRAINING')
