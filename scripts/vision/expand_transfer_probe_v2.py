"""Expand only frozen diagnostic poses after pilot review; never train."""
import argparse
import asyncio
import fcntl
from pathlib import Path
from scripts.vision import run_condition_transfer_pilot as pilot
from scripts.vision.import_transfer_pilot_review import main as review, validate

OUT=pilot.DESIGN/'expansion-v2'
prior=pilot.prior


def gate():
    p=pilot.preflight(); r=review()
    e=prior.read(pilot.OUT/'review/evidence.json');prior.verify(e)
    if not validate(e,r['decisions']):raise ValueError('Unresolved label review')
    cp=pilot.OUT/'completion.json'; c=prior.read(cp);prior.verify(c)
    if len(c['units'])!=20:raise ValueError('Incomplete pilot')
    paths=[cp,pilot.OUT/'review/label-review.json',pilot.DESIGN/'protocol.json',Path(__file__).resolve()]
    for source in p['sources']:
        sp=Path(source['source_receipt']);s=prior.read(sp);prior.verify(s);paths.append(sp)
        if s['status']!='original_pixel_evidence_certified' or any(x['missing_targets'] for x in s['full_mask_coverage']):
            raise ValueError('Source full-instance coverage gap')
    for u in c['units']:
        rp=Path(u['receipt']);a=prior.read(rp);prior.verify(a);paths.append(rp)
        if not a['process_cleanup_complete'] or a['stable_frames']<3:raise ValueError('Unstable pilot')
        expected='original_pixel_evidence_certified' if u['condition']=='original_control' else 'candidate_rendered_review_pending'
        if a['status']!=expected:raise ValueError('Pilot capture gate failed')
        if any(x['skew_ms']>33.334 or x['maximum_box_delta_px']>1 for x in a['records']):raise ValueError('Alignment gate')
    ids=[s['source_pose_id'] for s in p['sources'] if s['source_pose_id'] not in p['pilot_source_ids']]
    if len(ids)!=8:raise ValueError('Frozen expansion membership changed')
    OUT.mkdir(exist_ok=True);dest=OUT/'protocol.json'
    if dest.exists():prior.verify(prior.read(dest))
    else:prior.frozen(dest,dict(status='frozen_diagnostic_expansion',source_ids=ids,helper=p['helper'],
        training_ready=False,training_started=False,training_admitted=False,promotable=False,
        gate_scope='Source mask full-target coverage and explicit pilot all-label content review; not training admission.',
        unresolved_scope='Unlabelled scene assets remain configured non-targets, not visually certified training negatives.',
        inputs={str(x):prior.file_sha256(x) for x in paths}))
    return p,ids


async def run():
    p,ids=gate();dest=OUT/'completion.json'
    if dest.exists():prior.verify(prior.read(dest));return prior.read(dest)
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=pilot.existing.OUT;pilot.existing.OUT=OUT;results=[];paths=[OUT/'protocol.json']
        try:
            for sid in ids:
                source=next(s for s in p['sources'] if s['source_pose_id']==sid);frame=source['source_frame']
                control=await pilot.existing.unit(frame,sid+'/original_control')
                paths.append(Path(control['receipt']));results.append(dict(source_id=sid,condition='original_control',**control))
                print(sid,'original_control',control['status'],flush=True)
                if control['status']!='original_pixel_evidence_certified':break
                for u in [x for x in p['units'] if x['source_pose_id']==sid]:
                    result=await pilot.existing.unit(frame,sid+'/'+u['condition'],u['world_path'],Path(source['reference_mask']))
                    paths.append(Path(result['receipt']));results.append(dict(source_id=sid,condition=u['condition'],**result))
                    print(sid,u['condition'],result['status'],flush=True)
                    if result['status']!='candidate_rendered_review_pending':break
                else:continue
                break
        finally:pilot.existing.OUT=old
        return prior.frozen(dest,dict(status='captured_explicit_review_pending' if len(results)==40 and all(x['status'] in ('original_pixel_evidence_certified','candidate_rendered_review_pending') for x in results) else 'blocked',
            units=results,training_ready=False,training_started=False,
            inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--capture',action='store_true');a=ap.parse_args()
    if a.capture:print(asyncio.run(run())['status'])
    else:gate();print('PREFLIGHT_ONLY')
