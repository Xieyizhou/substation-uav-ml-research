"""URI-safe repaired replay; a fresh explicit retry-budget receipt is required."""
import argparse
import asyncio
import hashlib
import shutil
from pathlib import Path
from scripts.vision.closed_exterior_material_capture import OUT as SOURCE,freeze,prior,read_record
from scripts.vision.build_closed_exterior_evidence import verified_run
from scripts.vision.verify_designed_full_scene_poses import replay_frame,DESIGN

OUT=SOURCE/'original-replays-repaired-v1'


def safe_unit(pair):
    return 'u-'+hashlib.sha256(pair.encode()).hexdigest()[:24]


def check_path(path):
    if ':' in str(Path(path).resolve()):raise ValueError('URI_unsafe_Gazebo_world_path')


def preflight():
    p=freeze();old=SOURCE/'original-replays/completion.json';r=prior.read(old);prior.verify(r)
    rows=[];deps=[old,SOURCE/'protocol.json',Path(__file__)]
    for unit in r['results']:
        rp=Path(unit['receipt']);v=prior.read(rp);prior.verify(v)
        attempts=list(rp.parent.parent.glob('attempt-*/receipt.json'))
        if len(attempts)!=3:raise ValueError('Unexpected prior retry accounting')
        for attempt in attempts:
            a=prior.read(attempt);prior.verify(a);log=attempt.parent/'simulator.log'
            text=log.read_text()
            if a['status']!='technical_failure' or 'Unable to find file with URI' not in text or 'Failed to find world' not in text:
                raise ValueError('Failure cause differs; manual investigation required')
            if not a['process_cleanup_complete']:raise ValueError('Prior cleanup not complete')
            deps.extend([attempt,log])
        proposed=OUT/safe_unit(unit['pair_id'])/'replay/original/attempt-01/world.sdf';check_path(proposed)
        rows.append(dict(pair_id=unit['pair_id'],old_attempts=3,safe_unit=safe_unit(unit['pair_id']),proposed_world_path=str(proposed)))
    if len({x['safe_unit'] for x in rows})!=len(rows):raise ValueError('Path identity collision')
    OUT.mkdir(exist_ok=True);dest=OUT/'preflight.json'
    if dest.exists():record=prior.read(dest);prior.verify(record);return record
    return prior.frozen(dest,dict(status='path_fix_verified_fresh_retry_authorization_required',units=rows,
        old_attempts_preserved=48,worlds_successfully_loaded_in_failed_attempts=0,
        retry_budget_reset_authorized=False,training_started=False,
        inputs={str(path):prior.file_sha256(path) for path in deps}))


def authorization():
    p=preflight();path=OUT/'retry-authorization.json'
    if not path.exists():raise ValueError('Fresh retry budget not authorized; refusing fourth attempt')
    a=prior.read(path);prior.verify(a)
    if a.get('status')!='explicit_user_authorized_repaired_retry_budget' or a.get('attempts_per_frame')!=3:
        raise ValueError('Invalid fresh retry authorization')
    if a.get('preflight_identity')!=p['identity']:raise ValueError('Authorization identity mismatch')
    return p,path


async def run():
    pf,auth=authorization();p=freeze();results=[];deps=[OUT/'preflight.json',auth]
    for runrow in (r for r in p['runs'] if r['variant']=='original'):
        rows,mapping,cp=verified_run(runrow)
        for u in (u for u in p['units'] if u['run_key']==runrow['key']):
            row=rows[u['view_id']];ep=SOURCE/'evidence'/u['key']/'evidence.json';e=prior.read(ep);prior.verify(e)
            unit=OUT/safe_unit(u['pair_id']);check_path(unit);unit.mkdir(exist_ok=True);sp=unit/'source.json'
            if not sp.exists():prior.frozen(sp,dict(row,inputs={str(cp):prior.file_sha256(cp)}))
            f=dict(member_id=u['key'],lineage_id=u['pair_id'],class_name=u['category'],review_ids=['original'],
                source_image=row['rgb_path'],source_receipt=str(sp),source_plan=u['plan_path'],source_world=str(Path(u['plan_path']).parent/'world.sdf'),
                actual_pose=row['actual_pose'],world_name=read_record(u['plan_path'])['world_name'],instance_mapping=mapping,
                events=[dict(review_id='label-'+x['runtime_label'],runtime_label=int(x['runtime_label']),object_id=x['object_id'],bbox_xyxy=x['truth']['bbox_xyxy']) for x in e['events']])
            helper=unit/'gz_visibility_capture_cleanup_fixed'
            if not helper.exists():shutil.copy2(DESIGN.parent/'native-source-replay-v2/gz_visibility_capture_cleanup_fixed',helper)
            pp=unit/'protocol.json'
            if not pp.exists():prior.frozen(pp,dict(frame=f,training_ready=False,inputs={str(x):prior.file_sha256(x) for x in (cp,ep,sp,helper,auth,Path(__file__))}))
            prior.verify(prior.read(pp));rp,r=await replay_frame(f,unit);deps.extend([pp,rp])
            results.append(dict(pair_id=u['pair_id'],status=r['status'] if r else 'attempts_exhausted',receipt=str(rp)))
            # Stop the entire batch at the first failed unit instead of repeating the same defect across frames.
            if not r or r['status']!='original_pixel_evidence_certified':
                prior.frozen(OUT/'blocked-completion.json',dict(status='repaired_replay_blocked_no_expansion',results=results,
                    inputs={str(x):prior.file_sha256(x) for x in deps}));return
    prior.frozen(OUT/'completion.json',dict(status='original_checks_complete_not_variant_certification',results=results,
        inputs={str(x):prior.file_sha256(x) for x in deps}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');a=ap.parse_args()
    if a.replay:asyncio.run(run())
    else:print(preflight()['status'],'NO_REPLAY_NO_TRAINING')
