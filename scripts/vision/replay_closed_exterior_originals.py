"""Bounded exact original-frame mask checks; does not certify variants or review content."""
import argparse
import asyncio
import shutil
from pathlib import Path
from scripts.vision.closed_exterior_material_capture import OUT,freeze,prior,read_record
from scripts.vision.build_closed_exterior_evidence import verified_run
from scripts.vision.verify_designed_full_scene_poses import replay_frame,DESIGN


async def main():
    p=freeze();dest=OUT/'original-replays';dest.mkdir(exist_ok=True);results=[];deps=[OUT/'protocol.json',Path(__file__)]
    for run in (r for r in p['runs'] if r['variant']=='original'):
        rows,mapping,cp=verified_run(run)
        for u in (u for u in p['units'] if u['run_key']==run['key']):
            row=rows[u['view_id']];ep=OUT/'evidence'/u['key']/'evidence.json';e=prior.read(ep);prior.verify(e)
            unit=dest/u['pair_id'];unit.mkdir(exist_ok=True);sp=unit/'source.json'
            if not sp.exists():prior.frozen(sp,dict(row,inputs={str(cp):prior.file_sha256(cp)}))
            f=dict(member_id=u['key'],lineage_id=u['pair_id'],class_name=u['category'],review_ids=['original'],
                source_image=row['rgb_path'],source_receipt=str(sp),source_plan=u['plan_path'],source_world=str(Path(u['plan_path']).parent/'world.sdf'),
                actual_pose=row['actual_pose'],world_name=read_record(u['plan_path'])['world_name'],instance_mapping=mapping,
                events=[dict(review_id='label-'+x['runtime_label'],runtime_label=int(x['runtime_label']),object_id=x['object_id'],bbox_xyxy=x['truth']['bbox_xyxy']) for x in e['events']])
            helper=unit/'gz_visibility_capture_cleanup_fixed'
            if not helper.exists():shutil.copy2(DESIGN.parent/'native-source-replay-v2/gz_visibility_capture_cleanup_fixed',helper)
            pp=unit/'protocol.json'
            if not pp.exists():prior.frozen(pp,dict(frame=f,training_ready=False,inputs={str(x):prior.file_sha256(x) for x in (cp,ep,sp,helper,Path(__file__))}))
            prior.verify(prior.read(pp));print('REPLAY',u['pair_id'],flush=True)
            rp,r=await replay_frame(f,unit);deps.extend([pp,rp])
            results.append(dict(pair_id=u['pair_id'],status=r['status'] if r else 'attempts_exhausted',receipt=str(rp),reason=r.get('reason') if r else 'missing'))
            print('REPLAY_RESULT',results[-1],flush=True)
    completion=dest/'completion.json'
    if not completion.exists():prior.frozen(completion,dict(status='original_checks_complete_not_variant_certification',results=results,inputs={str(x):prior.file_sha256(x) for x in deps}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');a=ap.parse_args()
    if a.replay:asyncio.run(main())
    else:print('NO_REPLAY_NO_TRAINING')
