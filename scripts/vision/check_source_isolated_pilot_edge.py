"""Explicit same-frame replay of a visually unresolved edge; no label changes."""
import argparse
import asyncio
import shutil
from pathlib import Path
from scripts.vision.source_isolated_material_capture import OUT, prior, freeze, verify_capture, read_record
from scripts.vision.verify_designed_full_scene_poses import replay_frame, DESIGN


async def main():
    p=freeze();u=next(x for x in p['units'] if x['key']=='layout-A:reactor:1:original')
    cp=OUT/'captures'/u['key']/'collection-receipt.json';r=read_record(cp);row=verify_capture(u,cp)
    ep=OUT/'evidence'/u['key']/'evidence.json';e=prior.read(ep);prior.verify(e)
    dest=OUT/'edge-replay-v1';dest.mkdir(exist_ok=True)
    sp=dest/'source.json'
    if not sp.exists():prior.frozen(sp,dict(row,inputs={str(cp):prior.file_sha256(cp)}))
    f=dict(member_id=u['key'],lineage_id=u['pair_id'],class_name='reactor',review_ids=['edge-pilot'],
        source_image=row['rgb_path'],source_receipt=str(sp),source_plan=u['plan_path'],
        source_world=str(Path(u['plan_path']).parent/'world.sdf'),actual_pose=row['actual_pose'],
        world_name=read_record(u['plan_path'])['world_name'],instance_mapping=r['collection_checks']['instance_mapping'],
        events=[dict(review_id='label-'+x['runtime_label'],runtime_label=int(x['runtime_label']),object_id=x['object_id'],bbox_xyxy=x['truth']['bbox_xyxy']) for x in e['events']])
    helper=dest/'gz_visibility_capture_cleanup_fixed'
    if not helper.exists():shutil.copy2(DESIGN.parent/'native-source-replay-v2/gz_visibility_capture_cleanup_fixed',helper)
    pp=dest/'protocol.json'
    if not pp.exists():prior.frozen(pp,dict(frame=f,training_ready=False,
        inputs={str(path):prior.file_sha256(path) for path in (cp,sp,ep,helper,Path(__file__))}))
    prior.verify(prior.read(pp))
    rp,result=await replay_frame(f,dest)
    completion=dest/'completion.json'
    if not completion.exists():prior.frozen(completion,dict(status=result['status'] if result else 'attempts_exhausted',
        reason=result.get('reason') if result else 'missing_result',replay_receipt=str(rp),
        inputs={str(path):prior.file_sha256(path) for path in (pp,rp)}))
    print(prior.read(completion)['status'])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');a=ap.parse_args()
    if a.replay:asyncio.run(main())
    else:print('NO_REPLAY_NO_TRAINING')
