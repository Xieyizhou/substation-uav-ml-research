"""One prespecified saved-frame replay control, no training or asset edits."""
import argparse
import asyncio
import fcntl
import os
from pathlib import Path
import shutil
from scripts.vision.test_body_material_applicability import OUT as SOURCE,PRIOR,ROOT,read,verify,frozen,file_sha256
import scripts.vision.run_visibility_cleanup_validation as replay

# Flat independent directory avoids macOS PATH_MAX after many historical stages.
# Frozen input hashes, not pathname nesting, carry lineage.
OUT=ROOT/'data/research/ml_training_recovery_v1/asset-visibility-replay-v1'


def select(diagnosis,review):
    ds={d['review_id']:d for d in review['decisions']}
    rows=[r for r in diagnosis['training'] if r['class_name']=='capacitor_bank' and not r['gaps']
          and ds[r['review_id']]['decision']=='condition_recorded'
          and ds[r['review_id']]['occlusion_condition']=='no_obvious_foreground_occlusion'
          and min(r['actual_exposures'].values())>0]
    if not rows:raise ValueError('No eligible saved source')
    return sorted(rows,key=lambda r:r['member_id'])[0]


def freeze():
    path=OUT/'protocol.json'
    if path.exists():verify(read(path));return read(path)
    d=read(SOURCE/'diagnosis.json');rv=read(PRIOR/'review.json')
    for p in (SOURCE/'diagnosis.json',SOURCE/'completion.json',PRIOR/'review.json',PRIOR/'evidence.json'):verify(read(p))
    chosen=select(d,rv);cp=Path(chosen['capture_path']);receipt=read(cp)
    vid=Path(chosen['source_image_path']).parent.name
    views=[v for v in receipt['views'] if v['view_id']==vid]
    if len(views)!=1:raise ValueError('Ambiguous source view')
    view=views[0];wp=Path(chosen['world_path']);pp=wp.parent/'plan.json';plan=read(pp)
    ev=next(r for r in read(PRIOR/'evidence.json')['training'] if r['review_id']==chosen['review_id'])
    if file_sha256(wp)!=chosen['world_sha256'] or view['image_sha256']!=chosen['source_image_sha256']:raise ValueError('Stale selected source')
    OUT.mkdir(exist_ok=True)
    sp=OUT/'source-view.json';frozen(sp,{**view,'inputs':{str(cp):file_sha256(cp)}})
    binary=replay.OUT/'gz_visibility_capture_cleanup_fixed';dest=OUT/binary.name
    if dest.exists():raise ValueError('Unfrozen binary exists')
    shutil.copy2(binary,dest)
    frame=dict(member_id=chosen['member_id'],review_ids=[chosen['review_id']],source_image=chosen['source_image_path'],
        source_receipt=str(sp),source_plan=str(pp),source_world=str(wp),actual_pose=view['actual_pose'],
        world_name=plan['world_name'],instance_mapping=receipt['collection_checks']['instance_mapping'],
        events=[dict(review_id=chosen['review_id'],runtime_label=int(chosen['runtime_label']),
            object_id=chosen['object_id'],bbox_xyxy=ev['target']['truth']['bbox_xyxy'])])
    paths=[SOURCE/'diagnosis.json',SOURCE/'completion.json',PRIOR/'review.json',PRIOR/'evidence.json',
        Path(__file__),Path(replay.__file__),ROOT/'scripts/vision/instance_visibility_diagnosis.py',
        binary,dest,ROOT/'tools/gz_visibility_capture_cleanup_fixed.cc',cp,sp,pp,wp,
        pp.parent/'obstacles.json',Path(frame['source_image'])]
    return frozen(path,dict(status='control_frozen_before_render',frames=[frame],max_attempts=3,
        selection_rule='Lexicographically first member among reviewed capacitor targets with no obvious foreground occlusion, no provenance gap, positive exposure in every seed; no model score selection.',
        control_policy='Only add RGB-aligned panoptic sensor; saved actual pose, original geometry and appearance unchanged. Require exact RGB, <=1px boxes, <=.05m/1deg pose, <=33.334ms, first three internally stable frames.',
        subsequent_diagnostic_policy='Only after valid control: separately bound diagnostic copy may remove selected capacitor body VISUAL only, leaving collision, cylinders, labels, world and camera otherwise intact. No original-frame certification or training use for modified frames.',
        training_started=False,training_ready=False,inputs={str(p):file_sha256(p) for p in paths}))


async def run():
    p=freeze();dest=OUT/'control-completion.json'
    if dest.exists():verify(read(dest));print('VALID_CONTROL_COMPLETION_REUSED');return
    with (OUT/'control.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=replay.OUT;logs=os.environ.get('GZ_LOG_PATH');replay.OUT=OUT
        logpath=OUT/'runtime-logs';logpath.mkdir(exist_ok=True);os.environ['GZ_LOG_PATH']=str(logpath)
        paths=[OUT/'protocol.json'];last=None
        try:
            for n in range(1,4):
                rp=OUT/'replay'/p['frames'][0]['review_ids'][0]/f'attempt-{n:02}'/'receipt.json'
                if rp.exists():r=read(rp);verify(r)
                elif rp.parent.exists():continue
                else:r=await replay.attempt(p['frames'][0],n)
                paths.append(rp);last=r
                if not r['process_cleanup_complete']:raise ValueError('Process cleanup incomplete')
                if r['status']!='technical_failure':break
        finally:
            replay.OUT=previous
            if logs is None:os.environ.pop('GZ_LOG_PATH',None)
            else:os.environ['GZ_LOG_PATH']=logs
        passed=bool(last and last['status']=='original_pixel_evidence_certified')
        frozen(dest,dict(status='control_passed_diagnostic_not_started' if passed else 'control_blocked_no_asset_intervention',
            control_passed=passed,last_status=last['status'] if last else 'incomplete_attempts_exhausted',
            reason=last.get('reason','') if last else 'No complete attempt',training_started=False,training_ready=False,
            diagnostic_started=False,inputs={str(p):file_sha256(p) for p in paths}))
        print(read(dest)['status'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');args=ap.parse_args()
    if args.replay:asyncio.run(run())
    else:freeze();print('FROZEN_NO_REPLAY_NO_TRAINING')
