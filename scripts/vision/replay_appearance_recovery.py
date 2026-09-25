"""Reuse the verified cleanup-fixed panoptic replay for nine held pilot frames."""
import asyncio
import fcntl
import shutil
from pathlib import Path
from scripts.vision.prepare_appearance_recovery_candidates import OUT as SOURCE,read,save,file_sha256,verify_tree,ROOT
import scripts.vision.run_visibility_cleanup_validation as replay

OUT=SOURCE/'instance-replay-v1'
VERIFIED_HELPER=replay.OUT/'gz_visibility_capture_cleanup_fixed'

def prepare():
    path=OUT/'protocol.json'
    if path.exists():verify_tree(path);return read(path)
    verify_tree(SOURCE/'pilot-review-handoff.json');verify_tree(replay.OUT/'protocol.json')
    decisions=read(SOURCE/'review/decisions.json');planruns=read(SOURCE/'pilot-protocol.json')['runs'];captures=read(SOURCE/'capture-progress.json')['runs']
    plans={r['variant']:r['plan_path'] for r in planruns};receipts={r['variant']:r['receipt_path'] for r in captures}
    OUT.mkdir(parents=True,exist_ok=True);shutil.copyfile(VERIFIED_HELPER,OUT/'gz_visibility_capture_cleanup_fixed');shutil.copymode(VERIFIED_HELPER,OUT/'gz_visibility_capture_cleanup_fixed')
    frames=[];inputs={};ordinal=0
    for f in decisions['frames']:
        held=[o for o in f['objects'] if o['review_status']=='content_risk_pending_replay']
        if not held:continue
        cp=Path(receipts[f['variant']]);capture=read(cp);rows=[r for r in capture['views'] if r['view_id']==f['view_id']]
        if len(rows)!=1:raise ValueError('Missing/ambiguous source view')
        row=rows[0];pp=Path(plans[f['variant']]);plan=read(pp)
        if file_sha256(row['rgb_path'])!=row['image_sha256']:raise ValueError('Changed original image')
        rp=OUT/(f['view_id']+'-source.json');save(rp,{**row,'inputs':{str(cp):file_sha256(cp)}})
        events=[]
        for o in held:
            ordinal+=1;events.append(dict(review_id=f'P{ordinal:03}',runtime_label=int(o['runtime_label']),bbox_xyxy=o['bbox_xyxy'],object_id=o['object_id'],initial_reason=o['reason']))
        frames.append(dict(member_id=f['view_id'],variant=f['variant'],category=f['category'],source_image=row['rgb_path'],source_plan=str(pp),source_world=str(pp.parent/'world.sdf'),source_receipt=str(rp),actual_pose=row['actual_pose'],world_name=plan['world_name'],instance_mapping=capture['collection_checks']['instance_mapping'],events=events,review_ids=[e['review_id'] for e in events]))
        for p in (cp,pp,rp,pp.parent/'world.sdf',Path(row['rgb_path'])):inputs[str(p)]=file_sha256(p)
    if len(frames)!=9 or ordinal!=21:raise ValueError('Unexpected held-frame scope')
    for p in (SOURCE/'pilot-review-handoff.json',Path(__file__),Path(replay.__file__),VERIFIED_HELPER,OUT/'gz_visibility_capture_cleanup_fixed'):inputs[str(p)]=file_sha256(p)
    return save(path,dict(status='frozen_before_replay',frames=frames,max_attempts=3,inputs=inputs,policy='Unchanged original pose/world; only colocated panoptic sensor added. First internally stable three RGB/mask frames. Exact original RGB and <=1px box delta required; uncertain component identity remains unknown.'))

async def main():
    p=prepare();replay.OUT=OUT
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);results=[];inputs={str(OUT/'protocol.json'):file_sha256(OUT/'protocol.json')}
        for frame in p['frames']:
            verify_tree(OUT/'protocol.json');result=None
            for n in range(1,4):
                rp=OUT/'replay'/frame['review_ids'][0]/f'attempt-{n:02}'/'receipt.json'
                if rp.exists():verify_tree(rp);r=read(rp)
                elif rp.parent.exists():continue
                else:r=await replay.attempt(frame,n)
                inputs[str(rp)]=file_sha256(rp);result=dict(member_id=frame['member_id'],review_ids=frame['review_ids'],receipt_path=str(rp),status=r['status'])
                if r['status']!='technical_failure':break
            if result is None:result=dict(member_id=frame['member_id'],status='attempts_exhausted_incomplete',review_ids=frame['review_ids'])
            results.append(result)
            save(OUT/'progress.json',dict(status='replay_complete_pending_review' if len(results)==9 and all(r['status']=='original_pixel_evidence_certified' for r in results) else 'in_progress_or_held',frames=results,inputs=inputs))
        print('REPLAY_END',[(r['review_ids'],r['status']) for r in results],flush=True)

if __name__=='__main__':asyncio.run(main())
