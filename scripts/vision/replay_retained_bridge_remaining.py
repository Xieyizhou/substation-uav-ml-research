"""Complete retained-source replay; historical pilot remains immutable."""
import asyncio
import fcntl
import shutil
from pathlib import Path
from scripts.vision.replay_retained_bridge_pilot import OUT as PILOT, SOURCE, HELPER, read,save,file_sha256,verify_tree,unique
import scripts.vision.run_visibility_cleanup_validation as replay

OUT=SOURCE/'instance-replay-remaining-v1'

def remaining(trace,pilot):
    excluded={f['member_id'] for f in pilot['frames']}
    rows=sorted((f for f in trace['frames'] if f['member_id'] not in excluded),key=lambda f:(f['lineage_id'],f['variant']))
    if len(rows)!=33 or len({f['lineage_id'] for f in rows})!=11:raise ValueError('Remaining scope mismatch')
    for lineage in {f['lineage_id'] for f in rows}:
        group=[f for f in rows if f['lineage_id']==lineage]
        if len(group)!=3 or {f['variant'] for f in group}!={'original','neutral_bridge','background_bridge'}:raise ValueError('Incomplete variants')
    if any(f['status']!='source_verified_replay_pending' for f in rows):raise ValueError('Unverified source')
    return rows

def prepare():
    path=OUT/'protocol.json'
    if path.exists():verify_tree(path);return read(path)
    verify_tree(PILOT/'completion.json')
    chosen=remaining(read(SOURCE/'trace.json'),read(PILOT/'protocol.json'))
    OUT.mkdir(parents=True,exist_ok=True);shutil.copy2(HELPER,OUT/HELPER.name)
    inputs={str(p):file_sha256(p) for p in (PILOT/'completion.json',SOURCE/'trace.json',Path(__file__),Path(replay.__file__),HELPER,OUT/HELPER.name)}
    frames=[];ordinal=0
    for f in chosen:
        receipt=read(f['receipt_path']);row=unique(receipt['views'],'view_id')[f['view_id']]
        pp=Path(f['plan_path']);plan=read(pp);rp=OUT/(f['view_id']+'-source.json')
        save(rp,{**row,'inputs':{f['receipt_path']:file_sha256(f['receipt_path'])}})
        events=[]
        for obj in f['objects']:
            ordinal+=1;events.append(dict(review_id=f'M{ordinal:03}',runtime_label=obj['instance_label'],object_id=obj['device_id'],category=obj['category'],bbox_xyxy=obj['bbox_xyxy']))
        frames.append(dict(member_id=f['member_id'],lineage_id=f['lineage_id'],variant=f['variant'],source_image=row['rgb_path'],source_receipt=str(rp),source_plan=str(pp),source_world=str(pp.parent/'world.sdf'),actual_pose=row['actual_pose'],world_name=plan['world_name'],instance_mapping=f['instance_mapping'],events=events,review_ids=[e['review_id'] for e in events]))
        inputs[str(rp)]=file_sha256(rp)
    if ordinal!=135:raise ValueError('Full instance scope changed')
    return save(path,dict(status='frozen_before_replay',frames=frames,inputs=inputs,max_attempts=3,
        scope='Remaining 33 frames / 11 lineages / 135 boxes; no label changes, collection for training, training or sealed testing.',
        selection_rule='All source-verified members excluding immutable six-frame pilot, sorted by lineage and variant.'))

async def main():
    p=prepare();replay.OUT=OUT
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        results=[];inputs={str(OUT/'protocol.json'):file_sha256(OUT/'protocol.json')}
        for frame in p['frames']:
            result=None
            for number in range(1,4):
                rp=OUT/'replay'/frame['review_ids'][0]/f'attempt-{number:02}'/'receipt.json'
                if rp.exists():verify_tree(rp);r=read(rp)
                elif rp.parent.exists():continue
                else:r=await replay.attempt(frame,number)
                inputs[str(rp)]=file_sha256(rp);result=dict(member_id=frame['member_id'],receipt_path=str(rp),status=r['status'])
                if r['status']!='technical_failure':break
            results.append(result or dict(member_id=frame['member_id'],status='attempts_exhausted_incomplete'))
            save(OUT/'progress.json',dict(status='replay_complete_pending_review' if len(results)==len(p['frames']) else 'in_progress',frames=results,inputs=inputs))
        print('REMAINING_END',len(results),flush=True)

if __name__=='__main__':asyncio.run(main())
