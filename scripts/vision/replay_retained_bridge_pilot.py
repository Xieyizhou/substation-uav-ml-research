"""Six-frame deterministic source pilot; never grants training admission."""
import asyncio
import fcntl
import shutil
from pathlib import Path
from scripts.vision.trace_retained_bridge_sources import OUT as SOURCE, read, save, file_sha256, verify_tree, unique
import scripts.vision.run_visibility_cleanup_validation as replay

OUT=SOURCE/'instance-replay-pilot-v1'
HELPER=replay.OUT/'gz_visibility_capture_cleanup_fixed'

def select(frames):
    originals=sorted((f for f in frames if f['variant']=='original' and f['status']=='source_verified_replay_pending'),key=lambda f:f['lineage_id'])
    edge=next(f for f in originals if any(o['category']=='switchgear' and o['boundary_contact_within_one_pixel'] for o in f['objects']))
    control=next(f for f in originals if any(o['category']=='switchgear' for o in f['objects']) and not any(o['boundary_contact_within_one_pixel'] for o in f['objects']))
    chosen=sorted((f for f in frames if f['lineage_id'] in {edge['lineage_id'],control['lineage_id']}),key=lambda f:(f['lineage_id'],f['variant']))
    if len(chosen)!=6 or any(f['status']!='source_verified_replay_pending' for f in chosen):raise ValueError('Incomplete pilot groups')
    return chosen

def prepare():
    path=OUT/'protocol.json'
    if path.exists():verify_tree(path);return read(path)
    verify_tree(SOURCE/'validation.json')
    chosen=select(read(SOURCE/'trace.json')['frames']);OUT.mkdir(parents=True,exist_ok=True)
    shutil.copy2(HELPER,OUT/HELPER.name)
    inputs={str(p):file_sha256(p) for p in (SOURCE/'validation.json',Path(__file__),Path(replay.__file__),HELPER,OUT/HELPER.name)}
    frames=[];ordinal=0
    for f in chosen:
        receipt=read(f['receipt_path']);row=unique(receipt['views'],'view_id')[f['view_id']]
        pp=Path(f['plan_path']);plan=read(pp);rp=OUT/(f['view_id']+'-source.json')
        save(rp,{**row,'inputs':{f['receipt_path']:file_sha256(f['receipt_path'])}})
        events=[]
        for obj in f['objects']:
            ordinal+=1
            events.append(dict(review_id=f'B{ordinal:03}',runtime_label=obj['instance_label'],object_id=obj['device_id'],category=obj['category'],bbox_xyxy=obj['bbox_xyxy']))
        frames.append(dict(member_id=f['member_id'],lineage_id=f['lineage_id'],variant=f['variant'],
            source_image=row['rgb_path'],source_receipt=str(rp),source_plan=str(pp),source_world=str(pp.parent/'world.sdf'),
            actual_pose=row['actual_pose'],world_name=plan['world_name'],instance_mapping=f['instance_mapping'],events=events,review_ids=[e['review_id'] for e in events]))
        inputs[str(rp)]=file_sha256(rp)
    return save(path,dict(status='frozen_before_replay',frames=frames,inputs=inputs,max_attempts=3,
        selection_rule='Lexicographically first original lineage with boundary switchgear and first with switchgear but no boundary boxes; all three variants; no model scores.',
        scope='Provenance and instance visibility only; all original labels immutable; no training or automatic review decisions.'))

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
                inputs[str(rp)]=file_sha256(rp)
                result=dict(member_id=frame['member_id'],receipt_path=str(rp),status=r['status'])
                if r['status']!='technical_failure':break
            results.append(result or dict(member_id=frame['member_id'],status='attempts_exhausted_incomplete'))
            save(OUT/'progress.json',dict(status='replay_complete_pending_review' if len(results)==len(p['frames']) else 'in_progress',frames=results,inputs=inputs))
        print('PILOT_END',[(r['member_id'],r['status']) for r in results],flush=True)

if __name__=='__main__':asyncio.run(main())
