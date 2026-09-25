"""Bounded native-renderer source replay. No training, label edits or deployment."""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import shutil
from scripts.vision.establish_material_view_candidates import OUT as SOURCE, prior
from scripts.vision import run_visibility_cleanup_validation as replay

OUT = SOURCE/'native-source-replay-v1'
PILOT = ('N04', 'N02')  # Visually reviewed control, then unboxed-blue identity question.


def freeze():
    rp=SOURCE/'source-review.json'; cp=SOURCE/'source-review-completion-v1.json'
    for p in (rp,cp,SOURCE/'source-inventory.json'):
        prior.verify(prior.read(p))
    dest=OUT/'protocol.json'
    if dest.exists():
        p=prior.read(dest);prior.verify(p);return p
    OUT.mkdir(parents=True,exist_ok=True)
    helper=prior.ROOT/'data/research/ml_training_recovery_v1/asset-visibility-replay-v1/gz_visibility_capture_cleanup_fixed'
    shutil.copy2(helper,OUT/helper.name)
    r=prior.read(rp);frames=[]
    paths=[rp,cp,SOURCE/'source-inventory.json',helper,OUT/helper.name,Path(__file__),Path(replay.__file__),
           prior.ROOT/'scripts/vision/instance_visibility_diagnosis.py']
    for sid in PILOT:
        s=next(x for x in r['sources'] if x['source_review_id']==sid)
        sp=OUT/(sid+'-source.json')
        prior.frozen(sp,dict(s['source_record'],inputs={s['source_capture']:prior.file_sha256(s['source_capture'])}))
        frames.append(dict(member_id=s['source_member_id'],lineage_id=s['lineage_id'],class_name=s['category'],
            review_ids=[sid],source_image=s['source_image'],source_receipt=str(sp),source_plan=s['source_plan'],
            source_world=s['source_world'],actual_pose=s['actual_pose'],world_name=s['world_name'],
            instance_mapping=s['instance_mapping'],events=[dict(review_id=e['event_id'],runtime_label=int(e['runtime_label']),
                object_id=e['object_id'],bbox_xyxy=e['truth']['bbox_xyxy']) for e in r['events'] if e['source_review_id']==sid]))
        paths += [sp,*[Path(s[k]) for k in ('source_image','source_capture','source_plan','source_world')]]
    return prior.frozen(dest,dict(status='two_source_replay_frozen',frames=frames,max_technical_attempts=3,
        policy='Native installed renderer; add only co-located panoptic sensor. No deployment, geometry/material/pose alteration, historical label edit or training. First control must pass before risk-frame replay. Semantic failures are not retried.',
        training_ready=False,training_started=False,inputs={str(p):prior.file_sha256(p) for p in paths}))


async def run():
    p=freeze()
    summary=OUT/'completion.json'
    if summary.exists():
        prior.verify(prior.read(summary));print('VERIFIED_EXISTING_COMPLETION');return
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original_out=replay.OUT; original_command=replay.command; oldlog=os.environ.get('GZ_LOG_PATH')
        replay.OUT=OUT
        logs=OUT/'runtime-logs';logs.mkdir(exist_ok=True);os.environ['GZ_LOG_PATH']=str(logs)
        async def clock_checked(*args,**kwargs):
            # Gazebo omits zero-valued seconds during startup; do not accept such a fence.
            if args[:3]!=('gz','topic','-e') or not any(str(a).endswith('/pose/info') for a in args):
                return await original_command(*args,**kwargs)
            for _ in range(6):
                raw=await original_command(*args,**dict(kwargs,timeout=5))
                stamp=json.loads(raw).get('header',{}).get('stamp',{})
                if stamp.get('sec',stamp.get('seconds',0))>=1:return raw
                await asyncio.sleep(.25)
            raise ValueError('Explicit post-move clock unavailable after six bounded samples')
        replay.command=clock_checked
        results=[];paths=[OUT/'protocol.json']
        try:
            for f in p['frames']:
                last=None;rp=None
                for n in range(1,4):
                    rp=OUT/'replay'/f['review_ids'][0]/f'attempt-{n:02}'/'receipt.json'
                    if rp.exists():
                        replay.verify_tree(rp);last=prior.read(rp)
                    elif rp.parent.exists():continue
                    else:last=await replay.attempt(f,n)
                    paths.append(rp)
                    if not last['process_cleanup_complete']:raise ValueError('Process cleanup incomplete')
                    if last['status']!='technical_failure':break
                results.append(dict(source_review_id=f['review_ids'][0],status=last['status'] if last else 'attempts_exhausted',
                    reason=last.get('reason','') if last else 'Incomplete attempts retained',receipt=str(rp)))
                if not last or last['status']!='original_pixel_evidence_certified':break
        finally:
            replay.OUT=original_out;replay.command=original_command
            if oldlog is None:os.environ.pop('GZ_LOG_PATH',None)
            else:os.environ['GZ_LOG_PATH']=oldlog
        return prior.frozen(summary,dict(status='pilot_complete_review_required' if len(results)==2 and all(r['status']=='original_pixel_evidence_certified' for r in results) else 'pilot_blocked_no_expansion',
            results=results,training_ready=False,training_started=False,
            inputs={str(p):prior.file_sha256(p) for p in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');a=ap.parse_args()
    if a.replay:asyncio.run(run())
    else:freeze();print('FROZEN_NO_REPLAY_NO_TRAINING')
