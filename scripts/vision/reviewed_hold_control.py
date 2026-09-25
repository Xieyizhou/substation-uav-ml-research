"""Independent quality gate for the authorized continuous control experiment."""
import argparse,asyncio,fcntl,shutil
from pathlib import Path
from scripts.vision.optimize_revision_compensation import OUT as COUNTS,prior
from scripts.vision.replay_revision_siblings import base,DEPTH,BUILD,clock,regression,PrefixAsync
from scripts.vision.test_body_material_applicability import model_mapping,equal_rgb,label_correspondence
from scripts.vision.structure_fit import truth_for
from src.vision.canonical.gates import instance_mapping,annotation_mode_from_world

OUT=prior.ROOT/'data/research/ml_training_recovery_v1/reviewed-hold-compensation-control-v1'


def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():prior.verify(prior.read(dest));return prior.read(dest)
    cp=COUNTS/'counts.json';prior.verify(prior.read(cp));p=prior.read(cp)
    paths=[cp,Path(__file__),DEPTH/'completion.json',Path(clock.__file__),Path(regression.__file__)]
    prior.verify(prior.read(paths[2]));sources={}
    for root in (prior.CONTROL,prior.SECOND):
        ep=root/'evidence.json';prior.verify(prior.read(ep));paths.append(ep)
        for f in prior.read(ep)['events']:
            if f['member']['member_id'] in sources:raise ValueError('Duplicate source evidence')
            sources[f['member']['member_id']]=f
    OUT.mkdir(parents=True,exist_ok=True);frames=[]
    for i,q in enumerate(p['review_queue'],1):
        r=q['member'];f=sources[r['member_id']];rec=prior.read(f['source_receipt']);plan=prior.read(f['source_plan']);wp=Path(f['source_world'])
        for kind in ('image','label'):
            if prior.file_sha256(Path(r[kind+'_path']))!=r[kind+'_sha256']:raise ValueError('Pool identity changed')
        equal_rgb(f['source_image'],r['image_path']);label_correspondence(truth_for(r),rec['truth']['objects'])
        if prior.file_sha256(wp)!=plan['files']['world.sdf'] or annotation_mode_from_world(wp)!='full_2d':raise ValueError('World identity or mode mismatch')
        mapping=instance_mapping(plan);worldmap,_=model_mapping(base.ET.parse(wp))
        if len({v['object_id'] for v in mapping.values()})!=len(mapping):raise ValueError('Mapping collision')
        for k,v in mapping.items():
            if worldmap.get(str(k))!=v['object_id']:raise ValueError('World mapping mismatch')
        frames.append(dict(member_id=r['member_id'],review_ids=[f'Q{i:02}'],source_receipt=f['source_receipt'],source_plan=f['source_plan'],
            source_world=f['source_world'],source_image=f['source_image'],actual_pose=rec['actual_pose'],world_name=plan['world_name'],
            instance_mapping=mapping,source_evidence=f,source_trace_nature='posthoc_not_original_gate_certification'))
        paths += [Path(f[k]) for k in ('source_receipt','source_plan','source_world','source_image')]+[wp.parent/'obstacles.json',Path(r['image_path']),Path(r['label_path'])]
    helper=OUT/'gz_visibility_capture_cleanup_fixed';shutil.copy2(DEPTH/helper.name,helper);paths.append(helper)
    return prior.frozen(dest,dict(status='seven_member_quality_gate_frozen',frames=frames,max_attempts=3,max_resolves=1,
        counts_identity=p['identity'],training_ready=False,training_started=False,
        inputs={str(p):prior.file_sha256(p) for p in paths}))


async def run():
    p=freeze();base.guard();replay=base.replay
    with (BUILD/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=(base.OUT,base.VARIANT,replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command)
        base.OUT,base.VARIANT=DEPTH,'depth'
        replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command=OUT,base.dual_sensor,base.validate_dual,regression.analyze,PrefixAsync(),clock.command
        results=[];paths=[OUT/'protocol.json']
        try:
            for f in p['frames']:
                for n in range(1,4):
                    clock.CURRENT=OUT/'replay'/f['review_ids'][0]/f'attempt-{n:02}';rp=clock.CURRENT/'receipt.json'
                    if rp.exists():last=base.read(rp);base.verify(last)
                    elif clock.CURRENT.exists():raise ValueError('Incomplete attempt preserved; cannot reuse')
                    else:last=await replay.attempt(f,n)
                    paths.append(rp)
                    if not last['process_cleanup_complete']:raise ValueError('Cleanup incomplete')
                    if last['status']!='technical_failure':break
                results.append(dict(member_id=f['member_id'],review_id=f['review_ids'][0],receipt=str(rp),status=last['status'],reason=last.get('reason')))
        finally:(base.OUT,base.VARIANT,replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command)=old
        base.guard();dest=OUT/'replay-receipt.json'
        if dest.exists():prior.verify(prior.read(dest));return
        prior.frozen(dest,dict(status='replay_evidence_collected',results=results,training_ready=False,training_started=False,inputs={str(x):prior.file_sha256(x) for x in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');args=ap.parse_args()
    if args.run:asyncio.run(run())
    else:freeze();print('QUALITY_PREFLIGHT_ONLY_NO_TRAINING')
