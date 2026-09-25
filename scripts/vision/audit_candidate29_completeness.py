"""Bounded 29-source completeness audit; semantic findings accumulate, never train."""
import argparse,asyncio,fcntl,shutil
from pathlib import Path
from scripts.vision import run_physical_lighting_capture_v2 as capture
from scripts.vision import optimize_revision_compensation as counts
from scripts.vision.replay_l05_increase_queue import OUT as INCREASE
from scripts.vision.prepare_physical_lighting_control import history,prior

OUT=prior.ROOT/'data/research/ml_training_recovery_v1/candidate29-completeness-audit-v1'

def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():prior.verify(prior.read(dest));return prior.read(dest)
    cp=counts.prior.OUT/'counts.json';c=prior.read(cp);prior.verify(c)
    candidates=c['reviewed_increase_candidates']
    if len(candidates)!=29 or len(set(candidates))!=29:raise ValueError('Candidate scope changed')
    paths=[cp,Path(__file__).resolve(),Path(capture.__file__)];sources={};reusable={}
    for root in (history.CONTROL,history.SECOND):
        ep=root/'evidence.json';e=prior.read(ep);prior.verify(e);paths.append(ep)
        for f in e['events']:
            mid=f['member']['member_id']
            if mid in sources:raise ValueError('Ambiguous source')
            sources[mid]=f
    for root,name in ((capture.OUT,'capture-receipt.json'),(INCREASE,'receipt.json')):
        pp,rp=root/'protocol.json',root/name;p,r=prior.read(pp),prior.read(rp)
        prior.verify(p);prior.verify(r);paths += [pp,rp]
        frames={f['pair_id']:f for f in p['frames'] if f['variant']=='original'}
        for result in r['results']:
            if result.get('variant','original')!='original':continue
            f=frames[result['pair_id']];receipt=Path(result['receipt']);v=prior.read(receipt);prior.verify(v)
            if not v['process_cleanup_complete']:raise ValueError('Unclean old attempt')
            reusable[f['member_id']]=dict(frame=f,receipt=str(receipt));paths.append(receipt)
    OUT.mkdir(exist_ok=True);frames=[]
    from scripts.vision.test_body_material_applicability import equal_rgb,label_correspondence,model_mapping
    from scripts.vision.structure_fit import truth_for
    from src.vision.canonical.gates import instance_mapping,annotation_mode_from_world
    for i,mid in enumerate(sorted(candidates),1):
        s=sources[mid];plan=prior.read(s['source_plan']);rec=prior.read(s['source_receipt']);row=s['member']
        for kind in ('image','label'):
            path=Path(row[kind+'_path'])
            if prior.file_sha256(path)!=row[kind+'_sha256']:raise ValueError('Member identity changed')
            paths.append(path)
        equal_rgb(s['source_image'],row['image_path']);label_correspondence(truth_for(row),rec['truth']['objects'])
        if prior.file_sha256(Path(s['source_world']))!=plan['files']['world.sdf'] or annotation_mode_from_world(Path(s['source_world']))!='full_2d':raise ValueError('Source world/mode conflict')
        mapping={str(k):v for k,v in instance_mapping(plan).items()};wm,_=model_mapping(capture.base.ET.parse(s['source_world']))
        if len({v['object_id'] for v in mapping.values()})!=len(mapping) or any(wm.get(k)!=v['object_id'] for k,v in mapping.items()):raise ValueError('Source instance ambiguity')
        pid=f'A{i:02}';frame=dict(pair_id=pid,variant='original',review_ids=[pid+'-original'],member_id=mid,
            source_image=s['source_image'],source_receipt=s['source_receipt'],source_plan=s['source_plan'],source_world=s['source_world'],
            world_name=plan['world_name'],actual_pose=rec['actual_pose'],instance_mapping=mapping,member=row)
        if mid in reusable:
            old=reusable[mid]
            if old['frame']['source_image']!=s['source_image'] or old['frame']['source_receipt']!=s['source_receipt']:raise ValueError('Reuse source mismatch')
            frame['reused_receipt']=old['receipt'];frame['reused_frame']=old['frame']
        frames.append(frame);paths += [Path(s[k]) for k in ('source_image','source_receipt','source_plan','source_world')]
    helper=OUT/'gz_visibility_capture_cleanup_fixed';shutil.copy2(capture.DEPTH/helper.name,helper);paths.append(helper)
    return prior.frozen(dest,dict(status='29_sources_frozen',frames=frames,max_attempts=3,
        authorization='Read-only original-frame diagnosis; collect semantic findings across the frozen candidate scope; no solving, relabeling or training.',
        training_ready=False,training_started=False,inputs={str(x):prior.file_sha256(x) for x in paths}))

async def run():
    p=freeze();base=capture.base;replay=base.replay;base.guard();paths=[OUT/'protocol.json'];results=[]
    with (capture.BUILD/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=(capture.OUT,base.OUT,base.VARIANT,replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command)
        capture.OUT=OUT;base.OUT,base.VARIANT=capture.DEPTH,'depth'
        replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command=OUT,base.dual_sensor,base.validate_dual,capture.analyze,capture.PrefixAsync(),capture.clock.command
        try:
            for f in p['frames']:
                if 'reused_receipt' in f:
                    rp=Path(f['reused_receipt']);last=prior.read(rp);prior.verify(last)
                else:
                    last=None
                    for n in range(1,4):
                        capture.clock.CURRENT=OUT/'replay'/f['review_ids'][0]/f'attempt-{n:02}';rp=capture.clock.CURRENT/'receipt.json'
                        if rp.exists():last=prior.read(rp);prior.verify(last)
                        elif capture.clock.CURRENT.exists():continue
                        else:last=await replay.attempt(f,n)
                        if not last['process_cleanup_complete']:raise ValueError('Process cleanup incomplete')
                        if last['status']!='technical_failure':break
                    if last is None:raise ValueError('No valid attempt within cap')
                paths.append(rp);results.append(dict(pair_id=f['pair_id'],member_id=f['member_id'],receipt=str(rp),reused='reused_receipt' in f,status=last['status'],reason=last.get('reason')))
                print(f['pair_id'],last['status'],'reused' if 'reused_receipt' in f else 'new',flush=True)
        finally:(capture.OUT,base.OUT,base.VARIANT,replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command)=old
    base.guard();prior.frozen(OUT/'replay-index.json',dict(status='all_sources_have_replay_result_not_quality_approval',results=results,training_ready=False,
        inputs={str(x):prior.file_sha256(x) for x in paths}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if a.run:asyncio.run(run())
    else:p=freeze();print('FROZEN',len(p['frames']),'REUSE',sum('reused_receipt' in f for f in p['frames']))
