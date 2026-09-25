"""Whole-source risk containment and bounded N05 triplet expansion. No training."""
import argparse,asyncio,copy,fcntl,json,os,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.vision.establish_material_view_candidates import OUT as SOURCE,prior
from scripts.vision.build_material_view_world_drafts import valid_clock
from scripts.vision import run_visibility_cleanup_validation as replay

OUT=SOURCE/'N05-expansion-v1'


def mask_gate(result,folder,reference):
    if result['stable_frames']!=3 or any(r['maximum_box_delta_px']>1 for r in result['records']):
        raise ValueError('Unstable frame or shifted label')
    for n in range(1,4):
        if prior.file_sha256(folder/f'first-stable-window/frame-{n}-mask.bin')!=prior.file_sha256(reference):
            raise ValueError('Instance mask changed')


def freeze():
    paths=[SOURCE/'source-review.json',SOURCE/'source-review-completion-v1.json',SOURCE/'world-drafts-v1/manifest.json',
           SOURCE/'N04-material-capture-v1/reviewed-completion.json']
    for p in paths:prior.verify(prior.read(p))
    OUT.mkdir(exist_ok=True);dest=OUT/'protocol.json'
    if dest.exists():p=prior.read(dest);prior.verify(p);return p
    review=prior.read(paths[0]);s=next(s for s in review['sources'] if s['source_review_id']=='N05')
    sp=OUT/'N05-source.json';prior.frozen(sp,dict(s['source_record'],inputs={s['source_capture']:prior.file_sha256(s['source_capture'])}))
    frame=dict(member_id=s['source_member_id'],lineage_id=s['lineage_id'],class_name=s['category'],review_ids=['N05'],
        source_image=s['source_image'],source_receipt=str(sp),source_plan=s['source_plan'],source_world=s['source_world'],
        actual_pose=s['actual_pose'],world_name=s['world_name'],instance_mapping=s['instance_mapping'],
        events=[dict(review_id=e['event_id'],runtime_label=int(e['runtime_label']),object_id=e['object_id'],bbox_xyxy=e['truth']['bbox_xyxy']) for e in review['events'] if e['source_review_id']=='N05'])
    n04=next(s for s in review['sources'] if s['source_review_id']=='N04')
    helper=SOURCE/'native-source-replay-v2/gz_visibility_capture_cleanup_fixed'
    paths += [sp,helper,Path(__file__),Path(replay.__file__),prior.ROOT/'scripts/vision/build_material_view_world_drafts.py']
    paths += [Path(frame[k]) for k in ('source_image','source_plan','source_world')]
    return prior.frozen(dest,dict(status='risk_contained_one_source_expansion_frozen',frame=frame,
        held_source=dict(source_review_id='N04',lineage_id=n04['lineage_id'],variants=['original','warm','cool'],
            decision='whole_source_held_no_training_export',review_type='AI辅助审核',
            reason='Nonplanned transformer_sw has only broad truncated side and base, without bushings or sufficient discriminative content. Exact instance mask confirms presence, not adequate supervision. Hold all same-pose variants; no label deletion, ignore mask or historical edit.'),
        selection='N05 was the other already-frozen visually reviewed source. Not chosen using model scores; N04 is not replaced or counted as solved. Other ten sources remain held.',
        variants=[r for r in prior.read(paths[2])['rows'] if r['source_review_id']=='N05'],
        helper=str(helper),max_attempts_per_unit=3,training_ready=False,training_started=False,
        historical_admission_chain_valid=False,inputs={str(p):prior.file_sha256(p) for p in paths}))


async def unit(frame,variant,world=None,reference=None):
    folder=OUT/variant;folder.mkdir(exist_ok=True);helper=folder/'gz_visibility_capture_cleanup_fixed'
    if not helper.exists():shutil.copy2(prior.read(OUT/'protocol.json')['helper'],helper)
    pp=folder/'protocol.json'
    deps=[OUT/'protocol.json',helper]
    if world:deps.append(Path(world))
    if reference:deps.append(reference)
    if not pp.exists():prior.frozen(pp,dict(variant=variant,training_ready=False,inputs={str(p):prior.file_sha256(p) for p in deps}))
    prior.verify(prior.read(pp))
    saved=(replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.command);oldlog=os.environ.get('GZ_LOG_PATH')
    replay.OUT=folder;logs=folder/'runtime-logs';logs.mkdir(exist_ok=True);os.environ['GZ_LOG_PATH']=str(logs)
    if world:
        tree=ET.parse(world)
        replay.add_sensor=lambda original:saved[1](copy.deepcopy(tree))
        replay.validate_world=lambda original,derived:saved[2](tree,derived)
        def analyze(path,f,fence):
            result=saved[3](path,f,fence);mask_gate(result,path,reference)
            result.update(status='candidate_rendered_review_pending',reason='Expected RGB appearance change; exact source mask and complete boxes preserved. No admission.')
            return result
        replay.analyze=analyze
    async def command(*args,**kw):
        if args[:3]!=('gz','topic','-e') or not any(str(a).endswith('/pose/info') for a in args):return await saved[4](*args,**kw)
        for _ in range(6):
            raw=await saved[4](*args,**dict(kw,timeout=5))
            if valid_clock(json.loads(raw)):return raw
            await asyncio.sleep(.25)
        raise ValueError('Clock preflight failed')
    replay.command=command;last=None;rp=None
    try:
        for n in range(1,4):
            rp=folder/f'replay/N05/attempt-{n:02}/receipt.json'
            if rp.exists():last=prior.read(rp);prior.verify(last)
            elif rp.parent.exists():continue
            else:last=await replay.attempt(frame,n)
            if not last['process_cleanup_complete']:raise ValueError('Cleanup incomplete')
            if last['status']!='technical_failure' or any(e in last.get('reason','') for e in ('TypeError:','KeyError:','AttributeError:')):break
    finally:
        replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.command=saved
        if oldlog is None:os.environ.pop('GZ_LOG_PATH',None)
        else:os.environ['GZ_LOG_PATH']=oldlog
    return dict(variant=variant,status=last['status'] if last else 'attempts_exhausted',receipt=str(rp))


async def run():
    p=freeze();dest=OUT/'completion.json'
    if dest.exists():prior.verify(prior.read(dest));return
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=await unit(p['frame'],'source-control');results=[result]
        if result['status']=='original_pixel_evidence_certified':
            reference=Path(result['receipt']).parent/'first-stable-window/frame-1-mask.bin'
            for v in p['variants']:
                result=await unit(p['frame'],v['variant'],v['world_path'],reference);results.append(result)
                if result['status']!='candidate_rendered_review_pending':break
        paths=[OUT/'protocol.json',*[Path(r['receipt']) for r in results]]
        prior.frozen(dest,dict(status='source_plus_three_candidates_review_pending' if len(results)==4 and all(r['status']=='candidate_rendered_review_pending' for r in results[1:]) else 'expansion_blocked',
            units=results,training_ready=False,training_started=False,inputs={str(p):prior.file_sha256(p) for p in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--capture',action='store_true');a=ap.parse_args()
    if a.capture:asyncio.run(run())
    else:freeze();print('FROZEN_NO_CAPTURE_NO_TRAINING')
