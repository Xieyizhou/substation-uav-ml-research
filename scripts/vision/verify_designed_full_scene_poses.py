"""Bounded fresh canonical acquisition and exact replay for four designed views."""
import argparse,asyncio,fcntl,json,os,re,shutil
from pathlib import Path
import numpy as np
from scripts.vision.design_full_scene_poses import OUT as DESIGN,freeze,prior
from scripts.vision.build_material_view_world_drafts import valid_clock
from scripts.vision.audit_material_mask_coverage import coverage
from scripts.vision import run_visibility_cleanup_validation as replay
from src.vision.canonical.collect import collect

OUT=DESIGN/'real-verification-v1'


async def replay_frame(frame,unit):
    saved=(replay.OUT,replay.analyze,replay.command);oldlog=os.environ.get('GZ_LOG_PATH');replay.OUT=unit
    logs=unit/'runtime-logs';logs.mkdir(exist_ok=True);os.environ['GZ_LOG_PATH']=str(logs)
    def analyze(folder,f,fence):
        r=saved[1](folder,f,fence);checks=[]
        for n in range(1,4):
            a=np.fromfile(folder/f'first-stable-window/frame-{n}-mask.bin',dtype='u1').reshape(1080,1920,3)
            c=coverage(a,[e['runtime_label'] for e in f['events']],f['instance_mapping'])
            c['visible_pixels_by_label']={str(k):v for k,v in c['visible_pixels_by_label'].items()};checks.append(c)
        r['full_mask_coverage']=checks
        if any(c['missing_targets'] for c in checks):r.update(status='semantic_blocked',reason='Mapped visible target has no full box')
        return r
    async def command(*args,**kw):
        if args[:3]!=('gz','topic','-e') or not any(str(a).endswith('/pose/info') for a in args):return await saved[2](*args,**kw)
        for _ in range(6):
            raw=await saved[2](*args,**dict(kw,timeout=5))
            if valid_clock(json.loads(raw)):return raw
            await asyncio.sleep(.25)
        raise ValueError('Clock preflight failed')
    replay.analyze=analyze;replay.command=command;last=None;rp=None
    try:
        for n in range(1,4):
            rp=unit/'replay'/frame['review_ids'][0]/f'attempt-{n:02}/receipt.json'
            if rp.exists():last=prior.read(rp);prior.verify(last)
            elif rp.parent.exists():continue
            else:last=await replay.attempt(frame,n)
            if not last['process_cleanup_complete']:raise ValueError('Cleanup incomplete')
            if last['status']!='technical_failure' or any(e in last.get('reason','') for e in ('TypeError:','KeyError:','AttributeError:')):break
    finally:
        replay.OUT,replay.analyze,replay.command=saved
        if oldlog is None:os.environ.pop('GZ_LOG_PATH',None)
        else:os.environ['GZ_LOG_PATH']=oldlog
    return rp,last


async def run():
    p=freeze()
    if p['missing_classes']:raise ValueError('Frozen design lacks classes')
    OUT.mkdir(exist_ok=True);pp=OUT/'protocol.json';dest=OUT/'completion.json'
    deps=[DESIGN/'protocol.json',Path(__file__),prior.ROOT/'src/vision/canonical/collect.py',prior.ROOT/'src/vision/canonical/recovery.py',prior.ROOT/'scripts/vision/audit_material_mask_coverage.py',Path(replay.__file__)]
    if not pp.exists():prior.frozen(pp,dict(status='four_fixed_new_views_frozen',views=p['selected'],plan_path=p['plan_path'],
        policy='Each canonical view uses existing maximum three technical acquisitions, no outer retry. Each exact replay maximum three technical attempts; semantic failures never retried. Fixed sources are not replaced.',
        training_ready=False,training_started=False,inputs={str(p):prior.file_sha256(p) for p in deps}))
    prior.verify(prior.read(pp))
    if dest.exists():prior.verify(prior.read(dest));return
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);rows=[];paths=[pp]
        for v in p['selected']:
            sid=v['probe_id'];unit=OUT/sid;unit.mkdir(exist_ok=True);cp=unit/'capture/collection-receipt.json'
            if cp.exists():c=prior.read(cp)
            elif cp.parent.exists():rows.append(dict(probe_id=sid,status='incomplete_capture_preserved'));continue
            else:c=await collect(p['plan_path'],cp.parent,mode='calibration',view_id=v['view_id'])
            paths.append(cp)
            if c['status']!='complete_pending_review':rows.append(dict(probe_id=sid,status='capture_blocked',receipt=str(cp),error=c.get('error')));continue
            if len(c['views'])!=1 or c['views'][0]['view_id']!=v['view_id']:raise ValueError('Captured view identity conflict')
            source=c['views'][0];sp=unit/'source.json'
            if not sp.exists():prior.frozen(sp,dict(source,inputs={str(cp):prior.file_sha256(cp),source['rgb_path']:source['image_sha256']}))
            prior.verify(prior.read(sp));events=[]
            for n,t in enumerate(source['truth']['objects']):
                label=int(re.search(r'instance-(\d+)-',t['annotation_id'])[1]);identity=c['collection_checks']['instance_mapping'][str(label)]
                if identity['category']!=t['class_name']:raise ValueError('Instance class mismatch')
                events.append(dict(review_id=f'{sid}-{n:02}',runtime_label=label,object_id=identity['object_id'],bbox_xyxy=t['bbox_xyxy']))
            f=dict(member_id='new-pose:'+v['view_id'],lineage_id='pose:'+v['view_id'],class_name=v['category'],review_ids=[sid],
                source_image=source['rgb_path'],source_receipt=str(sp),source_plan=p['plan_path'],source_world=str(Path(p['plan_path']).parent/'world.sdf'),
                actual_pose=source['actual_pose'],world_name=c.get('world_name',prior.read(p['plan_path'])['world_name']),instance_mapping=c['collection_checks']['instance_mapping'],events=events)
            helper=unit/'gz_visibility_capture_cleanup_fixed'
            if not helper.exists():shutil.copy2(DESIGN.parent/'native-source-replay-v2/gz_visibility_capture_cleanup_fixed',helper)
            up=unit/'protocol.json'
            if not up.exists():prior.frozen(up,dict(frame=f,training_ready=False,inputs={str(pp):prior.file_sha256(pp),str(sp):prior.file_sha256(sp),str(helper):prior.file_sha256(helper)}))
            prior.verify(prior.read(up));rp,result=await replay_frame(f,unit)
            if rp and rp.exists():paths.append(rp)
            paths += [up,sp]
            rows.append(dict(probe_id=sid,status=result['status'] if result else 'replay_attempts_exhausted',capture_receipt=str(cp),replay_receipt=str(rp),reason=result.get('reason','') if result else 'Incomplete retained'))
        prior.frozen(dest,dict(status='four_new_pose_checks_finished_review_required',units=rows,training_ready=False,training_started=False,inputs={str(p):prior.file_sha256(p) for p in paths}))
        print([(r['probe_id'],r['status']) for r in rows])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--capture',action='store_true');a=ap.parse_args()
    if a.capture:asyncio.run(run())
    else:freeze();print('NO_CAPTURE_NO_TRAINING')
