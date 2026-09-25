"""Capture three N04 appearance candidates after exact source replay. No admission."""
import argparse
import asyncio
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from scripts.vision import replay_material_view_sources_v2 as fixed
from scripts.vision.build_material_view_world_drafts import valid_clock
from scripts.vision.establish_material_view_candidates import OUT as SOURCE, prior
from scripts.vision import run_visibility_cleanup_validation as replay

OUT=SOURCE/'N04-material-capture-v1'


async def run():
    manifest_path=SOURCE/'world-drafts-v1/manifest.json';m=prior.read(manifest_path);prior.verify(m)
    control_path=fixed.OUT/'replay/N04/attempt-01/receipt.json';control=prior.read(control_path);prior.verify(control)
    if control['status']!='original_pixel_evidence_certified':raise ValueError('Source control not certified')
    frame=next(f for f in prior.read(fixed.OUT/'protocol.json')['frames'] if f['review_ids']==['N04'])
    rootmask=control_path.parent/'first-stable-window/frame-1-mask.bin'
    rows=[r for r in m['rows'] if r['source_review_id']=='N04']
    if [r['variant'] for r in rows]!=['original','warm','cool']:raise ValueError('Variant matrix mismatch')
    OUT.mkdir(exist_ok=True);pp=OUT/'protocol.json'
    paths=[manifest_path,control_path,rootmask,Path(__file__),Path(replay.__file__),prior.ROOT/'scripts/vision/build_material_view_world_drafts.py']
    if not pp.exists():prior.frozen(pp,dict(status='three_variant_candidate_capture_frozen',frame=frame,variants=rows,max_attempts=3,
        purpose='Development candidate rendering only; historical admission-chain gap remains. Full labels unchanged, masks must equal exact-replay source, new RGB requires fresh review.',
        training_ready=False,training_started=False,inputs={str(p):prior.file_sha256(p) for p in paths}))
    prior.verify(prior.read(pp));results=[]
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for row in rows:
            unit=OUT/row['variant'];unit.mkdir(exist_ok=True)
            helper=unit/'gz_visibility_capture_cleanup_fixed'
            if not helper.exists():shutil.copy2(fixed.OUT/helper.name,helper)
            up=unit/'protocol.json'
            if not up.exists():prior.frozen(up,dict(variant=row,training_ready=False,inputs={str(pp):prior.file_sha256(pp),str(helper):prior.file_sha256(helper)}))
            prior.verify(prior.read(up))
            world=ET.parse(row['world_path'])
            saved=(replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.command);oldlog=os.environ.get('GZ_LOG_PATH')
            replay.OUT=unit;replay.add_sensor=lambda original:saved[1](copy.deepcopy(world))
            replay.validate_world=lambda original,derived:saved[2](world,derived)
            def analyze(folder,f,fence):
                r=saved[3](folder,f,fence)
                if r['stable_frames']!=3 or any(x['maximum_box_delta_px']>1 for x in r['records']):raise ValueError('Unstable or shifted full labels')
                for n in range(1,4):
                    mask=folder/f'first-stable-window/frame-{n}-mask.bin'
                    if prior.file_sha256(mask)!=prior.file_sha256(rootmask):raise ValueError('Material changes instance mask')
                r['status']='candidate_rendered_review_pending';r['reason']='RGB color change expected; exact source mask and complete box membership retained. Not original-RGB certification or admission.'
                return r
            replay.analyze=analyze
            async def command(*args,**kw):
                if args[:3]!=('gz','topic','-e') or not any(str(a).endswith('/pose/info') for a in args):return await saved[4](*args,**kw)
                for _ in range(6):
                    raw=await saved[4](*args,**dict(kw,timeout=5))
                    if valid_clock(json.loads(raw)):return raw
                    await asyncio.sleep(.25)
                raise ValueError('Clock preflight failed')
            replay.command=command;logs=unit/'runtime-logs';logs.mkdir(exist_ok=True);os.environ['GZ_LOG_PATH']=str(logs)
            last=None
            try:
                for n in range(1,4):
                    rp=unit/f'replay/N04/attempt-{n:02}/receipt.json'
                    if rp.exists():last=prior.read(rp);prior.verify(last)
                    elif rp.parent.exists():continue
                    else:last=await replay.attempt(frame,n)
                    if not last['process_cleanup_complete']:raise ValueError('Cleanup failed')
                    if last['status']!='technical_failure' or any(t in last.get('reason','') for t in ('TypeError:','KeyError:','AttributeError:')):break
            finally:
                replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.command=saved
                if oldlog is None:os.environ.pop('GZ_LOG_PATH',None)
                else:os.environ['GZ_LOG_PATH']=oldlog
            results.append(dict(variant=row['variant'],status=last['status'] if last else 'attempts_exhausted',receipt=str(rp)))
            paths.append(rp)
            if not last or last['status']!='candidate_rendered_review_pending':break
    dest=OUT/'completion.json'
    if dest.exists():prior.verify(prior.read(dest));return
    prior.frozen(dest,dict(status='three_candidates_captured_review_required' if len(results)==3 and all(x['status']=='candidate_rendered_review_pending' for x in results) else 'capture_blocked',
        units=results,training_ready=False,training_started=False,historical_admission_chain_valid=False,
        inputs={str(p):prior.file_sha256(p) for p in paths+[pp]}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--capture',action='store_true');a=ap.parse_args()
    if a.capture:asyncio.run(run())
    else:print('NO_CAPTURE_NO_TRAINING; use explicit --capture')
