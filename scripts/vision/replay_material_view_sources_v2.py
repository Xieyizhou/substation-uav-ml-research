"""Explicitly authorized repaired forensic replay; no data admission or training."""
import asyncio
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
from scripts.vision import replay_material_view_sources as old
from scripts.vision.build_material_view_world_drafts import valid_clock
from scripts.vision.establish_material_view_candidates import prior, OUT as SOURCE
from scripts.vision import run_visibility_cleanup_validation as replay

OUT=SOURCE/'native-source-replay-v2'


def freeze():
    dest=OUT/'protocol.json'
    if dest.exists():p=prior.read(dest);prior.verify(p);return p
    p=prior.read(old.OUT/'protocol.json');prior.verify(p)
    OUT.mkdir(exist_ok=True)
    helper=old.OUT/'gz_visibility_capture_cleanup_fixed';shutil.copy2(helper,OUT/helper.name)
    paths=[Path(__file__),Path(replay.__file__),prior.ROOT/'scripts/vision/build_material_view_world_drafts.py',helper,OUT/helper.name]
    for f in p['frames']:
        paths += [Path(f[k]) for k in ('source_image','source_receipt','source_plan','source_world')]
        prior.verify(prior.read(f['source_receipt']))
    return prior.frozen(dest,dict(status='repaired_forensic_replay_frozen',frames=p['frames'],max_attempts=3,
        authorization='User explicitly authorized processing blockers and arranging repaired replay and capture; fresh technical budget, old three failures retained.',
        historical_protocol_reference=dict(path=str(old.OUT/'protocol.json'),sha256=prior.file_sha256(old.OUT/'protocol.json')),
        historical_admission_chain_valid=False,
        boundary='Raw saved-frame forensic replay only. Does not consume historical content strata as approval. Historical review.json stale script binding remains unresolved; no training admission or material capture expansion unless forensic control passes and admission risks are separately addressed.',
        training_ready=False,training_started=False,inputs={str(p):prior.file_sha256(p) for p in paths}))


async def run():
    p=freeze();dest=OUT/'completion.json'
    if dest.exists():prior.verify(prior.read(dest));return
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        saved=(replay.OUT,replay.command);logenv=os.environ.get('GZ_LOG_PATH')
        replay.OUT=OUT;log=OUT/'runtime-logs';log.mkdir(exist_ok=True);os.environ['GZ_LOG_PATH']=str(log)
        clock_samples=[]
        async def command(*args,**kw):
            if args[:3]!=('gz','topic','-e') or not any(str(a).endswith('/pose/info') for a in args):return await saved[1](*args,**kw)
            for n in range(6):
                raw=await saved[1](*args,**dict(kw,timeout=5));m=json.loads(raw);ok=valid_clock(m)
                clock_samples.append(dict(sample=n,accepted=ok,message=m))
                if ok:return raw
                await asyncio.sleep(.25)
            raise ValueError('Bounded explicit clock preflight failed')
        replay.command=command;results=[];paths=[OUT/'protocol.json']
        try:
            for f in p['frames']:
                last=None
                for n in range(1,4):
                    rp=OUT/'replay'/f['review_ids'][0]/f'attempt-{n:02}'/'receipt.json'
                    if rp.exists():last=prior.read(rp);prior.verify(last)
                    elif rp.parent.exists():continue
                    else:last=await replay.attempt(f,n)
                    paths.append(rp)
                    if not last['process_cleanup_complete']:raise ValueError('Cleanup failed')
                    if last['status']!='technical_failure':break
                    # Deterministic code defects stop immediately; do not consume all attempts on same error.
                    if any(t in last.get('reason','') for t in ('TypeError:','KeyError:','AttributeError:')):break
                results.append(dict(source_review_id=f['review_ids'][0],status=last['status'] if last else 'attempts_exhausted',receipt=str(rp),reason=last.get('reason','') if last else 'Incomplete attempts retained'))
                if not last or last['status']!='original_pixel_evidence_certified':break
        finally:
            replay.OUT,replay.command=saved
            if logenv is None:os.environ.pop('GZ_LOG_PATH',None)
            else:os.environ['GZ_LOG_PATH']=logenv
        prior.frozen(dest,dict(status='forensic_pilot_passed_review_required' if len(results)==2 and all(x['status']=='original_pixel_evidence_certified' for x in results) else 'forensic_pilot_blocked',results=results,
            clock_samples=clock_samples,training_ready=False,training_started=False,historical_admission_chain_valid=False,
            inputs={str(p):prior.file_sha256(p) for p in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');a=ap.parse_args()
    if a.replay:asyncio.run(run())
    else:freeze();print('FROZEN_NO_REPLAY')
