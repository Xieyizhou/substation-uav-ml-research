"""Independent depth-only build replay on four frozen existing poses. Never trains."""
import argparse
import asyncio
import difflib
import fcntl
import os
import shutil
import subprocess
from pathlib import Path
from scripts.vision import run_near_clip_build_test as base
from scripts.vision import run_clip_existing_pose_regression as regression
from scripts.vision.run_near_clip_prefix_isolation import PrefixAsync
from scripts.vision.validate_edge_raster import OUT as PRIOR

BUILD=base.OUT
OUT=base.ROOT/'data/research/ml_training_recovery_v1/renderer-depth-clip-test-v1'


def controls(tag,variant='control'):
    if tag=='T027':return BUILD/'prefix-isolated'/variant/'replay/T027/attempt-01/receipt.json'
    return regression.OUT/variant/'replay'/tag/'attempt-01/receipt.json'


def freeze():
    base.guard();base.verify(base.read(PRIOR/'completion.json'));base.verify(base.read(BUILD/'protocol.json'))
    path=OUT/'protocol.json'
    if path.exists():base.verify(base.read(path));return base.read(path)
    old=BUILD/'patched-source';source=OUT/'source'
    changed=[];paths=[PRIOR/'completion.json',BUILD/'protocol.json',base.SOURCE/'protocol.json',Path(__file__),
        Path(regression.__file__),base.ROOT/'scripts/vision/run_near_clip_prefix_isolation.py',OUT/'diagnostic-server',
        OUT/'configure.log',OUT/'build.log',OUT/'build/CMakeCache.txt',OUT/'test-depth-clip',base.ROOT/'tools/renderer_depth/test_depth_clip.cc']
    for p in old.rglob('*'):
        if not p.is_file():continue
        q=source/p.relative_to(old)
        if not q.is_file():raise ValueError('Source missing: '+str(q))
        paths.extend([p,q])
        if base.file_sha256(p)!=base.file_sha256(q):changed.append(str(p.relative_to(old)))
    if changed!=['ogre2/src/DiagnosticHomogeneousClip.hh']:raise ValueError('Unexpected source differences: '+repr(changed))
    before=(old/changed[0]).read_text();after=(source/changed[0]).read_text()
    if after!=before.replace('// Diagnostic-only homogeneous polygon clipping. No runtime admission policy.', '// Diagnostic-only depth-plane clipping: preserve lateral full-box semantics.').replace('unsigned plane=0; plane<6','unsigned plane=4; plane<6'):
        raise ValueError('Unexpected depth-only patch')
    lib=OUT/'build/lib/libgz-rendering8-ogre2.8.2.3.dylib';dest=OUT/'variants/depth';dest.mkdir(parents=True)
    shutil.copy2(lib,dest/lib.name)
    for name in ('libgz-rendering8-ogre2.8.dylib','libgz-rendering8-ogre2.dylib','libgz-rendering-ogre2.dylib'):(dest/name).symlink_to(lib.name)
    helper=OUT/'gz_visibility_capture_cleanup_fixed';shutil.copy2(base.DUAL/'gz_visibility_capture_cleanup_fixed',helper)
    probe=BUILD/'resolve-plugin-prefix'
    env=dict(os.environ,GZ_RENDERING_PLUGIN_PATH=str(dest),GZ_RENDERING_INSTALL_PREFIX=str(OUT/'runtime-prefix/depth'))
    resolution=subprocess.check_output([str(probe)],env=env,text=True)
    resolved=next(x[7:] for x in resolution.splitlines() if x.startswith('result='))
    if Path(resolved).resolve()!=(dest/lib.name).resolve():raise ValueError('Loader preflight resolves different library')
    paths += [lib,dest/lib.name,helper,probe,OUT/'build/lib/libgz-rendering8.8.2.3.dylib']
    frames=base.read(base.SOURCE/'protocol.json')['frames']
    if [f['review_ids'][0] for f in frames]!=['T020','T036','T023','T027']:raise ValueError('Fixed pose membership changed')
    for f in frames:
        for variant in ('control','clipped'):
            rp=controls(f['review_ids'][0],variant);base.verify(base.read(rp));paths.append(rp)
        paths += [Path(f[k]) for k in ('source_receipt','source_world','source_plan','source_image')]
    return base.frozen(path,dict(status='depth_only_build_frozen',frames=frames,max_attempts=2,loader_probe=resolution,
        policy='Only near/far homogeneous clipping planes (4,5), retain lateral projection bounds and existing screen-coordinate clamp. Same-side rejection retained. Original and six-plane controls reused only with valid identities. All four fixed poses required.',
        source_diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='six-plane',tofile='depth-only')),
        training_ready=False,training_started=False,inputs={str(p):base.file_sha256(p) for p in paths}))


async def run():
    protocol=freeze();replay=base.replay
    with (BUILD/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=(base.OUT,base.VARIANT,replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio)
        base.OUT,base.VARIANT=OUT,'depth'
        replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio=OUT,base.dual_sensor,base.validate_dual,regression.analyze,PrefixAsync()
        paths=[OUT/'protocol.json'];results=[]
        try:
            for f in protocol['frames']:
                last=None
                for n in (1,2):
                    rp=OUT/'replay'/f['review_ids'][0]/f'attempt-{n:02}'/'receipt.json'
                    if rp.exists():last=base.read(rp);base.verify(last)
                    elif rp.parent.exists():continue
                    else:last=await replay.attempt(f,n)
                    paths.append(rp)
                    if last['status']!='technical_failure':break
                results.append(dict(review_id=f['review_ids'][0],status=last['status'] if last else 'incomplete_attempts'))
                if last is None or last['status']!='existing_pose_technical_checks_passed':break
        finally:
            (base.OUT,base.VARIANT,replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio)=old
        base.guard()
        dest=OUT/'run-receipt.json'
        if dest.exists():base.verify(base.read(dest));return
        base.frozen(dest,dict(status='runs_complete' if len(results)==4 and all(r['status']=='existing_pose_technical_checks_passed' for r in results) else 'runs_incomplete',
            results=results,training_ready=False,training_started=False,inputs={str(p):base.file_sha256(p) for p in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');args=ap.parse_args()
    if args.replay:asyncio.run(run())
    else:freeze();print('DEPTH_ONLY_BUILD_FROZEN_NO_REPLAY_NO_TRAINING')
