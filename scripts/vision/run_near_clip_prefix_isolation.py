"""Resolve default-install precedence before one final control replay."""
import argparse
import asyncio
import fcntl
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from scripts.vision import run_near_clip_build_test as base


class PrefixAsync:
    def __getattr__(self,name):return getattr(asyncio,name)
    async def create_subprocess_exec(self,*args,**kwargs):
        if args[:4]!=('gz','sim','-s','-r'):raise ValueError('Unexpected launch')
        kwargs['env']=dict(kwargs['env'],GZ_RENDERING_PLUGIN_PATH=str(base.OUT/'variants'/base.VARIANT),
            GZ_RENDERING_INSTALL_PREFIX=str(base.OUT/'runtime-prefix'/base.VARIANT),
            GZ_RENDERING_RESOURCE_PATH='/opt/homebrew/opt/gz-rendering8/share/gz/gz-rendering8')
        return await asyncio.create_subprocess_exec(str(base.OUT/'diagnostic-server'),args[4],**kwargs)


def prepare(variant):
    p=base.freeze();base.guard()
    dest=base.OUT/'prefix-isolated'/variant;dest.mkdir(parents=True,exist_ok=True)
    path=dest/'protocol.json'
    if path.exists():base.verify(base.read(path));return p,dest
    binary=base.OUT/'resolve-plugin-prefix'
    if not binary.exists():
        flags=shlex.split(subprocess.check_output(['pkg-config','--cflags','--libs','gz-rendering8'],text=True))
        subprocess.run(['clang++','-std=c++17',str(base.ROOT/'tools/renderer_clip/resolve_plugin.cc'),'-o',str(binary),*flags],check=True)
    env=dict(os.environ,GZ_RENDERING_PLUGIN_PATH=str(base.OUT/'variants'/variant),GZ_RENDERING_INSTALL_PREFIX=str(base.OUT/'runtime-prefix'/variant))
    resolved=subprocess.check_output([str(binary)],env=env,text=True)
    result=next(line[7:] for line in resolved.splitlines() if line.startswith('result='))
    expected=base.OUT/'variants'/variant/'libgz-rendering8-ogre2.8.2.3.dylib'
    if Path(result).resolve()!=expected.resolve():raise ValueError('Preflight still resolves wrong renderer')
    helper=dest/'gz_visibility_capture_cleanup_fixed';shutil.copy2(base.DUAL/'gz_visibility_capture_cleanup_fixed',helper)
    paths=[base.OUT/'protocol.json',Path(__file__),binary,base.ROOT/'tools/renderer_clip/resolve_plugin.cc',helper,expected]
    base.frozen(path,dict(status='prefix_isolation_preflight_passed',variant=variant,frames=p['frames'],
        loader_probe=resolved,max_attempts=1,training_ready=False,training_started=False,
        policy='Default installation prefix overridden only in this child; explicit resources remain production read-only. Control total launches capped at three including two rejected identity checks.',
        inputs={str(p):base.file_sha256(p) for p in paths}))
    return p,dest


async def run(variant):
    p,dest=prepare(variant);base.VARIANT=variant
    replay=base.replay
    with (base.OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=(replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio)
        replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio=dest,base.dual_sensor,base.validate_dual,base.analyze,PrefixAsync()
        rp=dest/'replay/T027/attempt-01/receipt.json'
        try:
            if rp.exists():last=base.read(rp);base.verify(last)
            elif rp.parent.exists():raise ValueError('Incomplete final attempt retained; no retry')
            else:last=await replay.attempt(p['frames'][0],1)
        finally:(replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio)=old
        base.guard()
        path=dest/'completion.json';paths=[dest/'protocol.json',rp]
        if path.exists():base.verify(base.read(path))
        else:base.frozen(path,dict(status=last['status'],training_ready=False,training_started=False,
            inputs={str(p):base.file_sha256(p) for p in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--variant',required=True,choices=['control','clipped']);ap.add_argument('--replay',action='store_true');args=ap.parse_args()
    if args.replay:asyncio.run(run(args.variant))
    else:prepare(args.variant);print('LOADER_PREFLIGHT_PASSED_NO_RENDER')
