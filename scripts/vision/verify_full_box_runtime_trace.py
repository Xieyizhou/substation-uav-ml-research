"""Second bounded confirmation, binding the actual engine-plugin binary before launch."""
import argparse
import asyncio
import fcntl
import json
import shutil
import subprocess
from pathlib import Path

from scripts.vision import run_full_box_runtime_trace as runner
from scripts.vision.run_full_box_runtime_trace import ROOT, LIBRARY, PINNED_HASH, read, verify, frozen, file_sha256

FIRST = runner.OUT
OUT = FIRST/'verified-binary-v1'
PLUGIN = LIBRARY.parent/'gz-rendering-8/engine-plugins'/LIBRARY.name
PLUGIN_HASH = '015b8894053d9b75c9c15d06595ae376ac2fd7534eaa5233ced1cdfeb3b31b62'


def guard():
    for p, expected in ((LIBRARY,PINNED_HASH),(PLUGIN,PLUGIN_HASH)):
        if file_sha256(p)!=expected:
            raise ValueError('Renderer binary identity changed: '+str(p))


def prepare():
    guard();verify(read(FIRST/'run-receipt.json'))
    dest=OUT/'binary-binding.json'
    if dest.exists():verify(read(dest));return
    OUT.mkdir(exist_ok=True)
    shutil.copy2(FIRST/'diagnostic-server',OUT/'diagnostic-server')
    dumps=[]
    for p in (LIBRARY,PLUGIN):
        dump=subprocess.check_output(
            ['lldb','-b','-o','target create '+str(p),'-o',
             'disassemble -n _ZN2gz9rendering2v822Ogre2BoundingBoxCamera17FullBoundingBoxesEv','-o','quit'],text=True)
        dumps.append(dump)
    instructions=[[s for s in d.splitlines() if '[0x' in s] for d in dumps]
    if not instructions[0] or instructions[0]!=instructions[1]:
        raise ValueError('FullBoundingBoxes instruction streams differ')
    paths=[LIBRARY,PLUGIN,FIRST/'run-receipt.json',OUT/'diagnostic-server',Path(__file__),
           ROOT/'scripts/vision/run_full_box_runtime_trace.py',ROOT/'scripts/vision/lldb_full_box_trace.py']
    frozen(dest,dict(status='actual_plugin_bound_before_confirmation', instruction_streams_identical=True,
        disassembly=dumps, maximum_total_launches=3, already_used_launches=1,
        correction='First run pinned the top-level library; actual loaded plugin has a different whole-file hash. Preserve first run and confirm with both identities frozen before launch.',
        training_ready=False,training_started=False,inputs={str(p):file_sha256(p) for p in paths}))


async def run():
    prepare();guard()
    old=runner.OUT;runner.OUT=OUT
    replay=runner.replay
    previous=(replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command,replay.stop_group)
    try:
        protocol=runner.freeze()
        with (OUT/'runner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze=OUT,runner.dual_sensor,runner.validate_dual,runner.analyze
            replay.asyncio,replay.command,replay.stop_group=runner.IsolatedAsync(),runner.command,runner.stop_owned
            runner.CURRENT_FOLDER=OUT/'replay/T027/attempt-01'
            rp=runner.CURRENT_FOLDER/'receipt.json'
            if rp.exists():
                result=read(rp);verify(result)
            elif runner.CURRENT_FOLDER.exists():
                raise ValueError('Incomplete confirmation attempt retained; no automatic retry')
            else:
                result=await replay.attempt(protocol['frames'][0],1)
            paths=[OUT/'protocol.json',OUT/'binary-binding.json',rp]
            dest=OUT/'run-receipt.json'
            if dest.exists():verify(read(dest))
            else:frozen(dest,dict(status=result['status'],debugger_cleanup=runner.CLEANUP,
                training_ready=False,training_started=False,inputs={str(p):file_sha256(p) for p in paths}))
    finally:
        runner.OUT=old
        (replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command,replay.stop_group)=previous
    guard()
    paths=[OUT/'binary-binding.json',OUT/'run-receipt.json',OUT/'protocol.json',LIBRARY,PLUGIN]
    for p in paths[:3]:verify(read(p))
    # One confirmation is authorized here; no unbounded repetition.
    for folder in (OUT/'replay/T027').iterdir():
        if folder.is_dir():
            paths.append(folder/'receipt.json');verify(read(paths[-1]))
    dest=OUT/'binary-postcheck.json'
    if dest.exists():verify(read(dest))
    else:frozen(dest,dict(status='actual_plugin_unchanged_after_confirmation',
        training_ready=False,training_started=False,inputs={str(p):file_sha256(p) for p in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');args=ap.parse_args()
    if args.replay:asyncio.run(run())
    else:prepare();print('ACTUAL_PLUGIN_BOUND_NOT_STARTED')
