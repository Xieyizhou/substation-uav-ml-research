"""Launch an isolated debugger/server pair, reuse strict same-frame checks."""
import argparse
import asyncio
import fcntl
import json
import os
import shutil
from pathlib import Path

import psutil
from scripts.vision.diagnose_full_box_projection import OUT as ANALYTIC, ROOT, read, verify, frozen, file_sha256
from scripts.vision.run_dual_box_diagnosis import OUT as DUAL, SOURCE, dual_sensor, validate_dual, analyze
import scripts.vision.run_visibility_cleanup_validation as replay

OUT = ANALYTIC/'runtime-trace-v1'
LIBRARY = Path('/opt/homebrew/opt/gz-rendering8/lib/libgz-rendering8-ogre2.8.2.3.dylib')
PINNED_HASH = '336cda8e141c9f25848134d87988bfc019b86c0d0841efea06dd805fd0f88f9a'
ORIGINAL_COMMAND = replay.command
CLEANUP = []


def freeze():
    if file_sha256(LIBRARY) != PINNED_HASH:
        raise ValueError('Binary does not match inspected arm64 instruction offsets')
    for path in (ANALYTIC/'diagnosis.json', DUAL/'completion.json', DUAL/'protocol.json', SOURCE/'protocol.json'):
        verify(read(path))
    path = OUT/'protocol.json'
    if path.exists():
        verify(read(path)); return read(path)
    shutil.copy2(DUAL/'gz_visibility_capture_cleanup_fixed', OUT/'gz_visibility_capture_cleanup_fixed')
    paths = [ANALYTIC/'diagnosis.json', DUAL/'completion.json', DUAL/'protocol.json',
             SOURCE/'protocol.json', LIBRARY, OUT/'diagnostic-server', OUT/'gz_visibility_capture_cleanup_fixed',
             Path(__file__), ROOT/'scripts/vision/lldb_full_box_trace.py', ROOT/'tools/gz_isolated_diagnostic_server.cc',
             Path(replay.__file__), ROOT/'scripts/vision/run_dual_box_diagnosis.py']
    return frozen(path, dict(status='runtime_trace_frozen', frames=read(DUAL/'protocol.json')['frames'],
        policy='Launch dedicated gz::sim::Server under LLDB. Original installed renderer unchanged on disk; software breakpoints only in diagnostic process. Read matrices, projection extrema, visible-map label and actual accept/reject destinations. No target expressions or branch forcing.',
        max_attempts=3, training_ready=False, training_started=False,
        inputs={str(p):file_sha256(p) for p in paths}))


class IsolatedAsync:
    def __getattr__(self, name):
        return getattr(asyncio, name)

    async def create_subprocess_exec(self, *args, **kwargs):
        if args[:4] != ('gz','sim','-s','-r'):
            raise ValueError('Unexpected server launch')
        folder = Path(args[4]).parent
        kwargs['env'] = dict(kwargs['env'], EDGE_TRACE_DIR=str(folder))
        return await asyncio.create_subprocess_exec('lldb', '-b',
            '-o', 'settings set target.disable-aslr false',
            '-o', 'command script import '+str(ROOT/'scripts/vision/lldb_full_box_trace.py'),
            '-o', 'run', '-o', 'quit', '--', str(OUT/'diagnostic-server'), args[4], **kwargs)


async def command(*args, **kwargs):
    result = await ORIGINAL_COMMAND(*args, **kwargs)
    if args[:2] == ('gz','service') and '/set_pose' in args[3] and 'true' in result:
        await asyncio.sleep(1)
        # Exactly one active attempt is selected by the runner, not a directory glob.
        frozen(CURRENT_FOLDER/'pose-moved.json', dict(status='saved_pose_requested', response=result))
    return result


async def stop_owned(process):
    if process is None:
        return
    owned = []
    try:
        owned = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        pass
    identities = [(p.pid,p.create_time()) for p in owned]
    # Children may occupy separate debugger process groups; stop only captured identities.
    for child in reversed(owned):
        try: child.kill()
        except psutil.NoSuchProcess: pass
    await replay_stop(process)
    _, alive = psutil.wait_procs(owned, timeout=3)
    survivors = [p.pid for p in alive if p.is_running() and p.status()!=psutil.STATUS_ZOMBIE]
    CLEANUP.append(dict(root=process.pid, owned=identities, survivors=survivors))
    if survivors:
        raise RuntimeError('Diagnostic descendants still alive: '+str(survivors))


replay_stop = replay.stop_group
CURRENT_FOLDER = None


async def run():
    global CURRENT_FOLDER
    protocol = freeze()
    with (OUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        old = (replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command,replay.stop_group)
        replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze = OUT,dual_sensor,validate_dual,analyze
        replay.asyncio,replay.command,replay.stop_group = IsolatedAsync(),command,stop_owned
        paths = [OUT/'protocol.json']; last = None
        try:
            for number in range(1,4):
                CURRENT_FOLDER = OUT/'replay/T027'/f'attempt-{number:02}'
                rp = CURRENT_FOLDER/'receipt.json'
                if rp.exists():
                    last=read(rp);verify(last)
                elif CURRENT_FOLDER.exists():
                    continue
                else:
                    last=await replay.attempt(protocol['frames'][0],number)
                paths.append(rp)
                if last['status']!='technical_failure':break
        finally:
            (replay.OUT,replay.add_sensor,replay.validate_world,replay.analyze,replay.asyncio,replay.command,replay.stop_group)=old
        if file_sha256(LIBRARY)!=PINNED_HASH:
            raise ValueError('Installed renderer changed')
        dest=OUT/'run-receipt.json'
        if dest.exists():verify(read(dest)); print('VALID_RUNTIME_RUN_REUSED'); return
        frozen(dest,dict(status=last['status'] if last else 'incomplete_attempts',
            debugger_cleanup=CLEANUP, installed_library_unchanged=True, training_ready=False,training_started=False,
            inputs={str(p):file_sha256(p) for p in paths}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');args=ap.parse_args()
    if args.replay:asyncio.run(run())
    else:freeze();print('RUNTIME_TRACE_FROZEN_NOT_STARTED')
