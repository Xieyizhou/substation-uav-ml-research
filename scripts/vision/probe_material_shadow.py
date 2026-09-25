"""Bounded isolated static Gazebo test; no PX4, arming or training data intake."""
import asyncio
import os
from pathlib import Path
import signal
import sys
import time
from scripts.vision import material_shadow as shadow
from scripts.vision.prepare_paired_visual_factors import BASE
from src.vision.canonical.plan import CARRIER,read_record,write_record
from src.ml.artifacts import file_sha256

OUT=Path('data/research/material-shadow-v1/isolated-probe-001').resolve()

async def command(*args):
    proc=await asyncio.create_subprocess_exec(*args,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
    try:
        stdout,_=await asyncio.wait_for(proc.communicate(),8)
        if proc.returncode:raise RuntimeError(stdout.decode(errors='replace'))
        return stdout.decode(errors='replace')
    finally:
        if proc.returncode is None:proc.kill();await proc.wait()

async def main():
    OUT.mkdir(parents=True,exist_ok=False)
    base=BASE/'original/plan';p=read_record(base/'plan.json');world=base/'world.sdf'
    if file_sha256(world)!=p['files']['world.sdf']:raise ValueError('World hash drift')
    view=p['calibration_views'][0]
    os.environ['GZ_PARTITION']='material_shadow_probe_'+str(os.getpid());os.environ['GZ_IP']='127.0.0.1'
    server=child=None;error=None
    started=time.monotonic()
    try:
        with (OUT/'simulator.log').open('x') as log:
            server=await asyncio.create_subprocess_exec('gz','sim','-s','-r',str(world),stdout=log,stderr=asyncio.subprocess.STDOUT,start_new_session=True)
            end=time.monotonic()+45
            while time.monotonic()<end:
                if server.returncode is not None:raise RuntimeError('Simulator exited '+str(server.returncode))
                topics=await command('gz','topic','-l')
                if '/research_camera/image' in topics.splitlines():break
                await asyncio.sleep(1)
            else:raise TimeoutError('RGB topic not discovered in 45 seconds')
            x,y,z=view['position'];qx,qy,qz,qw=view['orientation']
            req=f'name: "{CARRIER}" position {{x:{x} y:{y} z:{z}}} orientation {{x:{qx} y:{qy} z:{qz} w:{qw}}}'
            answer=await command('gz','service','-s',f'/world/{p["world_name"]}/set_pose','--reqtype','gz.msgs.Pose','--reptype','gz.msgs.Boolean','--timeout','3000','--req',req)
            if 'true' not in answer:raise RuntimeError('Pose service rejected: '+answer)
            write_record(OUT/'setup.json',dict(world_sha256=file_sha256(world),requested_view=view,pose_service_response=answer,actual_pose_verified=False,partition=os.environ['GZ_PARTITION'],px4_started=False,control_authority='none',training_admitted=False,promotable=False))
            with (OUT/'sidecar.log').open('x') as childlog:
                child=await asyncio.create_subprocess_exec(sys.executable,'-m','scripts.vision.material_shadow','--mode','live','--seconds','15','--topic','/research_camera/image','--output',str(OUT/'live'),stdout=childlog,stderr=asyncio.subprocess.STDOUT)
                await asyncio.wait_for(child.wait(),65)
                if child.returncode:raise RuntimeError('Sidecar failed; see sidecar.log')
    except Exception as e:error=f'{type(e).__name__}: {e}'
    finally:
        if child is not None and child.returncode is None:
            child.terminate()
            try:await asyncio.wait_for(child.wait(),5)
            except asyncio.TimeoutError:child.kill();await child.wait()
        if server is not None and server.returncode is None:
            os.killpg(server.pid,signal.SIGINT)
            try:await asyncio.wait_for(server.wait(),8)
            except asyncio.TimeoutError:os.killpg(server.pid,signal.SIGKILL);await server.wait()
        write_record(OUT/'probe.json',dict(status='blocked' if error else 'static_live_probe_complete',error=error,elapsed_s=time.monotonic()-started,owned_processes_exited=(server is None or server.returncode is not None) and (child is None or child.returncode is not None),px4_started=False,flight_tested=False,training_admitted=False,promotable=False,inputs={str(q):file_sha256(q) for q in [world,base/'plan.json',Path(__file__).resolve()]}))
    print(error or 'STATIC_LIVE_PROBE_COMPLETE',flush=True)

if __name__=='__main__':asyncio.run(main())
