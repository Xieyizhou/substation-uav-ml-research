"""Repeat-test harness. Geometry is never provided to LiveObstacleMap."""
import asyncio
import json
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

CASES = {
    'baseline': dict(east=1.15, north=3.7, trigger_north=.8, expected='goal'),
    'west': dict(east=.95, north=3.7, trigger_north=.8, expected='goal'),
    'later': dict(east=1.15, north=3.7, trigger_north=1.05, expected='goal'),
    'no_path': dict(east=1.15, north=4.3, trigger_north=.8, expected='safe_rejection'),
}


def ready(case, point, position):
    return position['vn'] >= .15 and point[1] >= CASES[case]['trigger_north']


async def insert(out, case):
    c = CASES[case]
    sdf = f'<sdf version="1.9"><model name="unregistered_pillar"><static>true</static><pose>{c["east"]-10} {c["north"]-10} 2 0 0 0</pose><link name="pillar"><collision name="collision"><geometry><box><size>0.3 0.3 4</size></box></geometry></collision><visual name="visual"><geometry><box><size>0.3 0.3 4</size></box></geometry></visual></link></model></sdf>'
    (out/'pillar.sdf').write_text(sdf)
    p = await asyncio.create_subprocess_exec('/opt/homebrew/bin/gz','service','-s','/world/substation_simple/create','--reqtype','gz.msgs.EntityFactory','--reptype','gz.msgs.Boolean','--timeout','3000','--req','sdf: '+json.dumps(sdf),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(p.communicate(),5)
    finally:
        if p.returncode is None:
            p.kill()
            await p.wait()
    write_record(out/'pillar-receipt.json',dict(returncode=p.returncode,stdout=stdout.decode(),stderr=stderr.decode(),sha256=file_sha256(out/'pillar.sdf'),case=case))
    if p.returncode or 'data: true' not in stdout.decode():
        raise RuntimeError('Fixture creation failed')
