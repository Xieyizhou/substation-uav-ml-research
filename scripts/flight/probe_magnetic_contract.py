"""Static multi-heading Gazebo magnetometer contract; no PX4 or flight."""
import asyncio
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.flight.check_px4_shadow import ROOT,BUILD,PX4,stop
from scripts.maps.prepare_research_vehicle import prepare_world_with_vehicle
from scripts.vision.probe_material_shadow import command
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256

OUT=ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1/magnetic-contract-001'


def decode(message):
    field=message.get('field_tesla',message.get('fieldTesla'))
    if not isinstance(field,dict):raise ValueError('Missing magnetic field')
    v=[float(field.get(k,0.)) for k in ('x','y','z')]
    if not all(math.isfinite(x) for x in v):raise ValueError('Nonfinite magnetic field')
    return v


async def main():
    OUT.mkdir(parents=True,exist_ok=False)
    world=OUT/'world.sdf';config=OUT/'server.config'
    prepare_world_with_vehicle(ROOT/'simulation/worlds/substation_simple.sdf',world,model_name='x500_research',entity_name='x500_research_0',pose='-8.5,-8.5,1.1,0,0,0')
    tree=ET.parse(world);w=tree.getroot().find('world')
    vehicle=next(i for i in w.findall('include') if i.findtext('name')=='x500_research_0')
    ET.SubElement(vehicle,'static').text='true';tree.write(world,encoding='utf-8',xml_declaration=True)
    tree=ET.parse(PX4/'Tools/simulation/gz/server.config')
    mag=next(p for p in tree.getroot().findall('.//plugin') if p.get('name')=='gz::sim::systems::Magnetometer')
    ET.SubElement(mag,'use_earth_frame_ned').text='false'
    ET.SubElement(mag,'use_units_gauss').text='true'
    tree.write(config,encoding='utf-8',xml_declaration=True)
    os.environ.update(GZ_IP='127.0.0.1',GZ_PARTITION=f'magnetic_contract_{os.getpid()}',
        GZ_SIM_SERVER_CONFIG_PATH=str(config),GZ_SIM_RESOURCE_PATH=f'{ROOT}/simulation/models:{PX4}/Tools/simulation/gz/models',GZ_SIM_SYSTEM_PLUGIN_PATH=str(BUILD/'src/modules/simulation/gz_plugins'))
    process=None;results=[];failure=None
    log=(OUT/'gazebo.log').open('x')
    try:
        process=await asyncio.create_subprocess_exec('gz','sim','-s','-r',str(world),stdout=log,stderr=asyncio.subprocess.STDOUT,start_new_session=True)
        deadline=asyncio.get_running_loop().time()+45
        while True:
            topics=(await command('gz','topic','-l')).splitlines()
            matches=[t for t in topics if t.endswith('/magnetometer') and 'x500_research_0' in t]
            if len(matches)==1:break
            if len(matches)>1:raise ValueError('Ambiguous magnetometer')
            if asyncio.get_running_loop().time()>deadline:raise TimeoutError('No magnetometer')
            await asyncio.sleep(1)
        for yaw in (0.,90.,180.):
            a=math.radians(yaw)
            request=f'name: "x500_research_0" position {{x:-8.5 y:-8.5 z:1.1}} orientation {{z:{math.sin(a/2)} w:{math.cos(a/2)}}}'
            answer=await command('gz','service','-s','/world/substation_simple/set_pose','--reqtype','gz.msgs.Pose','--reptype','gz.msgs.Boolean','--timeout','3000','--req',request)
            if 'true' not in answer:raise RuntimeError('Fixture pose rejected')
            await asyncio.sleep(1)
            raw=await command('gz','topic','-e','--json-output','-n','1','-t',matches[0])
            (OUT/f'mag-{int(yaw)}.json').write_text(raw)
            x,y,z=decode(json.loads(raw))
            east=math.cos(a)*x-math.sin(a)*y;north=math.sin(a)*x+math.cos(a)*y
            result=dict(requested_world_yaw_deg=yaw,body_flu_gauss=[x,y,z],bridge_frd_gauss=[x,-y,-z],world_enu_gauss=[east,north,z],declination_deg=math.degrees(math.atan2(east,north)),strength_gauss=math.sqrt(x*x+y*y+z*z))
            results.append(result);print(result,flush=True)
        spread=max(math.dist(r['world_enu_gauss'],results[0]['world_enu_gauss']) for r in results)
        if spread>.005 or not all(.4<r['strength_gauss']<.6 for r in results):raise RuntimeError('Magnetic contract inconsistent')
    except Exception as e:failure=f'{type(e).__name__}: {e}'
    finally:
        if process is not None:await stop(process)
        log.close()
        paths=[world,config,Path(__file__).resolve(),*OUT.glob('mag-*.json')]
        write_record(OUT/'receipt.json',dict(status='failed' if failure else 'static_magnetic_contract_consistent',error=failure,results=results,
            limitation='Fixture pose service acknowledged; not a synchronized flight pose certification. field_tesla protobuf carries gauss under explicit plugin setting.',
            px4_started=False,owned_process_exited=process is None or process.returncode is not None,inputs={str(p):file_sha256(p) for p in paths},training_admitted=False,promotable=False))
    print(failure or 'STATIC_MAGNETIC_CONTRACT_CONSISTENT')


if __name__=='__main__':asyncio.run(main())
