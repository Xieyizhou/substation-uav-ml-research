"""Stationary Gazebo sensor gate. No PX4, arming, or vehicle motion commands."""
import asyncio
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from scripts.flight.check_px4_shadow import ROOT, BUILD, PX4, stop
from scripts.maps.prepare_research_vehicle import prepare_world_with_vehicle
from scripts.vision.probe_material_shadow import command
from src.sensors.gazebo_lidar import GazeboLidar2DSource
from src.perception.lidar_detector import LidarRiskDetector
from src.flight.safety_supervisor import SafetySupervisor
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256

OUT=ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1/sensor-gate-001'


async def main():
    OUT.mkdir(parents=True,exist_ok=False)
    world=OUT/'world.sdf'
    prepare_world_with_vehicle(ROOT/'simulation/worlds/substation_simple.sdf',world,
        model_name='x500_research',entity_name='x500_research_0',pose='-8.5,-8.5,1.1,0,0,0')
    tree=ET.parse(world);w=tree.getroot().find('world')
    vehicle=next(i for i in w.findall('include') if i.findtext('name')=='x500_research_0')
    ET.SubElement(vehicle,'static').text='true'
    wall=ET.fromstring('''<model name="diagnostic_wall"><static>true</static><pose>-5.5 -8.5 1 0 0 0</pose><link name="wall"><collision name="collision"><geometry><box><size>0.4 4 2</size></box></geometry></collision><visual name="visual"><geometry><box><size>0.4 4 2</size></box></geometry></visual></link></model>''')
    w.append(wall);tree.write(world,encoding='utf-8',xml_declaration=True)
    os.environ.update(GZ_IP='127.0.0.1',GZ_PARTITION=f'avoidance_gate_{os.getpid()}',
        GZ_SIM_RESOURCE_PATH=f'{ROOT}/simulation/models:{PX4}/Tools/simulation/gz/models',
        GZ_SIM_SYSTEM_PLUGIN_PATH=str(BUILD/'src/modules/simulation/gz_plugins'),
        GZ_SIM_SERVER_CONFIG_PATH=str(PX4/'Tools/simulation/gz/server.config'))
    log=(OUT/'gazebo.log').open('x');process=None;source=None;failure=None;results=[]
    try:
        process=await asyncio.create_subprocess_exec('gz','sim','-s','-r',str(world),stdout=log,stderr=asyncio.subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+45
        while 'lidar' not in await command('gz','topic','-l'):
            if process.returncode is not None or time.monotonic()>deadline:raise RuntimeError('LiDAR startup failed')
            await asyncio.sleep(1)
        source=GazeboLidar2DSource();await source.start();await source.wait_ready(15)
        detector=LidarRiskDetector(source,resolution_m=1.,nominal_speed_m_s=.3)
        supervisor=SafetySupervisor()
        for name,x,expected,level,action in [('far',-5.5,2.9,'detected','continue'),('warning',-6.8,1.6,'warning','slow_down'),('danger',-7.6,.8,'danger','hover_then_land')]:
            request=f'name: "diagnostic_wall" position {{x:{x} y:-8.5 z:1}} orientation {{w:1}}'
            answer=await command('gz','service','-s','/world/substation_simple/set_pose','--reqtype','gz.msgs.Pose','--reptype','gz.msgs.Boolean','--timeout','3000','--req',request)
            if 'true' not in answer:raise RuntimeError('Wall fixture movement failed')
            await asyncio.sleep(1)
            scans=[];seen=set();deadline=time.monotonic()+5
            while len(scans)<10 and time.monotonic()<deadline:
                scan=source.latest()
                if scan.sequence not in seen:
                    seen.add(scan.sequence);index=min(range(len(scan.ranges_m)),key=lambda i:abs(scan.angle_at(i)))
                    d=detector.detect(0.,0.,yaw_deg=90.,altitude_m=1.1)
                    decision=supervisor.evaluate(d,'stop_and_land')
                    scans.append(dict(sequence=scan.sequence,front_m=scan.ranges_m[index],risk=d['risk_level'],action=decision.action,healthy=source.health().healthy))
                await asyncio.sleep(.05)
            passed=len(scans)==10 and all(abs(s['front_m']-expected)<.15 and s['risk']==level and s['action']==action and s['healthy'] for s in scans)
            results.append(dict(case=name,expected_distance_m=expected,passed=passed,scans=scans))
            print(name,passed,flush=True)
            if not passed:raise RuntimeError('Sensor fixture gate failed: '+name)
        await source.stop();await asyncio.sleep(.7)
        decision=supervisor.evaluate(detector.detect(0.,0.,yaw_deg=90.),'stop_and_land')
        results.append(dict(case='stream_stopped',passed=decision.action=='hover_then_land',action=decision.action,reason=decision.reason))
        if decision.action!='hover_then_land':raise RuntimeError('Stream loss not rejected')
    except Exception as e:failure=f'{type(e).__name__}: {e}'
    finally:
        if source is not None:await source.stop()
        if process is not None:await stop(process)
        log.close()
        write_record(OUT/'receipt.json',dict(status='failed' if failure else 'stationary_sensor_gate_passed',error=failure,cases=results,
            autonomous_flight_tested=False,physical_collision_avoidance_certified=False,px4_started=False,
            owned_simulator_exited=process is None or process.returncode is not None,
            training_admitted=False,promotable=False,
            inputs={str(p):file_sha256(p) for p in [world,Path(__file__).resolve(),ROOT/'src/perception/lidar_detector.py',ROOT/'src/flight/safety_supervisor.py']}))
    print(failure or 'STATIONARY_SENSOR_GATE_PASSED',flush=True)


if __name__=='__main__':asyncio.run(main())
