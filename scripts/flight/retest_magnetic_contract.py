"""Explicit ENU-gauss simulation contract with unmodified auto WMM policy."""
import asyncio
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.flight import fly_px4_shadow_hover as trial
from scripts.flight import retest_sim_heading as prearm
from scripts.flight import check_final_heading as airborne
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256

OUT=trial.ROOT/'data/research/material-shadow-v1/autonomy-avoidance-v1/enu-magnetic-flight-001'


def validate_contract(config):
    plugins=[p for p in ET.parse(config).getroot().findall('.//plugin') if p.get('name')=='gz::sim::systems::Magnetometer']
    if len(plugins)!=1 or plugins[0].findtext('use_earth_frame_ned')!='false' or plugins[0].findtext('use_units_gauss')!='true':
        raise ValueError('Magnetometer plugin/bridge contract mismatch')


async def main():
    source=OUT.parent/'magnetic-contract-001';receipt=json.loads((source/'receipt.json').read_text())
    if receipt['status']!='static_magnetic_contract_consistent':raise ValueError('Static contract not passed')
    config=source/'server.config'
    if file_sha256(config)!=receipt['inputs'][str(config)]:raise ValueError('Configuration hash drift')
    validate_contract(config)
    OUT.mkdir(parents=True,exist_ok=False)
    paths=[config,source/'receipt.json',Path(__file__).resolve(),Path(trial.__file__).resolve(),trial.BUILD/'bin/px4',trial.PX4/'src/modules/simulation/gz_bridge/GZBridge.cpp']
    write_record(OUT/'protocol.json',dict(simulation_only=True,takeoff_altitude_m=2.,hover_s=10,horizontal_flight=False,
        parameters={'EKF2_DECL_TYPE':3,'EKF2_MAG_DECL':0},bridge_mode='PX4_GZ_MAG_ENU_GAUSS=1',
        heading_gate_deg=1.,position_residual_gate_m=.05,inputs={str(p):file_sha256(p) for p in paths},training_admitted=False,promotable=False))
    os.environ.update(PX4_GZ_MAG_ENU_GAUSS='1',PX4_PARAM_EKF2_DECL_TYPE='3',PX4_PARAM_EKF2_MAG_DECL='0')
    prearm.OUT=airborne.OUT=OUT;trial.OUT=OUT/'runtime'
    await trial.main(fly=True,preflight_hook=prearm.gate,takeoff_altitude=2.,post_hover_hook=airborne.post_hover,server_config=config)


if __name__=='__main__':asyncio.run(main())
