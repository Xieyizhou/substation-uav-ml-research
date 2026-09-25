"""Three isolated static latency runs using the corrected production sidecar."""
import asyncio
from pathlib import Path
from unittest.mock import patch
from scripts.vision import probe_material_shadow as probe
from scripts.vision import material_shadow as sidecar
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256

ROOT=Path('data/research/material-shadow-v1/rect-fixed-retest-v1').resolve()

async def main():
    if sidecar.PROTOCOL.get('rect') is not False:raise ValueError('Square padding required')
    ROOT.mkdir(parents=True,exist_ok=False)
    paths=[]
    for i in range(1,4):
        folder=ROOT/f'live-{i:02}'
        with patch.object(probe,'OUT',folder):await probe.main()
        r=sidecar.reference.checked(folder/'probe.json')
        if r['status']!='static_live_probe_complete' or not r['owned_processes_exited']:raise ValueError('Live probe failed')
        p=folder/'live/completion.json';sidecar.reference.checked(p);paths.extend([p,folder/'probe.json'])
        print('VERIFIED_LIVE_RUN',i,flush=True)
    paths.extend([Path(__file__).resolve(),Path(sidecar.__file__).resolve()])
    write_record(ROOT/'completion.json',dict(status='three_corrected_static_runs_complete',flight_tested=False,training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths}))

if __name__=='__main__':asyncio.run(main())
