"""Alternating PNG/memory static probes, identical model and frozen pose."""
import asyncio
from pathlib import Path
from unittest.mock import patch
from scripts.vision import probe_material_shadow as probe
from scripts.vision import material_shadow as sidecar
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256

ROOT=Path('data/research/material-shadow-v1/memory-benchmark-v1').resolve()

async def main():
    sidecar.reference.checked(Path('data/research/material-shadow-v1/memory-offline-v1/completion.json'))
    ROOT.mkdir(parents=True,exist_ok=False);paths=[]
    original=asyncio.create_subprocess_exec
    for i,kind in enumerate(('png','memory','memory','png','png','memory'),1):
        out=ROOT/f'{i:02}-{kind}'
        async def spawn(*args,**kwargs):
            args=tuple('scripts.vision.material_shadow_memory' if x=='scripts.vision.material_shadow' and kind=='memory' else x for x in args)
            return await original(*args,**kwargs)
        with patch.object(probe,'OUT',out),patch.object(asyncio,'create_subprocess_exec',spawn):await probe.main()
        c=sidecar.reference.checked(out/'probe.json')
        if c['status']!='static_live_probe_complete' or not c['owned_processes_exited']:raise ValueError('Probe failed')
        sidecar.reference.checked(out/'live/completion.json');paths.extend([out/'probe.json',out/'live/completion.json'])
        print('BENCHMARK_COMPLETE',i,kind,flush=True)
    paths.extend([Path(__file__).resolve(),Path('src/sensors/gazebo_camera_memory.py').resolve(),Path('scripts/vision/material_shadow_memory.py').resolve()])
    write_record(ROOT/'completion.json',dict(status='six_static_probes_complete',training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths}))

if __name__=='__main__':asyncio.run(main())
