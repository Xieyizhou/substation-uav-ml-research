"""New bounded comparison after diagnosing eager decode contention."""
import asyncio
from pathlib import Path
from unittest.mock import patch
from scripts.vision import probe_material_shadow as probe
from scripts.vision import material_shadow as sidecar
from src.vision.canonical.plan import write_record
from src.ml.artifacts import file_sha256

ROOT=Path('data/research/material-shadow-v1/latest-benchmark-v1').resolve()

async def main():
    offline=Path('data/research/material-shadow-v1/latest-offline-v1/completion.json').resolve();sidecar.reference.checked(offline)
    ROOT.mkdir(parents=True,exist_ok=False);paths=[offline];original=asyncio.create_subprocess_exec
    for i,kind in enumerate(('png','latest','latest','png','png','latest'),1):
        out=ROOT/f'{i:02}-{kind}'
        async def spawn(*args,**kw):
            return await original(*(('scripts.vision.material_shadow_latest' if x=='scripts.vision.material_shadow' and kind=='latest' else x) for x in args),**kw)
        with patch.object(probe,'OUT',out),patch.object(asyncio,'create_subprocess_exec',spawn):await probe.main()
        c=sidecar.reference.checked(out/'probe.json')
        if c['status']!='static_live_probe_complete' or not c['owned_processes_exited']:raise ValueError('Probe failed')
        sidecar.reference.checked(out/'live/completion.json');paths.extend([out/'probe.json',out/'live/completion.json'])
        if kind=='latest':sidecar.reference.checked(out/'live/latest-implementation.json');paths.append(out/'live/latest-implementation.json')
        print('BENCHMARK_COMPLETE',i,kind,flush=True)
    paths.append(Path(__file__).resolve())
    write_record(ROOT/'completion.json',dict(status='six_static_probes_complete',training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths}))

if __name__=='__main__':asyncio.run(main())
