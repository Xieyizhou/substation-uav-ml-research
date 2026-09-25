"""Opt-in memory RGB sidecar; same model protocol and no control authority."""
import argparse
from pathlib import Path
from unittest.mock import patch
from scripts.vision import material_shadow as base
from src.sensors.gazebo_camera_memory import GazeboMemoryCameraSource
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run(args):
    original=base.observe
    async def observe(model,out,topic,seconds):
        return await original(model,out,topic,seconds,source_factory=GazeboMemoryCameraSource)
    with patch.object(base,'observe',observe):base.run(args)
    if args.mode=='live':
        out=Path(args.output).resolve()
        paths=[out/'completion.json',Path(__file__).resolve(),Path('src/sensors/gazebo_camera_memory.py').resolve()]
        write_record(out/'memory-implementation.json',dict(status='memory_pipeline_identity_bound',pipeline='canonical_rgb_memory_raw_evidence',training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('preflight','smoke','live'),default='preflight')
    p.add_argument('--seed',type=int,choices=(7,17,27),default=7);p.add_argument('--topic',default='auto')
    p.add_argument('--seconds',type=float,default=30);p.add_argument('--output',default='data/research/material-shadow-v1/memory-run-001')
    run(p.parse_args())
