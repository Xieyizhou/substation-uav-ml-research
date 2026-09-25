"""Bounded 240-second read-only sidecar for independent SITL route trials."""
import argparse
import asyncio
from pathlib import Path
from scripts.vision import material_shadow as base
from src.sensors.gazebo_camera_latest import GazeboLatestMemorySource
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def validate_duration(seconds):
    if not 0<seconds<=240:raise ValueError('Route sidecar duration must be in (0,240]')

def run(args):
    validate_duration(args.seconds)
    identity=base.model_identity(args.seed);out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    write_record(out/'model.json',dict(**identity,duration_s=args.seconds,inputs={str(p):file_sha256(p) for p in (Path(__file__).resolve(),Path(base.__file__).resolve(),Path(identity['weights']),Path(identity['receipt']))}))
    try:
        from ultralytics import YOLO
        with base.locked_threads(4):
            model=YOLO(identity['weights']);write_record(out/'warmup.json',base.warmup(model))
            counts=asyncio.run(base.observe(model,out,args.topic,args.seconds,source_factory=GazeboLatestMemorySource))
        write_record(out/'completion.json',dict(status='live_complete_not_flight_certified',counts=counts,control_authority='none',training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in out.rglob('*') if p.is_file()}))
    except BaseException as exc:
        write_record(out/'failure.json',dict(status='sidecar_failed_no_flight_action',error=f'{type(exc).__name__}: {exc}',control_authority='none',training_admitted=False,promotable=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,required=True);p.add_argument('--output',required=True);p.add_argument('--topic',default='/research_camera/image');p.add_argument('--seed',type=int,choices=(7,17,27),default=7);run(p.parse_args())
