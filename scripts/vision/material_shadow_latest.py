"""Latest-message-first memory sidecar; inference and evidence remain unchanged."""
import argparse
from pathlib import Path
from unittest.mock import patch
from scripts.vision import material_shadow_memory as memory
from src.sensors.gazebo_camera_latest import GazeboLatestMemorySource
from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import write_record

def run(args):
    with patch.object(memory,'GazeboMemoryCameraSource',GazeboLatestMemorySource):memory.run(args)
    if args.mode=='live':
        out=Path(args.output).resolve();paths=[out/'memory-implementation.json',Path(__file__).resolve(),Path('src/sensors/gazebo_camera_latest.py').resolve()]
        write_record(out/'latest-implementation.json',dict(status='latest_before_decode',training_admitted=False,promotable=False,inputs={str(p):file_sha256(p) for p in paths}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('preflight','smoke','live'),default='preflight');p.add_argument('--seed',type=int,choices=(7,17,27),default=7);p.add_argument('--topic',default='auto');p.add_argument('--seconds',type=float,default=30);p.add_argument('--output',default='data/research/material-shadow-v1/latest-run-001');run(p.parse_args())
