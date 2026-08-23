#!/usr/bin/env python3
"""Collect persistent synchronized RGB-D frames from an already running world."""
import argparse, asyncio, hashlib, json, sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sensors.gazebo_rgbd_native import NativeGazeboRgbDepthSource
from src.vision.collection.gazebo_truth import GazeboTruthSource


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ppm_metadata(path):
    with Image.open(path) as image:
        width, height = image.size
        pixels = list(image.convert("L").resize((9, 8)).getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | (pixels[offset + column] > pixels[offset + column + 1])
    return width, height, f"{value:016x}"

async def collect(args):
    frames=args.output/"frames"; source=NativeGazeboRgbDepthSource(frames,emit_stride=args.emit_stride,preserve_frames=True); truth_source=GazeboTruthSource(width=1920,height=1080,topic="auto"); members=[]; truth_rows=[]; first_truth=asyncio.Event(); truth_caught_up=asyncio.Event(); last_rgb_timestamp=None

    async def consume_truth():
        async for event in truth_source.events(timeout_s=args.timeout):
            truth_rows.append(event.truth.to_record()); first_truth.set()
            if last_rgb_timestamp is not None and event.truth.simulation_timestamp>=last_rgb_timestamp:
                truth_caught_up.set()

    await truth_source.start(); truth_task=asyncio.create_task(consume_truth()); await asyncio.wait_for(first_truth.wait(),timeout=args.timeout); await source.start()
    try:
        async for rgb,depth,skew in source.events(timeout_s=args.timeout):
            rgb_relative = Path("frames") / rgb.payload_relative_path
            rgb_path = args.output / rgb_relative
            width, height, perceptual_hash = _ppm_metadata(rgb_path)
            members.append({"frame_id":f"{args.map_id}-{args.seed}-{rgb.sequence:08d}","map_id":args.map_id,"seed":args.seed,"split":args.split,"rgb_timestamp":rgb.capture_timestamp,"depth_timestamp":depth.capture_timestamp,"skew_ms":skew,"rgb_path":str(rgb_relative),"depth_path":str(Path("frames")/depth.payload_relative_path),"image_width":width,"image_height":height,"image_sha256":_sha256(rgb_path),"perceptual_hash":perceptual_hash})
            if len(members)>=args.frames: break
        if members:
            last_rgb_timestamp=members[-1]["rgb_timestamp"]
            if truth_rows[-1]["simulation_timestamp"]>=last_rgb_timestamp: truth_caught_up.set()
            await asyncio.wait_for(truth_caught_up.wait(),timeout=args.timeout)
    finally:
        await source.stop(); truth_task.cancel()
        try: await truth_task
        except asyncio.CancelledError: pass
        await truth_source.stop()
    args.output.mkdir(parents=True,exist_ok=True); truth_path=args.output/"truth.jsonl"; truth_path.write_text("".join(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n" for row in truth_rows),encoding="utf-8")
    receipt={"schema_version":2,"protocol_id":"visual-hard-examples-v2.1","map_id":args.map_id,"seed":args.seed,"split":args.split,"frame_count":len(members),"members":members,"truth":{"relative_path":"truth.jsonl","frame_count":len(truth_rows),"sha256":_sha256(truth_path),"valid_frame_count":sum(row["validation_status"]=="valid" for row in truth_rows)},"source_health":source.receipt()}
    receipt["identity"]=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(",",":")).encode()).hexdigest(); args.output.mkdir(parents=True,exist_ok=True); (args.output/"collection-receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return 0 if len(members)==args.frames else 2
def main():
    p=argparse.ArgumentParser(); p.add_argument("--map-id",choices=("simple","medium"),required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--split",choices=("development","validation"),required=True); p.add_argument("--frames",type=int,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--emit-stride",type=int,default=6); p.add_argument("--timeout",type=float,default=15); return asyncio.run(collect(p.parse_args()))
if __name__=="__main__": raise SystemExit(main())
