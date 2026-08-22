#!/usr/bin/env python3
"""Collect persistent synchronized RGB-D frames from an already running world."""
import argparse, asyncio, hashlib, json
from pathlib import Path
from src.sensors.gazebo_rgbd_native import NativeGazeboRgbDepthSource

async def collect(args):
    frames=args.output/"frames"; source=NativeGazeboRgbDepthSource(frames,emit_stride=args.emit_stride,preserve_frames=True); await source.start(); members=[]
    try:
        async for rgb,depth,skew in source.events(timeout_s=args.timeout):
            members.append({"frame_id":f"{args.map_id}-{args.seed}-{rgb.sequence:08d}","map_id":args.map_id,"seed":args.seed,"split":args.split,"rgb_timestamp":rgb.capture_timestamp,"depth_timestamp":depth.capture_timestamp,"skew_ms":skew,"rgb_path":str(Path("frames")/rgb.payload_relative_path),"depth_path":str(Path("frames")/depth.payload_relative_path)})
            if len(members)>=args.frames: break
    finally: await source.stop()
    receipt={"schema_version":1,"protocol_id":"visual-hard-examples-v2.1","map_id":args.map_id,"seed":args.seed,"split":args.split,"frame_count":len(members),"members":members,"source_health":source.receipt()}
    receipt["identity"]=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(",",":")).encode()).hexdigest(); args.output.mkdir(parents=True,exist_ok=True); (args.output/"collection-receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return 0 if len(members)==args.frames else 2
def main():
    p=argparse.ArgumentParser(); p.add_argument("--map-id",choices=("simple","medium"),required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--split",choices=("development","validation"),required=True); p.add_argument("--frames",type=int,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--emit-stride",type=int,default=6); p.add_argument("--timeout",type=float,default=15); return asyncio.run(collect(p.parse_args()))
if __name__=="__main__": raise SystemExit(main())
