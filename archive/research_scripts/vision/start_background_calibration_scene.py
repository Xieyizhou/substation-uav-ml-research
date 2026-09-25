#!/usr/bin/env python3
"""Launch an identity-bound custom background world without catalog mutation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import object_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_id")
    parser.add_argument("--package",type=Path,default=ROOT/"data/research/background_calibration_v1")
    args=parser.parse_args()
    package=args.package.resolve()
    if (package/"PAUSED.json").exists():
        raise ValueError("Background package paused: excluded from product-scope collection")
    manifest=json.loads((package/"manifest.json").read_text())
    identity=manifest.pop("identity")
    if identity!=object_sha256(manifest) or manifest.get("development_only") is not True:
        raise ValueError("Invalid development scene manifest")
    scene=next(s for s in manifest["scenes"] if s["scene_id"]==args.scene_id)
    paths={}
    for key,record in scene["files"].items():
        path=(ROOT/record["path"]).resolve()
        if not path.is_relative_to(package.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=record["sha256"]:
            raise ValueError("Scene artifact path or hash mismatch")
        paths[key]=path
    env={**os.environ,"PROJECT_ROOT":str(ROOT),"MAP_ID":"custom","WORLD_NAME":args.scene_id,"WORLD_SRC":str(paths["world.sdf"]),"OBSTACLE_CONFIG":str(paths["obstacles.json"]),"SIM_MODEL":"x500_research","PX4_GZ_MODEL_POSE":",".join(map(str,scene["spawn_pose"])),"GZ_IP":"127.0.0.1","GZ_PARTITION":"substation_background_calibration","UAV_SANDBOX_DISPLAY_MODE":"headless"}
    print(f"Launching development-only {args.scene_id}; package {identity}. No flight is armed by this entry point.",flush=True)
    os.execve("/bin/bash",["bash",str(ROOT/"scripts/flight/start_px4_substation.sh")],env)


if __name__=="__main__":
    main()
