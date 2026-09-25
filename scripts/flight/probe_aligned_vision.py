"""Unarmed PX4/Gazebo YOLO-depth-estimator timing and geometry probe."""

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from scripts.flight import sandbox_replan_lifecycle as lifecycle
from scripts.vision.locked_cpu_threads import locked_threads
from src.flight.aligned_sim_rgbd import AlignedSimRgbdSource
from src.ml.artifacts import file_sha256, write_json
from src.sandbox.live_replan_gate import require_repeat_gate
from src.sandbox.workbench_inference import verified_model
from src.sensors.gazebo_depth_memory import aligned_depth, bbox_depth
from src.vision.localization.rgbd_localization import localize_bbox


async def probe(output, model_run):
    from ultralytics import YOLO
    require_repeat_gate(lifecycle.ROOT)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    info = verified_model(model_run)
    model = YOLO(str(info["onnx_path"]), task="detect")
    model.predict(source=np.zeros((info["imgsz"], info["imgsz"], 3), np.uint8),
                  imgsz=info["imgsz"], rect=False, device="cpu", verbose=False)
    rows, counts = [], {}

    async def hook(drone, state, origin):
        source = AlignedSimRgbdSource(drone, gazebo_sitl=True)
        def infer(event):
            result = model.predict(source=np.ascontiguousarray(event.decoded.image.array[:, :, ::-1]),
                                   imgsz=info["imgsz"], conf=info["threshold"], rect=False,
                                   device="cpu", iou=.7, verbose=False)[0]
            return [] if result.boxes is None else [
                dict(class_name=result.names[int(cls)], confidence=conf, bbox=box)
                for cls, conf, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist())]
        async def consume():
            async for event, depth, match in source.events():
                predictions = await asyncio.to_thread(infer, event)
                now = time.monotonic()
                # Revalidate freshness after inference before any planning use.
                aligned_depth(event.frame, depth, now=now)
                match = source.pose.match(event.frame, now=now)
                intrinsics = dict(fx=event.frame.width/(2*math.tan(1.466/2)),
                                  fy=event.frame.width/(2*math.tan(1.466/2)),
                                  cx=event.frame.width/2, cy=event.frame.height/2)
                observations = []
                for prediction in predictions:
                    distance = bbox_depth(depth, (event.frame.width, event.frame.height), prediction["bbox"])
                    if distance is not None:
                        observations.append({**prediction, "depth_m": distance,
                            "local_position": localize_bbox(distance, prediction["bbox"], intrinsics, match["pose"],
                                dict(forward_m=.18, right_m=0, down_m=-.12))})
                rows.append(dict(frame=event.frame.to_record(), rgb_sha256=event.decoded.image.decoded_content_sha256,
                                 depth_sha256=depth.payload_sha256, depth_timestamp=depth.capture_timestamp,
                                 pose_match=match, result_monotonic=now, predictions=predictions, observations=observations))
                if len(rows) >= 100:
                    return
        try:
            await source.start()
            try:
                await asyncio.wait_for(consume(), 20)
            except asyncio.TimeoutError:
                pass
            if len(rows) < 20:
                raise RuntimeError(f"insufficient synchronized frames: {len(rows)}")
            if state.get("armed", (0, True))[1] is not False:
                raise RuntimeError("unarmed state lost")
        finally:
            await source.stop()
            counts.update(source.counts)
            write_json(output / "observations.json", dict(rows=rows, counts=counts, control_authority="none"))

    lifecycle.OUT = output / "runtime"
    base = lifecycle.ROOT / "data/research/material-shadow-v1/autonomy-avoidance-v1"
    os.environ.update(PX4_GZ_MAG_ENU_GAUSS="1", PX4_PARAM_EKF2_DECL_TYPE="3", PX4_PARAM_EKF2_MAG_DECL="0")
    await lifecycle.main(fly=False, preflight_hook=hook,
                         server_config=base / "magnetic-contract-001/server.config",
                         source_world=base / "envelope-aware-scene-v1/world.sdf")
    runtime = json.loads((lifecycle.OUT / "receipt.json").read_text())
    paths = [Path(__file__), Path("src/flight/sim_pose_sync.py"), Path("src/flight/aligned_sim_rgbd.py"),
             Path("src/sensors/gazebo_depth_memory.py"), output / "observations.json", lifecycle.OUT / "receipt.json",
             lifecycle.PX4 / "src/modules/simulation/gz_bridge/GZBridge.cpp",
             *[lifecycle.PX4 / f"src/modules/mavlink/streams/{name}.hpp" for name in ("ATTITUDE", "LOCAL_POSITION_NED")]]
    result = dict(state="complete" if runtime["status"] == "unarmed_health_check_passed" else "failed",
                  error=runtime["error"], arm_commands_sent=0, flight_tested=False,
                  model_sha256=info["onnx_model_sha256"], frames=len(rows), counts=counts,
                  localized_observations=sum(len(row["observations"]) for row in rows),
                  owned_processes_exited=runtime["owned_processes_exited"],
                  inputs={str(path.resolve()): file_sha256(path) for path in paths if path.is_file()})
    write_json(output / "probe.json", result)
    if result["state"] != "complete":
        raise RuntimeError(result["error"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-run", type=Path, required=True)
    args = parser.parse_args()
    with locked_threads(4):
        print(json.dumps(asyncio.run(probe(args.output, args.model_run)), indent=2))
