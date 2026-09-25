"""Prepare lazy model dependencies before starting the MAVSDK/gRPC runtime."""

import argparse
import runpy
import time

from src.sandbox.live_replan_stop import validate_run_id
from src.sandbox.visual_runtime_model import visual_runtime_model


def prepare_model_runtime():
    import numpy as np
    from ultralytics import YOLO
    from scripts.vision.locked_cpu_threads import locked_threads
    info = visual_runtime_model("material-routed-480-7")
    started = time.monotonic()
    # Ultralytics lazily imports plotting/font helpers on its first prediction.
    # A cold Matplotlib font cache can spawn discovery processes. Finish this
    # work before MAVSDK creates gRPC threads; no simulator is connected here.
    with locked_threads(4):
        model = YOLO(str(info["model_path"]), task="detect")
        model.predict(source=np.zeros((info["imgsz"], info["imgsz"], 3), np.uint8),
                      imgsz=info["imgsz"], rect=False, device="cpu", verbose=False)
    print(f"Visual model dependencies prepared before MAVSDK in {time.monotonic()-started:.3f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fly", action="store_true", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    validate_run_id(args.run_id)
    prepare_model_runtime()
    runpy.run_module("scripts.flight.fly_visual_replan", run_name="__main__")


if __name__ == "__main__":
    main()
