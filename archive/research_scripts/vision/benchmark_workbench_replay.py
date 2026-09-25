"""Compare the original two-pass replay with one-pass replay on a fixed run."""

import argparse
import json
from pathlib import Path
import statistics
import time

from src.ml.artifacts import file_sha256, write_json
from src.sandbox.workbench_models import WorkbenchExperimentRecipe
from src.sandbox.workbench_runner import replay_workbench_model
from src.vision.evaluation.detection_metrics import threshold_metrics
from src.vision.evaluation.yolo_evaluation import collect_predictions
from src.vision.replay.static_runtime import runtime_environment


def benchmark(run_root, output):
    from ultralytics import YOLO

    root = Path(run_root)
    recipe = WorkbenchExperimentRecipe.from_record(json.loads((root / "recipe.json").read_text()))
    view, size = root / "view", recipe.parameters["imgsz"]
    model_path = root / f"model/model_{size}.onnx"
    threshold = json.loads((root / "validation.json").read_text())["confidence_evaluation"]["selected"]["threshold"]
    records, reference = [], None
    destination = Path(output).parent / "one-pass-replay"
    for mode in ("original", "one-pass", "one-pass", "original"):
        started = time.perf_counter()
        if mode == "original":
            model = YOLO(str(model_path))
            for image in sorted((view / "images/validation").iterdir()):
                model.predict(source=str(image), imgsz=size, conf=0.05,
                              device="cpu", verbose=False)
            frames = collect_predictions(model_path, view, "validation", device="cpu", imgsz=size)
            metrics, count = threshold_metrics(frames, threshold), len(frames)
        else:
            result = replay_workbench_model(model_path, view, destination, recipe, threshold)
            metrics, count = result["metrics"], result["frame_count"]
        elapsed = time.perf_counter() - started
        if reference is None:
            reference = metrics
        records.append({"mode": mode, "elapsed_seconds": elapsed, "frame_count": count,
                        "metrics_equal": metrics == reference})
    medians = {mode: statistics.median(row["elapsed_seconds"] for row in records if row["mode"] == mode)
               for mode in ("original", "one-pass")}
    environment, identity = runtime_environment()
    result = {"benchmark_schema_version": 1, "recipe_identity_sha256": recipe.recipe_identity_sha256,
              "model_sha256": file_sha256(model_path), "threshold": threshold,
              "view": json.loads((root / "view.json").read_text()),
              "trials": records, "median_seconds": medians,
              "speedup": medians["original"] / medians["one-pass"],
              "all_metrics_equal": all(row["metrics_equal"] for row in records),
              "runtime": environment, "runtime_identity_sha256": identity,
              "scope": "CPU replay only; ABBA order; includes model startup; two trials per mode"}
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.run, args.output), indent=2))
