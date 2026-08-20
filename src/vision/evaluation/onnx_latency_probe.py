"""Measure first-use and sustained ONNX inference latency."""

from __future__ import annotations

from datetime import datetime, timezone
import platform
from pathlib import Path
import statistics
import time

from src.ml.artifacts import file_sha256, write_json


def _percentile(values, percentile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _predict(model, image_path, *, imgsz, device):
    return model.predict(
        source=str(image_path),
        imgsz=int(imgsz),
        device=str(device),
        conf=0.05,
        verbose=False,
    )


def probe_onnx_latency(
    model_path,
    image_path,
    output_path,
    *,
    imgsz=416,
    device="cpu",
    warm_iterations=200,
):
    """Record model construction plus first inference and warm latency tails."""
    model_path, image_path, output_path = map(
        Path, (model_path, image_path, output_path)
    )
    if not model_path.is_file() or model_path.suffix.lower() != ".onnx":
        raise ValueError("latency probe requires an existing ONNX model")
    if not image_path.is_file():
        raise ValueError("latency probe input image does not exist")
    if int(warm_iterations) < 20:
        raise ValueError("latency probe requires at least 20 warm iterations")

    try:
        import onnxruntime
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("latency probe requires the visual ML environment") from error

    first_start = time.perf_counter()
    model = YOLO(str(model_path), task="detect")
    _predict(model, image_path, imgsz=imgsz, device=device)
    first_use_ms = (time.perf_counter() - first_start) * 1000.0

    warm_ms = []
    for _ in range(int(warm_iterations)):
        started = time.perf_counter()
        _predict(model, image_path, imgsz=imgsz, device=device)
        warm_ms.append((time.perf_counter() - started) * 1000.0)

    result = {
        "onnx_latency_probe_schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": file_sha256(model_path),
        "input_sha256": file_sha256(image_path),
        "input_size": int(imgsz),
        "device": str(device),
        "warm_iterations": len(warm_ms),
        "first_use_ms": first_use_ms,
        "warm_latency_ms": {
            "mean": statistics.fmean(warm_ms),
            "p50": _percentile(warm_ms, 50),
            "p95": _percentile(warm_ms, 95),
            "p99": _percentile(warm_ms, 99),
            "maximum": max(warm_ms),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "ultralytics": ultralytics.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
        "interpretation": (
            "first_use_ms includes model construction and the first prediction; "
            "warm latency reuses the same model and input image"
        ),
    }
    write_json(output_path, result)
    return result
