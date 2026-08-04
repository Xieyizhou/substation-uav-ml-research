"""Runtime summaries for static visual replay."""

from __future__ import annotations

import importlib.metadata
import platform
import sys

from src.ml.artifacts import object_sha256


def timing_summary(values):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)

    def percentile(fraction):
        index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
        return ordered[index]

    return {
        "count": len(ordered), "min_ms": ordered[0], "max_ms": ordered[-1],
        "mean_ms": sum(ordered) / len(ordered),
        "p50_ms": percentile(0.50), "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
    }


def runtime_environment():
    values = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "ultralytics": importlib.metadata.version("ultralytics"),
        "onnxruntime": importlib.metadata.version("onnxruntime"),
    }
    return values, object_sha256(values)
