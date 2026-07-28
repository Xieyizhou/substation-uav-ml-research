"""ONNX LiDAR risk runtime with a strict, versioned output contract."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

from src.ml import RISK_LABELS


class OnnxRiskModel:
    def __init__(self, model_path, scan_size=360):
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "ONNX risk inference requires requirements-ml.txt"
            ) from error
        self.np = np
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self.scan_size = int(scan_size)
        self.input_name = self.session.get_inputs()[0].name
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.model_id = metadata.get("model_id", self.model_path.stem)
        self.schema_version = int(metadata.get("schema_version", 1))
        if self.schema_version != 1:
            raise ValueError(f"unsupported ONNX risk schema {self.schema_version}")

    def _input(self, scan):
        values = [
            min(max(value, scan.range_min_m), scan.range_max_m) / scan.range_max_m
            if math.isfinite(value)
            else 1.0
            for value in scan.ranges_m
        ]
        indexes = self.np.linspace(0, len(values) - 1, self.scan_size)
        resized = self.np.interp(indexes, self.np.arange(len(values)), values)
        return resized.astype("float32")[None, None, :]

    def predict(self, scan):
        started = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: self._input(scan)})
        if len(outputs) < 3:
            raise ValueError(
                "risk model must output risk_logits, traversability, and direction"
            )
        logits = self.np.asarray(outputs[0])[0]
        shifted = logits - logits.max()
        probabilities = self.np.exp(shifted) / self.np.exp(shifted).sum()
        class_index = int(probabilities.argmax())
        traversability = self.np.asarray(outputs[1])[0].reshape(-1)
        direction = float(self.np.asarray(outputs[2]).reshape(-1)[0])
        uncertainty = (
            float(self.np.asarray(outputs[3]).reshape(-1)[0])
            if len(outputs) > 3
            else float(1.0 - probabilities[class_index])
        )
        return {
            "risk_level": RISK_LABELS[class_index],
            "confidence": float(probabilities[class_index]),
            "traversability": traversability.tolist(),
            "recommended_direction_deg": direction,
            "uncertainty": uncertainty,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "model_id": self.model_id,
        }
