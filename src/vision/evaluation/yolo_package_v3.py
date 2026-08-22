"""Real-domain gate binding for visual model packages."""

import json
from pathlib import Path
import shutil

from src.ml.artifacts import file_sha256


def bind_v3_gate(gate_path, weights, training_view, output_root):
    if gate_path is None:
        return None
    gate_path, output_root = Path(gate_path), Path(output_root)
    gate = json.loads(gate_path.read_text())
    if gate.get("passed") is not True:
        raise ValueError("real-domain v3 package requires a passing gate")
    if gate.get("model_sha256") != file_sha256(weights):
        raise ValueError("real-domain v3 gate references different weights")
    if (
        gate.get("training_view_identity_sha256")
        != training_view.training_view_identity_sha256
    ):
        raise ValueError("real-domain v3 gate references another training view")
    destination = output_root / "real_domain_v3_gate.json"
    shutil.copy2(gate_path, destination)
    return {"path": destination.name, "sha256": file_sha256(destination)}


def validate_v3_gate(root, reference):
    if reference is None:
        return
    root = Path(root)
    gate_path = root / reference.get("path", "")
    if file_sha256(gate_path) != reference.get("sha256"):
        raise ValueError("real-domain v3 gate hash mismatch")
    if json.loads(gate_path.read_text()).get("passed") is not True:
        raise ValueError("real-domain v3 gate is not passing")
