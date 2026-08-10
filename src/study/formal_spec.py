"""Freeze the candidate, matrix, and comparison contract before formal flight."""

from __future__ import annotations

import json
from pathlib import Path

from src.ml.artifacts import git_commit, object_sha256, write_json
from src.study.comparison import FORMAL_COMPARISONS
from src.study.flight_budget import flight_timeout_policy
from src.study.matrix import FORMAL_CONDITIONS, tier_matrix


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORMAL_SPEC = ROOT / "config/perception/lidar_formal_comparison_v1.json"


def load_formal_spec(path=DEFAULT_FORMAL_SPEC):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("formal_comparison_schema_version") != 1:
        raise ValueError("unsupported formal comparison schema")
    if tuple(value.get("conditions", ())) != FORMAL_CONDITIONS:
        raise ValueError("formal comparison conditions do not match the frozen matrix")
    comparisons = tuple(tuple(item) for item in value.get("comparisons", ()))
    if comparisons != FORMAL_COMPARISONS:
        raise ValueError("formal paired comparisons do not match the frozen protocol")
    bootstrap = value.get("bootstrap", {})
    if bootstrap != {"confidence": 0.95, "samples": 2000, "seed": 17}:
        raise ValueError("formal bootstrap configuration is not frozen v1")
    return value


def freeze_formal_study(
    registry, study_id, results_dir, replay_receipt, spec_path=DEFAULT_FORMAL_SPEC,
    *, qualification_study_id=None,
    flight_timeout_override_s=None,
):
    study = registry.get_study(study_id)
    model = registry.get_model(study["candidate_model"])
    model_manifest = json.loads(model["manifest_json"])
    specification = load_formal_spec(spec_path)
    commit = git_commit(ROOT)
    if commit == "unknown" or commit.endswith("-dirty"):
        raise ValueError("formal study requires a clean tracked commit")
    record = {
        "formal_study_schema_version": 1,
        "study_id": study_id,
        "qualification_study_id": qualification_study_id or study_id,
        "candidate_model_id": study["candidate_model"],
        "candidate_onnx_sha256": model["onnx_hash"],
        "dataset_id": model_manifest["dataset_id"],
        "dataset_sha256": model_manifest["dataset_sha256"],
        "replay_gate_identity_sha256": replay_receipt[
            "replay_gate_identity_sha256"
        ],
        "formal_matrix_identity_sha256": object_sha256(tier_matrix("formal")),
        "comparison_specification": specification,
        "comparison_specification_identity_sha256": object_sha256(specification),
        "execution_code_commit": commit,
        "flight_timeout_policy": flight_timeout_policy(
            flight_timeout_override_s
        ),
    }
    record["formal_study_identity_sha256"] = object_sha256(record)
    path = Path(results_dir) / study_id / "formal/formal_study_receipt.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != record:
            raise ValueError("formal study receipt does not match the frozen execution")
    else:
        write_json(path, record)
    return record
