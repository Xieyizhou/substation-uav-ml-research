"""Integrity-bound receipt for the controlled LiDAR capability challenge."""

from __future__ import annotations

import json
from pathlib import Path

from src.ml.artifacts import file_sha256, git_commit, object_sha256, write_json
from src.study.capability_gate import capability_coverage_report
from src.study.challenge_spec import DEFAULT_CHALLENGE_SPEC
from src.study.registry import ResearchRegistry


CHALLENGE_RECEIPT_SCHEMA_VERSION = 1


def _result_paths(results_dir, study_id, runs):
    root = Path(results_dir) / study_id / "challenge" / "results"
    paths = []
    for run in sorted(runs, key=lambda item: (item["scenario_id"], item["condition"])):
        path = root / f"{run['scenario_id']}__{run['condition']}.json"
        if not path.is_file():
            raise ValueError(f"challenge result is missing: {path.name}")
        paths.append(path)
    return paths


def materialize_challenge_receipt(registry_path, study_id, results_dir):
    registry = ResearchRegistry(registry_path)
    study = registry.get_study(study_id)
    model = registry.get_model(study["candidate_model"])
    runs = registry.run_metrics(study_id, "challenge")
    report = capability_coverage_report(runs, tier="challenge")
    paths = _result_paths(results_dir, study_id, runs)
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    commits = {value.get("code_commit") for value in results}
    if None in commits or len(commits) != 1:
        raise ValueError("challenge results must share one recorded code commit")
    record = {
        "challenge_receipt_schema_version": CHALLENGE_RECEIPT_SCHEMA_VERSION,
        "study_id": study_id,
        "model_id": study["candidate_model"],
        "model_sha256": model["onnx_hash"],
        "software_commit_sha": commits.pop(),
        "challenge_spec_sha256": file_sha256(DEFAULT_CHALLENGE_SPEC),
        "run_config_sha256": object_sha256(sorted({run["config_hash"] for run in runs})),
        "result_sha256": {
            path.name: file_sha256(path) for path in paths
        },
        "capability_report": report,
        "passed": report["passed"],
    }
    record["challenge_receipt_identity_sha256"] = object_sha256(record)
    output = Path(results_dir) / study_id / "challenge" / "challenge_receipt.json"
    write_json(output, record)
    return {**record, "path": str(output)}


def inspect_challenge_receipt(
    path, *, project_root=None, registry_path=None, require_current=False
):
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    supplied = value.pop("challenge_receipt_identity_sha256", None)
    if supplied != object_sha256(value):
        raise ValueError("challenge receipt identity mismatch")
    if value.get("challenge_receipt_schema_version") != CHALLENGE_RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported challenge receipt schema")
    if value.get("challenge_spec_sha256") != file_sha256(DEFAULT_CHALLENGE_SPEC):
        raise ValueError("challenge specification changed after the challenge run")
    result_root = path.parent / "results"
    for name, expected in value.get("result_sha256", {}).items():
        result = result_root / name
        if not result.is_file() or file_sha256(result) != expected:
            raise ValueError(f"challenge result changed after receipt creation: {name}")
    if registry_path is not None:
        registry = ResearchRegistry(registry_path)
        study = registry.get_study(value["study_id"])
        model = registry.get_model(study["candidate_model"])
        if study["candidate_model"] != value.get("model_id"):
            raise ValueError("challenge receipt model no longer matches its study")
        if model["onnx_hash"] != value.get("model_sha256"):
            raise ValueError("challenge receipt model hash no longer matches")
    current = None
    if project_root is not None:
        current = git_commit(Path(project_root)) == value.get("software_commit_sha")
        if require_current and not current:
            raise ValueError("challenge receipt does not match the current clean commit")
    return {
        **value,
        "challenge_receipt_identity_sha256": supplied,
        "current": current,
        "path": str(path),
    }
