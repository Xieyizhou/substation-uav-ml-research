"""Read-only summaries for the visual ML experiment lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
import re

from src.vision.evaluation.yolo_package import validate_yolo_package


TRAINING_IDENTITY = Path(
    "data/research/visual_yolo_v2/identity/training_view_identity.json"
)
PACKAGE_ROOT = Path("models/equipment/visual-yolo11n-baseline-v2-package")
BLIND_RESULT = Path(
    "outputs/visual_yolo_v2/paired_blind/paired_heldout_results.json"
)
REPLAY_ROOT = Path("outputs/visual_yolo_v2/static_replay")
REPLAY_416_REPEAT = Path(
    "outputs/visual_yolo_v2/static_replay_replicates/416_every_frame_r01"
    "/results/visual-static-v1-416-every-frame.json"
)
CONDITION_PATTERN = re.compile(
    r"^visual-static-v1-(320|416|640)-(every-frame|every-2nd|every-3rd)$"
)


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{Path(path).name} must contain a JSON object")
    return value


def _relative(root, path):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def _training(project_root):
    path = Path(project_root) / TRAINING_IDENTITY
    if not path.is_file():
        return {"status": "missing"}
    value = _json(path)
    return {
        "status": "complete",
        "identity": value.get("training_view_identity_sha256"),
        "algorithm": value.get("sampling_algorithm"),
        "train_frames": value.get("train_frame_count"),
        "selection_validation_frames": value.get("validation_frame_count"),
        "full_validation_frames": value.get("full_validation_frame_count"),
        "path": _relative(project_root, path),
    }


def _package(project_root):
    root = Path(project_root) / PACKAGE_ROOT
    if not (root / "manifest.json").is_file():
        return {"status": "missing"}
    try:
        value = validate_yolo_package(root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"status": "invalid", "error": str(error)}
    return {
        "status": "complete",
        "identity": value["package_identity_sha256"],
        "architecture": value["architecture"],
        "threshold": value["frozen_confidence_threshold"],
        "exports": sorted(int(size) for size in value["exports"]),
        "path": _relative(project_root, root),
    }


def _model_metrics(value):
    standard = value["standard_metrics"]
    frozen = value["frozen_threshold_metrics"]
    return {
        "precision": standard["precision"],
        "recall": standard["recall"],
        "map50": standard["mAP50"],
        "map50_95": standard["mAP50_95"],
        "macro_f1": frozen["macro_f1"],
        "small_recall": frozen["small_object_recall"],
        "no_target_fpr": frozen["no_target_false_positive_rate"],
        "per_class_recall": {
            name: row["recall"] for name, row in frozen["per_class"].items()
        },
    }


def _blind(project_root):
    path = Path(project_root) / BLIND_RESULT
    if not path.is_file():
        return {"status": "sealed"}
    value = _json(path)
    bootstrap = value["comparison"]["paired_bootstrap"]
    return {
        "status": "complete",
        "frame_count": value["frame_count"],
        "commit": value["evaluation_code_commit_sha"],
        "v1": _model_metrics(value["results"]["v1"]),
        "v2": _model_metrics(value["results"]["v2"]),
        "map_delta": value["comparison"]["mAP50_95_delta_v2_minus_v1"],
        "macro_f1_delta": value["comparison"]["macro_f1_delta_v2_minus_v1"],
        "bootstrap_ci95": bootstrap["macro_f1_delta_ci95"],
        "bootstrap_improvement_probability": bootstrap["probability_v2_improves"],
        "path": _relative(project_root, path),
    }


def _replay_row(path, source="formal"):
    match = CONDITION_PATTERN.match(Path(path).stem)
    if match is None:
        return None
    value = _json(path)
    timing = value["timing_summaries"]["end_to_end_ms"]
    return {
        "condition": Path(path).stem,
        "input_size": int(match.group(1)),
        "policy": match.group(2).replace("-", "_"),
        "status": value["completion_status"],
        "source": source,
        "inferred": value["frame_counts"]["inferred"],
        "skipped": value["frame_counts"]["skipped"],
        "throughput_fps": value["resource_metrics"]["throughput_fps"],
        "precision": value["visual_metrics"]["precision"],
        "recall": value["visual_metrics"]["recall"],
        "small_recall": value["visual_metrics"]["small_object_recall"],
        "p50_ms": timing["p50_ms"],
        "p95_ms": timing["p95_ms"],
        "p99_ms": timing["p99_ms"],
    }


def _replay(project_root):
    root = Path(project_root) / REPLAY_ROOT
    result_root = root / "results"
    rows = [
        row
        for path in sorted(result_root.glob("*.json"))
        if (row := _replay_row(path)) is not None
    ]
    repeat = Path(project_root) / REPLAY_416_REPEAT
    note = None
    if repeat.is_file():
        replacement = _replay_row(repeat, "controlled_replicate")
        rows = [
            replacement if row["condition"] == replacement["condition"] else row
            for row in rows
        ]
        note = "416 every-frame uses the controlled replicate for latency."
    return {
        "status": "complete" if len(rows) == 9 else "partial",
        "completed": sum(row["status"] == "completed" for row in rows),
        "total": 9,
        "rows": rows,
        "note": note,
        "path": _relative(project_root, root),
    }


def research_summary(config):
    root = config.project_root
    stages = {
        "training_view": _training(root),
        "model_package": _package(root),
        "paired_blind": _blind(root),
        "static_replay": _replay(root),
    }
    complete = sum(value["status"] == "complete" for value in stages.values())
    return {"complete_stage_count": complete, "stage_count": 4, **stages}
