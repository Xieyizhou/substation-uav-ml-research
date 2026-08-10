"""Materialize and evaluate one identity-bound paired blind comparison."""

from __future__ import annotations

import json
from pathlib import Path
import random

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.evaluation.detection_metrics import threshold_metrics
from src.vision.evaluation.heldout_view import (
    _heldout_dataset_yaml,
    load_heldout_source,
)
from src.vision.evaluation.yolo_evaluation import (
    PREDICTION_CONFIDENCE_FLOOR,
    _formal_commit,
    _require_finite,
    _standard_metrics,
    collect_predictions,
)
from src.vision.evaluation.yolo_package import validate_yolo_package
from src.vision.training.yolo_dataset import materialize_yolo_partition


CANDIDATE_ORDER = ("v1", "v2")
BOOTSTRAP_SEED = 17
BOOTSTRAP_REPETITIONS = 2000


def _candidate(package_root):
    package_root = Path(package_root)
    package = validate_yolo_package(package_root)
    export = package["exports"]["640"]
    return {
        "package_root": str(package_root.resolve()),
        "package_identity_sha256": package["package_identity_sha256"],
        "canonical_model_relative_path": export["path"],
        "canonical_model_sha256": export["sha256"],
        "frozen_confidence_threshold": package["frozen_confidence_threshold"],
    }


def materialize_paired_heldout_view(
    collection_root, v1_package_root, v2_package_root, output_root
):
    output_root = Path(output_root)
    receipt_path = output_root / "identity/paired_heldout_access_receipt.json"
    candidates = {
        "v1": _candidate(v1_package_root),
        "v2": _candidate(v2_package_root),
    }
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text())
        identities = {
            name: existing["candidates"][name]["package_identity_sha256"]
            for name in CANDIDATE_ORDER
        }
        requested = {
            name: candidates[name]["package_identity_sha256"]
            for name in CANDIDATE_ORDER
        }
        if identities != requested:
            raise ValueError("paired blind view was unlocked for other packages")
        return {"receipt": existing, "dataset_yaml": str(output_root / "dataset.yaml")}
    heldout, rows = load_heldout_source(collection_root)
    if heldout.dataset_version != "visual-multiscenario-png-v2":
        raise ValueError("paired comparison requires the v2 blind dataset")
    result = materialize_yolo_partition(
        "heldout_test", rows, Path(collection_root), output_root
    )
    write_json(
        output_root / "identity/held_out_test_dataset_identity.json",
        heldout.to_record(),
    )
    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text(_heldout_dataset_yaml(output_root), encoding="utf-8")
    receipt = {
        "paired_heldout_access_schema_version": 1,
        "candidate_order": list(CANDIDATE_ORDER),
        "heldout_dataset_identity_sha256": heldout.dataset_identity_sha256,
        "membership_sha256": result["membership_sha256"],
        "frame_count": result["frame_count"],
        "canonical_input_size": 640,
        "candidates": candidates,
    }
    receipt["paired_heldout_access_identity_sha256"] = object_sha256(receipt)
    write_json(receipt_path, receipt)
    return {
        "receipt": receipt,
        "dataset_yaml": str(yaml_path),
        "link_modes": result["link_modes"],
    }


def _write_predictions(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for frame in frames:
            output.write(json.dumps(frame, sort_keys=True, separators=(",", ":")))
            output.write("\n")
    return file_sha256(path)


def _membership_recordings(dataset_root):
    path = Path(dataset_root) / "identity/heldout_test_membership.jsonl"
    result = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            stem = Path(row["image_relative_path"]).stem
            result[stem] = row["recording_id"]
    return result


def _recording_counts(frames, threshold, recordings):
    grouped = {}
    for frame in frames:
        recording = recordings.get(frame["sample_id"])
        if recording is None:
            raise ValueError("prediction membership is absent from the blind receipt")
        grouped.setdefault(recording, []).append(frame)
    return {
        name: threshold_metrics(rows, threshold)["per_class"]
        for name, rows in grouped.items()
    }


def _macro_f1(counts, draw):
    values = []
    for class_name in next(iter(counts.values())):
        totals = {key: 0 for key in ("tp", "fp", "fn")}
        for recording in draw:
            for key in totals:
                totals[key] += counts[recording][class_name][key]
        precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
        recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
        values.append(2 * precision * recall / max(precision + recall, 1e-12))
    return sum(values) / len(values)


def _paired_bootstrap(counts):
    recording_ids = sorted(counts["v1"])
    if recording_ids != sorted(counts["v2"]):
        raise ValueError("paired candidates do not cover the same recordings")
    generator = random.Random(BOOTSTRAP_SEED)
    deltas = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        draw = [generator.choice(recording_ids) for _ in recording_ids]
        deltas.append(
            _macro_f1(counts["v2"], draw) - _macro_f1(counts["v1"], draw)
        )
    ordered = sorted(deltas)
    return {
        "unit": "recording",
        "seed": BOOTSTRAP_SEED,
        "repetitions": BOOTSTRAP_REPETITIONS,
        "macro_f1_delta_mean": sum(deltas) / len(deltas),
        "macro_f1_delta_ci95": [ordered[49], ordered[1950]],
        "probability_v2_improves": sum(value > 0 for value in deltas) / len(deltas),
    }


def _read_receipt(dataset_root):
    receipt_path = dataset_root / "identity/paired_heldout_access_receipt.json"
    if not receipt_path.is_file():
        raise ValueError("paired blind evaluation requires an access receipt")
    receipt = json.loads(receipt_path.read_text())
    supplied = receipt.get("paired_heldout_access_identity_sha256")
    unsigned = {
        key: value
        for key, value in receipt.items()
        if key != "paired_heldout_access_identity_sha256"
    }
    if supplied != object_sha256(unsigned):
        raise ValueError("paired blind access receipt identity mismatch")
    return receipt, supplied


def _evaluate_candidate(name, candidate, dataset_root, output_root, device, recordings):
    model = (
        Path(candidate["package_root"])
        / candidate["canonical_model_relative_path"]
    )
    if file_sha256(model) != candidate["canonical_model_sha256"]:
        raise ValueError(f"{name} canonical model hash mismatch")
    standard = _standard_metrics(
        model, dataset_root / "dataset.yaml", split="test", device=device, imgsz=640
    )
    frames = collect_predictions(
        model, dataset_root, "heldout_test", device=device, imgsz=640,
        confidence=PREDICTION_CONFIDENCE_FLOOR, batch=1,
    )
    threshold = candidate["frozen_confidence_threshold"]
    prediction_path = output_root / "predictions" / f"{name}.jsonl"
    result = {
        "model_sha256": candidate["canonical_model_sha256"],
        "frozen_confidence_threshold": threshold,
        "standard_metrics": standard,
        "frozen_threshold_metrics": threshold_metrics(frames, threshold),
        "prediction_manifest_sha256": _write_predictions(prediction_path, frames),
    }
    return result, [frame["sample_id"] for frame in frames], _recording_counts(
        frames, threshold, recordings
    )


def _comparison(results, counts):
    return {
        "macro_f1_delta_v2_minus_v1": (
            results["v2"]["frozen_threshold_metrics"]["macro_f1"]
            - results["v1"]["frozen_threshold_metrics"]["macro_f1"]
        ),
        "mAP50_95_delta_v2_minus_v1": (
            results["v2"]["standard_metrics"]["mAP50_95"]
            - results["v1"]["standard_metrics"]["mAP50_95"]
        ),
        "paired_bootstrap": _paired_bootstrap(counts),
    }


def evaluate_paired_heldout(dataset_root, output_root, *, device="cpu"):
    dataset_root, output_root = Path(dataset_root), Path(output_root)
    result_path = output_root / "paired_heldout_results.json"
    if result_path.exists():
        raise ValueError("paired blind evaluation result already exists")
    receipt, receipt_identity = _read_receipt(dataset_root)
    evaluation_commit = _formal_commit()
    results, counts, expected_order = {}, {}, None
    recordings = _membership_recordings(dataset_root)
    for name in receipt["candidate_order"]:
        results[name], order, counts[name] = _evaluate_candidate(
            name, receipt["candidates"][name], dataset_root, output_root,
            device, recordings,
        )
        if expected_order is not None and order != expected_order:
            raise ValueError("paired candidates produced a different frame order")
        expected_order = order
    output = {
        "paired_heldout_evaluation_schema_version": 1,
        "access_receipt_identity_sha256": receipt_identity,
        "evaluation_code_commit_sha": evaluation_commit,
        "prediction_confidence_floor": PREDICTION_CONFIDENCE_FLOOR,
        "frame_count": receipt["frame_count"],
        "candidate_order": receipt["candidate_order"],
        "results": results,
        "comparison": _comparison(results, counts),
    }
    _require_finite(output)
    write_json(result_path, output)
    return output
