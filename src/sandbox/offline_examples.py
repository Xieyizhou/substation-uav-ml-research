"""Small offline workflows using the existing dataset audit and detection metrics."""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import sys

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, git_commit, object_sha256, write_json
from src.vision.evaluation.detection_metrics import threshold_metrics

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "examples/fixtures"


def _new_output(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def _receipt(output, workflow, inputs):
    outputs = []
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise ValueError("An output artifact cannot be a symbolic link")
        if path.is_file():
            outputs.append(dict(path=path.relative_to(output).as_posix(),
                                bytes=path.stat().st_size, sha256=file_sha256(path)))
    value = dict(
        example_receipt_schema_version=1, workflow=workflow,
        dataset_role="demo", formal_evidence=False, source_commit_sha=git_commit(ROOT),
        inputs=inputs, outputs=outputs,
    )
    value["receipt_identity_sha256"] = object_sha256(value)
    return write_json(output / "receipt.json", value)


def verify_example_receipt(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Example receipt must be an object")
    identity = value.pop("receipt_identity_sha256", None)
    if identity != object_sha256(value):
        raise ValueError("Example receipt identity changed")
    if (value.get("example_receipt_schema_version") != 1
            or value.get("dataset_role") != "demo"
            or value.get("formal_evidence") is not False):
        raise ValueError("An offline demonstration receipt is required")
    outputs = value.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Receipt has no output artifacts")
    seen = set()
    for row in outputs:
        if not isinstance(row, dict):
            raise ValueError("Receipt output records must be objects")
        name = row.get("path")
        relative = PurePosixPath(name) if isinstance(name, str) else None
        if (not name or relative.is_absolute() or ".." in relative.parts
                or "\\" in name or relative.as_posix() != name or name in seen):
            raise ValueError("Unsafe or duplicate receipt output path")
        seen.add(name)
        artifact = path.parent / name
        if (artifact.is_symlink() or not artifact.is_file()
                or not artifact.resolve().is_relative_to(path.parent)
                or artifact.stat().st_size != row.get("bytes")
                or file_sha256(artifact) != row.get("sha256")):
            raise ValueError("Recorded output changed: " + name)
    return dict(valid=True, workflow=value["workflow"], checked_outputs=len(outputs),
                receipt_identity_sha256=identity, dataset_role="demo", formal_evidence=False,
                checks=["receipt_identity", "recorded_output_files"], inputs_reopened=False)


def run_dataset_audit(output):
    from src.sandbox.workbench_datasets import import_yolo_dataset

    output = _new_output(output)
    imported = import_yolo_dataset(FIXTURES / "dataset", output / "datasets", "format-demo")
    report = dict(
        offline_example_schema_version=1, workflow="dataset_audit",
        dataset_role="demo", formal_evidence=False,
        fixture_origin="project_generated_geometric_format_samples",
        dataset_identity_sha256=imported["dataset_identity_sha256"],
        split_counts=imported["split_counts"], class_counts=imported["class_counts"],
        duplicate_image_check="passed", labels_and_boxes_structurally_valid=True,
        independently_reviewed_equipment_labels=False, physical_site_independence_verified=False,
    )
    write_json(output / "audit.json", report)
    inputs = [dict(path=path.relative_to(FIXTURES).as_posix(), sha256=file_sha256(path))
              for path in sorted((FIXTURES / "dataset").rglob("*")) if path.is_file()]
    receipt = _receipt(output, "dataset_audit", inputs)
    return dict(output=str(output), audit=str(output / "audit.json"),
                receipt=str(receipt), verification=verify_example_receipt(receipt))


def _box(item, width, height):
    if item.get("class_name") not in EQUIPMENT_CLASSES:
        raise ValueError("Unsupported equipment class in cached predictions")
    box = item.get("bbox")
    if (not isinstance(box, list) or len(box) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in box)):
        raise ValueError("Each cached box needs four finite pixel coordinates")
    left, top, right, bottom = box
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Cached box extends outside the image")


def _predictions(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (value.get("prediction_fixture_schema_version") != 1
            or value.get("dataset_role") != "demo"
            or value.get("formal_evidence") is not False
            or value.get("prediction_origin") != "manually_constructed_cached_predictions"):
        raise ValueError("This example requires explicitly constructed demonstration predictions")
    frames = value.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Cached predictions must contain frames")
    seen = set()
    for frame in frames:
        identifier = frame.get("sample_id")
        if (not isinstance(identifier, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in seen):
            raise ValueError("Frame identifiers must be nonempty and unique")
        seen.add(identifier)
        width, height = frame.get("width"), frame.get("height")
        if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0):
            raise ValueError("Positive image dimensions are required")
        for truth in frame["truth"]:
            _box(truth, width, height)
            if type(truth.get("small")) is not bool:
                raise ValueError("The small-object convention must be explicit")
        for prediction in frame["predictions"]:
            _box(prediction, width, height)
            confidence = prediction.get("confidence")
            if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                    or not math.isfinite(confidence) or not 0 <= confidence <= 1):
                raise ValueError("Prediction confidence must be between zero and one")
    return frames


def compare_prediction_files(baseline_path, candidate_path, threshold=0.5):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between zero and one")
    baseline, candidate = _predictions(baseline_path), _predictions(candidate_path)
    membership = lambda rows: [
        {key: row[key] for key in ("sample_id", "width", "height", "image_sha256", "truth")}
        for row in rows
    ]
    if membership(baseline) != membership(candidate):
        raise ValueError("Comparison requires the same ordered images and ground truth")
    first, second = threshold_metrics(baseline, threshold), threshold_metrics(candidate, threshold)
    deltas = {key: None if first[key] is None or second[key] is None else second[key] - first[key]
              for key in ("macro_f1", "small_object_recall", "no_target_false_positive_rate")}
    return dict(
        offline_example_schema_version=1, dataset_role="demo", formal_evidence=False,
        prediction_origin="manually_constructed_cached_predictions",
        new_model_inference_performed=False, frame_count=len(baseline), threshold=threshold,
        membership_sha256=object_sha256(membership(baseline)),
        small_object_definition="box area below 32 squared pixels in the 320 by 240 fixture image",
        baseline=first, candidate=second, deltas=deltas,
        scope="Demonstrates matching and comparison; the constructed scores do not measure equipment recognition.",
    )


def run_prediction_comparison(output, threshold=0.5):
    from PIL import Image
    from src.sandbox.workbench_inference import _annotate

    baseline, candidate = FIXTURES / "baseline.json", FIXTURES / "candidate.json"
    report = compare_prediction_files(baseline, candidate, threshold)
    output = _new_output(output)
    write_json(output / "comparison.json", report)
    with (output / "per_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "class_name", "tp", "fp", "fn", "precision", "recall", "f1"])
        writer.writeheader()
        for name in ("baseline", "candidate"):
            for class_name, values in report[name]["per_class"].items():
                writer.writerow(dict(model=name, class_name=class_name, **values))
    images = []
    for name, source in (("baseline", baseline), ("candidate", candidate)):
        for frame in _predictions(source):
            image_path = (FIXTURES / frame["image_relative_path"]).resolve()
            if (not image_path.is_relative_to(FIXTURES.resolve())
                    or file_sha256(image_path) != frame["image_sha256"]):
                raise ValueError("Fixture image differs from cached prediction membership")
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            detections = [
                dict(class_id=EQUIPMENT_CLASSES.index(item["class_name"]),
                     class_name=item["class_name"], confidence=item["confidence"],
                     xyxy_pixels=item["bbox"])
                for item in frame["predictions"] if item["confidence"] >= threshold
            ]
            filename = name + "-" + frame["sample_id"] + ".png"
            _annotate(image, detections, output / filename)
            images.append((name, frame["sample_id"], filename))
    cells = "".join(
        "<figure><figcaption>" + escape(name + " / " + sample)
        + '</figcaption><img width="320" height="240" src="' + escape(filename)
        + '" alt="Constructed detection boxes"></figure>'
        for name, sample, filename in images
    )
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Cached prediction comparison</title><style>'
        'body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;color:#20352f}'
        'section{display:flex;flex-wrap:wrap;gap:18px}figure{margin:0}figcaption{padding:8px 0}'
        'pre{background:#f2f6f4;padding:16px;overflow:auto}</style>'
        '<h1>Cached prediction comparison</h1>'
        '<p>Geometric format fixtures and manually constructed predictions. '
        'No model inference or real equipment accuracy is represented.</p>'
        '<p>Shared confidence threshold: ' + escape(str(threshold)) + '</p><pre>'
        + escape(json.dumps({"deltas": report["deltas"], "frame_count": report["frame_count"]}, indent=2))
        + '</pre><section>' + cells + '</section></html>'
    )
    (output / "comparison.html").write_text(html, encoding="utf-8")
    receipt = _receipt(output, "prediction_comparison", [
        dict(path=path.name, sha256=file_sha256(path)) for path in (baseline, candidate)
    ])
    return dict(output=str(output), report=str(output / "comparison.html"),
                comparison=str(output / "comparison.json"), receipt=str(receipt),
                verification=verify_example_receipt(receipt))


def run_receipt_check(output):
    output = _new_output(output)
    original = output / "original"
    original.mkdir()
    write_json(original / "result.json", dict(dataset_role="demo", value=4))
    receipt = _receipt(original, "receipt_check", [])
    valid = verify_example_receipt(receipt)
    altered = output / "altered"
    shutil.copytree(original, altered)
    write_json(altered / "result.json", dict(dataset_role="demo", value=5))
    try:
        verify_example_receipt(altered / "receipt.json")
    except ValueError as error:
        rejection = str(error)
    else:
        raise AssertionError("The altered output was incorrectly accepted")
    report = dict(original=valid, altered_output_rejected=True, rejection=rejection,
                  dataset_role="demo", formal_evidence=False)
    write_json(output / "tamper_check.json", report)
    return dict(output=str(output), report=str(output / "tamper_check.json"), **report)


def example_main(workflow):
    parser = argparse.ArgumentParser(description={
        "dataset_audit": "Audit and import tiny geometric YOLO format fixtures.",
        "prediction_comparison": "Compare constructed cached predictions without a model or simulator.",
        "receipt_check": "Verify artifact hashes and demonstrate rejection of an altered output.",
    }[workflow])
    parser.add_argument("--output", type=Path)
    if workflow == "prediction_comparison":
        parser.add_argument("--threshold", type=float, default=0.5)
    if workflow == "receipt_check":
        parser.add_argument("--receipt", type=Path, help="Read-only check of an existing demonstration receipt.")
    args = parser.parse_args()
    try:
        if workflow == "receipt_check" and args.receipt is not None:
            if args.output is not None:
                parser.error("Choose --receipt or --output")
            result = verify_example_receipt(args.receipt)
        else:
            if args.output is None:
                parser.error("--output is required for a new example run")
            if workflow == "dataset_audit":
                result = run_dataset_audit(args.output)
            elif workflow == "prediction_comparison":
                result = run_prediction_comparison(args.output, args.threshold)
            else:
                result = run_receipt_check(args.output)
    except (OSError, ValueError, ImportError, KeyError, TypeError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps(result, indent=2))
    return 0
