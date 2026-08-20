"""Materialize and evaluate a provenance-bound real-image stress view."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.evaluation.detection_metrics import threshold_metrics
from src.vision.evaluation.yolo_evaluation import collect_predictions


SOURCE_NAME = "Power-equipment-image-dataset/substation-object-detection"
SOURCE_URL = "https://huggingface.co/datasets/sxiong/Power-equipment-image-dataset"
SOURCE_LICENSE = "MIT"
CLASS_MAPPING = {
    "capacitor": "capacitor_bank",
    "whole capacitor": "capacitor_bank",
    "switch": "switchgear",
    "switch+insulator": "switchgear",
    "breaker": "switchgear",
    "gis": "switchgear",
}


def _image_size(path):
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("reality stress materialization requires Pillow") from error
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def _box(region, width, height):
    shape = region.get("shape_attributes", {})
    xs, ys = shape.get("all_points_x", []), shape.get("all_points_y", [])
    if not xs or len(xs) != len(ys):
        raise ValueError("stress annotation contains an invalid polygon")
    left, right = max(0, min(xs)), min(width, max(xs))
    top, bottom = max(0, min(ys)), min(height, max(ys))
    if left >= right or top >= bottom:
        raise ValueError("stress annotation contains an empty bounding box")
    return (
        (left + right) / (2 * width),
        (top + bottom) / (2 * height),
        (right - left) / width,
        (bottom - top) / height,
    )


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        import shutil

        shutil.copy2(source, destination)
        return "copy"


def materialize_reality_stress(source_root, annotation_path, output_root):
    source_root, annotation_path = Path(source_root), Path(annotation_path)
    output_root = Path(output_root)
    if output_root.exists():
        raise ValueError("reality stress output already exists")
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    entries = sorted(annotations.values(), key=lambda row: row["filename"])
    if not 300 <= len(entries) <= 1000:
        raise ValueError("reality stress view must contain 300 to 1000 images")
    image_out, label_out = output_root / "images/stress", output_root / "labels/stress"
    image_out.mkdir(parents=True)
    label_out.mkdir(parents=True)
    source_types, mapped_counts, transfer_modes, membership = Counter(), Counter(), Counter(), []
    for entry in entries:
        filename = Path(entry["filename"]).name
        source = source_root / filename
        if not source.is_file():
            raise ValueError(f"stress image is missing: {filename}")
        width, height = _image_size(source)
        labels = []
        for region in entry.get("regions", []):
            source_type = str(region.get("region_attributes", {}).get("type", "")).strip().lower()
            source_types[source_type] += 1
            target = CLASS_MAPPING.get(source_type)
            if target is None:
                continue
            class_id = EQUIPMENT_CLASSES.index(target)
            values = _box(region, width, height)
            labels.append(f"{class_id} " + " ".join(f"{value:.9f}" for value in values))
            mapped_counts[target] += 1
        destination = image_out / filename
        transfer_modes[_link_or_copy(source, destination)] += 1
        (label_out / f"{Path(filename).stem}.txt").write_text(
            "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
        )
        membership.append({
            "sample_id": Path(filename).stem,
            "image_sha256": file_sha256(source),
            "mapped_object_count": len(labels),
        })
    dataset_yaml = (
        f"path: {output_root.resolve()}\n"
        "train: images/stress\nval: images/stress\ntest: images/stress\n"
        "names: [transformer, switchgear, capacitor_bank, reactor]\n"
    )
    (output_root / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    manifest = {
        "reality_stress_schema_version": 1,
        "source_name": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "source_license": SOURCE_LICENSE,
        "source_annotation_sha256": file_sha256(annotation_path),
        "class_mapping": CLASS_MAPPING,
        "class_order": list(EQUIPMENT_CLASSES),
        "frame_count": len(membership),
        "mapped_object_counts": dict(sorted(mapped_counts.items())),
        "unsupported_source_object_counts": {
            name: count for name, count in sorted(source_types.items()) if name not in CLASS_MAPPING
        },
        "unsupported_target_classes": [
            name for name in EQUIPMENT_CLASSES if mapped_counts[name] == 0
        ],
        "transfer_modes": dict(sorted(transfer_modes.items())),
        "ordered_membership_sha256": object_sha256(membership),
    }
    manifest["reality_stress_identity_sha256"] = object_sha256(manifest)
    write_json(output_root / "identity/reality_stress_identity.json", manifest)
    return manifest


def evaluate_reality_stress(model_path, dataset_root, output_path, *, device="cpu", imgsz=640, threshold=0.37):
    dataset_root, output_path = Path(dataset_root), Path(output_path)
    identity = json.loads((dataset_root / "identity/reality_stress_identity.json").read_text())
    frames = collect_predictions(
        model_path, dataset_root, "stress", device=device, imgsz=imgsz,
        confidence=0.05, batch=1,
    )
    metrics = threshold_metrics(frames, float(threshold))
    supported = {
        name: row for name, row in metrics["per_class"].items()
        if identity["mapped_object_counts"].get(name, 0) > 0
    }
    result = {
        "reality_stress_evaluation_schema_version": 1,
        "reality_stress_identity_sha256": identity["reality_stress_identity_sha256"],
        "model_sha256": file_sha256(Path(model_path)),
        "input_size": int(imgsz),
        "threshold": float(threshold),
        "frame_count": len(frames),
        "supported_class_metrics": supported,
        "unsupported_target_classes": identity["unsupported_target_classes"],
        "no_target_false_positive_rate": metrics["no_target_false_positive_rate"],
        "small_object_recall": metrics["small_object_recall"],
        "coverage_warning": "Metrics are reported only for target classes represented in the source dataset.",
    }
    write_json(output_path, result)
    return result
