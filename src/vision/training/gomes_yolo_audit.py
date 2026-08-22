"""Audit official Gomes Figshare YOLO capture-group archives."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import md5, sha256
import json
from pathlib import Path, PurePosixPath
import zipfile

from src.ml.artifacts import object_sha256, write_json
from src.vision.training.source_export_audit import IMAGE_SUFFIXES, _safe_entries
from src.vision.training.source_feasibility import load_source_registry


SCHEMA_VERSION = 1


def audit_gomes_yolo_component(registry_path, source_id, archive_path,
                               classes_path, output_root):
    _, sources = load_source_registry(registry_path)
    source = sources.get(source_id)
    if source is None or source.get("annotation_format") != "yolo-detection-capture-groups":
        raise ValueError("source is not a Gomes YOLO capture-group source")
    archive, classes_path = Path(archive_path), Path(classes_path)
    component = next((item for item in source.get("artifact_components", [])
                      if item["filename"] == archive.name), None)
    if component is None:
        raise ValueError("archive is not a pinned Gomes artifact component")
    _verify_file(archive, component)
    metadata = next(item for item in source["artifact_components"]
                    if item["filename"] == "classes.txt")
    _verify_file(classes_path, metadata)
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    with zipfile.ZipFile(archive) as bundle:
        entries = _safe_entries(bundle)
        images, labels = _pair_entries(entries)
        counts = _annotation_counts(bundle, labels, classes)
        image_hashes = {
            stem: sha256(bundle.read(entry)).hexdigest() for stem, entry in images.items()
        }
    missing = sorted(images.keys() - labels.keys())
    orphan = sorted(labels.keys() - images.keys())
    duplicate_count = len(image_hashes) - len(set(image_hashes.values()))
    destination = Path(output_root) / source_id / component["capture_group"] / "audit.json"
    if destination.exists():
        raise FileExistsError(f"Gomes component audit already exists: {destination}")
    receipt = {
        "gomes_yolo_component_audit_schema_version": SCHEMA_VERSION,
        "source_id": source_id, "source_family": source["source_family"],
        "site_group_id": "gomes-brazil-substation-01",
        "capture_group": component["capture_group"], "archive_name": archive.name,
        "archive_md5": component["md5"], "image_count": len(images),
        "label_count": len(labels), "missing_label_count": len(missing),
        "orphan_label_count": len(orphan), "exact_duplicate_count": duplicate_count,
        "class_image_count": counts["images"], "class_annotation_count": counts["annotations"],
        "transformer_image_count": counts["images"].get("Power transformer", 0),
        "transformer_annotation_count": counts["annotations"].get("Power transformer", 0),
        "structurally_valid": not missing and not orphan,
        "eligible_partition_after_family_review": "development",
        "training_eligible": False,
        "blocking_issues": list(source["blocking_issues"]),
    }
    receipt["gomes_component_identity_sha256"] = object_sha256(receipt)
    write_json(destination, receipt)
    return receipt


def _verify_file(path, metadata):
    if not path.is_file() or path.stat().st_size != int(metadata["bytes"]):
        raise ValueError(f"artifact size mismatch: {path.name}")
    digest = md5(path.read_bytes(), usedforsecurity=False).hexdigest()
    if digest != metadata["md5"]:
        raise ValueError(f"artifact MD5 mismatch: {path.name}")


def _pair_entries(entries):
    images, labels = {}, {}
    for entry in entries:
        if entry.is_dir():
            continue
        path = PurePosixPath(entry.filename)
        if path.suffix.lower() in IMAGE_SUFFIXES:
            images[path.stem] = entry
        elif path.suffix.lower() == ".txt" and "labels" in path.parts:
            labels[path.stem] = entry
    return images, labels


def _annotation_counts(bundle, labels, classes):
    annotations, class_images = Counter(), Counter()
    for entry in labels.values():
        seen = set()
        for line in bundle.read(entry).decode("utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"invalid YOLO row in {entry.filename}")
            class_value = float(fields[0])
            class_id = int(class_value)
            values = [float(value) for value in fields[1:]]
            if class_value != class_id or not 0 <= class_id < len(classes):
                raise ValueError(f"invalid class id in {entry.filename}")
            if not all(0 <= value <= 1 for value in values) or values[2] <= 0 or values[3] <= 0:
                raise ValueError(f"invalid normalized box in {entry.filename}")
            name = classes[class_id]
            annotations[name] += 1
            seen.add(name)
        class_images.update(seen)
    return {"images": dict(class_images), "annotations": dict(annotations)}
