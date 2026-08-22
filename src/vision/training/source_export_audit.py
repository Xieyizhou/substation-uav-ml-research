"""Inspect a manually exported YOLO archive without extracting it."""

from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath

import yaml

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.training.source_feasibility import load_source_registry
from src.vision.training.source_export_labels import inspect_label_entries


EXPORT_AUDIT_SCHEMA_VERSION = 2
MAX_ARCHIVE_ENTRIES = 100_000
MAX_UNCOMPRESSED_BYTES = 20 * 1024 * 1024 * 1024
MAX_DATA_YAML_BYTES = 1024 * 1024
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
SPLIT_ALIASES = {"train": "train", "valid": "validation", "val": "validation", "test": "test"}


def audit_source_export(registry_path, source_id, archive_path, output_root):
    """Create a structural quarantine receipt for one Roboflow-style YOLO ZIP."""
    _, sources = load_source_registry(registry_path)
    if source_id not in sources:
        raise ValueError(f"unknown real-domain source: {source_id}")
    source = sources[source_id]
    if source["ingestion_status"] != "quarantine":
        raise ValueError("export audit is limited to quarantine sources")
    archive = Path(archive_path)
    if archive.suffix.lower() != ".zip" or not archive.is_file():
        raise ValueError("source export must be an existing ZIP archive")
    destination = Path(output_root) / source_id / "export-audit.json"
    if destination.exists():
        raise FileExistsError(f"export audit already exists: {destination}")
    with zipfile.ZipFile(archive) as bundle:
        entries = _safe_entries(bundle)
        dataset_yaml, duplicate_yaml_count, yaml_warnings = _load_dataset_yaml(
            bundle, entries
        )
        export_metadata = _export_metadata(bundle, entries)
        class_names = _class_names(dataset_yaml)
        summary = _summarize_entries(bundle, entries, class_names)
    candidate_labels = _candidate_labels(source)
    missing_candidates = sorted(candidate_labels.difference(class_names))
    errors = list(summary.pop("errors"))
    if not class_names:
        errors.append("data.yaml does not declare class names")
    if missing_candidates:
        summary["candidate_label_warnings"] = [
            "registry candidate labels are absent from this export"
        ]
    if yaml_warnings:
        summary["data_yaml_warnings"] = yaml_warnings
    receipt = {
        "source_export_audit_schema_version": EXPORT_AUDIT_SCHEMA_VERSION,
        "source_id": source_id,
        "archive_name": archive.name,
        "archive_sha256": file_sha256(archive),
        "class_names": class_names,
        "registry_candidate_labels": sorted(candidate_labels),
        "missing_registry_candidate_labels": missing_candidates,
        "duplicate_entry_count": duplicate_yaml_count,
        "export_metadata": export_metadata,
        "preprocessing": export_metadata.get("preprocessing", []),
        "augmentation_declared": export_metadata.get("augmentation_declared"),
        **summary,
        "errors": errors,
        "structurally_valid": not errors,
        "training_eligible": False,
        "next_action": "manual_domain_provenance_and_semantic_review",
    }
    receipt["export_audit_identity_sha256"] = object_sha256(receipt)
    write_json(destination, receipt)
    return receipt


def _export_metadata(bundle, entries):
    names = {PurePosixPath(item.filename).name: item for item in entries}
    roboflow = names.get("README.roboflow.txt")
    dataset = names.get("README.dataset.txt")
    texts = {}
    for key, entry in (("roboflow", roboflow), ("dataset", dataset)):
        if entry is not None:
            texts[key] = bundle.read(entry).decode("utf-8", errors="replace")
    joined = "\n".join(texts.values())
    first = next((line.strip() for line in joined.splitlines() if line.strip()), "")
    image_match = __import__("re").search(r"includes\s+(\d+)\s+images", joined, __import__("re").I)
    url_match = __import__("re").search(r"https://universe\.roboflow\.com/\S+", joined)
    preprocessing = []
    in_preprocessing = False
    for line in texts.get("roboflow", "").splitlines():
        if "pre-processing was applied" in line:
            in_preprocessing = True
        elif in_preprocessing and line.startswith("*"):
            preprocessing.append(line[1:].strip())
        elif in_preprocessing and line.strip() and not line.startswith("*"):
            in_preprocessing = False
    augmentation = None
    if "No image augmentation techniques were applied" in joined:
        augmentation = False
    elif "augmentation" in joined.lower():
        augmentation = True
    return {
        "readme_present": bool(texts),
        "dataset_title_and_version": first,
        "declared_image_count": int(image_match.group(1)) if image_match else None,
        "dataset_url": url_match.group(0) if url_match else None,
        "preprocessing": preprocessing,
        "augmentation_declared": augmentation,
    }


def _safe_entries(bundle):
    entries = bundle.infolist()
    if len(entries) > MAX_ARCHIVE_ENTRIES:
        raise ValueError("source export contains too many entries")
    total = 0
    for entry in entries:
        path = PurePosixPath(entry.filename)
        if entry.filename.startswith(("/", "\\")) or ".." in path.parts:
            raise ValueError("source export contains an unsafe path")
        if "\\" in entry.filename:
            raise ValueError("source export contains a non-portable path")
        if (entry.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError("source export contains a symbolic link")
        total += entry.file_size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("source export exceeds the uncompressed size limit")
    return entries


def _load_dataset_yaml(bundle, entries):
    candidates = [item for item in entries if PurePosixPath(item.filename).name == "data.yaml"]
    if not candidates:
        raise ValueError("source export must contain a data.yaml")
    first = candidates[0]
    first_payload = bundle.read(first)
    if len(first_payload) > MAX_DATA_YAML_BYTES:
        raise ValueError("data.yaml exceeds the size limit")
    duplicate_count = 0
    warnings = []
    for candidate in candidates[1:]:
        payload = bundle.read(candidate)
        if len(payload) > MAX_DATA_YAML_BYTES:
            raise ValueError("data.yaml exceeds the size limit")
        if payload != first_payload:
            raise ValueError("data.yaml duplicates are present with conflicting content")
        duplicate_count += 1
    if duplicate_count:
        warnings.append(
            f"duplicate data.yaml entries are byte-identical and normalized to {first.filename}"
        )
    record = yaml.safe_load(first_payload.decode("utf-8"))
    if not isinstance(record, dict):
        raise ValueError("data.yaml must contain a mapping")
    return record, duplicate_count, warnings


def _class_names(record):
    names = record.get("names", [])
    if isinstance(names, dict):
        ordered = [names[key] for key in sorted(names, key=lambda value: int(value))]
    elif isinstance(names, list):
        ordered = names
    else:
        return []
    return [str(value).strip() for value in ordered if str(value).strip()]


def _candidate_labels(source):
    return {
        str(label).strip()
        for mapping in source.get("class_candidates", {}).values()
        for label in mapping.get("source_labels", [])
        if str(label).strip()
    }


def _summarize_entries(bundle, entries, class_names):
    images, labels = {}, {}
    duplicate_keys = []
    uncompressed = sum(item.file_size for item in entries)
    for entry in entries:
        if entry.is_dir():
            continue
        path = PurePosixPath(entry.filename)
        split = _split_for(path)
        if split is None:
            continue
        key = (split, path.stem)
        if path.suffix.lower() in IMAGE_SUFFIXES and "images" in path.parts:
            if key in images:
                duplicate_keys.append(f"image:{split}/{path.stem}")
            images[key] = entry
        elif path.suffix.lower() == ".txt" and "labels" in path.parts:
            if key in labels:
                duplicate_keys.append(f"label:{split}/{path.stem}")
            labels[key] = entry
    missing_labels = sorted(f"{split}/{stem}" for split, stem in images.keys() - labels.keys())
    orphan_labels = sorted(f"{split}/{stem}" for split, stem in labels.keys() - images.keys())
    split_counts = {
        split: sum(1 for item_split, _ in images if item_split == split)
        for split in ("train", "validation", "test")
    }
    invalid_labels = inspect_label_entries(bundle, labels.values(), class_names)
    errors = []
    if not images:
        errors.append("export contains no YOLO images")
    if missing_labels:
        errors.append("one or more images have no label file")
    if orphan_labels:
        errors.append("one or more label files have no image")
    if duplicate_keys:
        errors.append("duplicate sample stems exist within a split")
    if not split_counts["train"] or not split_counts["validation"]:
        errors.append("export must contain train and validation images")
    if invalid_labels["invalid_label_file_count"]:
        errors.append("one or more YOLO label rows are invalid")
    return {
        "archive_entry_count": len(entries),
        "uncompressed_bytes": uncompressed,
        "image_count": len(images),
        "label_count": len(labels),
        "split_image_counts": split_counts,
        "missing_label_count": len(missing_labels),
        "orphan_label_count": len(orphan_labels),
        "missing_label_examples": missing_labels[:20],
        "orphan_label_examples": orphan_labels[:20],
        "duplicate_sample_key_count": len(duplicate_keys),
        "duplicate_sample_key_examples": sorted(duplicate_keys)[:20],
        "invalid_label_file_count": invalid_labels["invalid_label_file_count"],
        "invalid_label_examples": invalid_labels["invalid_label_examples"],
        "class_image_count": invalid_labels["class_image_count"],
        "class_annotation_count": invalid_labels["class_annotation_count"],
        "detection_annotation_count": invalid_labels["detection_annotation_count"],
        "segmentation_annotation_count": invalid_labels["segmentation_annotation_count"],
        "annotation_conversion": invalid_labels["annotation_conversion"],
        "polygon_provenance": invalid_labels["polygon_provenance"],
        "errors": errors,
    }


def _split_for(path):
    for part in path.parts:
        if part.lower() in SPLIT_ALIASES:
            return SPLIT_ALIASES[part.lower()]
    return None
