"""Audit and register custom-map recordings as development datasets."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.sandbox.workbench_models import WorkbenchDataset
from src.vision.collection.pilot import REQUIRED_MANIFESTS, _read_json, _read_jsonl
from src.vision.collection.pilot_validation import (
    _load_and_validate_linkage,
    _validate_recording_identity,
)
from src.vision.training.yolo_dataset import link_image, yolo_label_text


VALID_SYNCHRONIZATION_STATUSES = {"exact", "nearest_within_tolerance"}


def _recording_data(recording_root):
    root = Path(recording_root)
    missing = [name for name in REQUIRED_MANIFESTS if not (root / name).is_file()]
    if missing:
        raise ValueError(f"custom-map recording is missing {missing[0]}")
    metadata = _read_json(root / "metadata.json")
    summary = _read_json(root / "summary.json")
    if (
        metadata.get("recording_type") != "labelled_visual_collection"
        or metadata.get("source_type") != "sandbox_custom_map"
        or metadata.get("dataset_role") != "development"
        or metadata.get("recording_state") != "complete"
        or summary.get("recording_state") != "complete"
    ):
        raise ValueError("custom-map recording metadata is not complete")
    lifecycle = metadata.get("flight_lifecycle") or {}
    if lifecycle.get("event_type") != "mission_completed" or not lifecycle.get(
        "landing_confirmed"
    ):
        raise ValueError("custom-map recording lacks a confirmed landing")
    frames, synchronization, annotations, _, _ = _load_and_validate_linkage(
        root, metadata
    )
    identity = _validate_recording_identity(root)
    invalid = [
        item for item in synchronization
        if item.synchronization_status not in VALID_SYNCHRONIZATION_STATUSES
    ]
    if invalid or len(annotations) != len(frames) or len(annotations) < 2:
        raise ValueError(
            "custom-map recording requires valid synchronized truth for every frame"
        )
    return metadata, frames, annotations, identity


def audit_sandbox_recording(recording_root):
    root = Path(recording_root)
    metadata, frames, annotations, identity = _recording_data(root)
    referenced_payloads = {item.payload_relative_path for item in frames}
    stored_payloads = {
        str(path.relative_to(root)) for path in (root / "frames").glob("*.png")
    }
    class_counts = Counter(
        item.class_name for row in annotations for item in row.objects
    )
    return {
        "valid": True,
        "recording_id": metadata["recording_id"],
        "recording_identity_sha256": identity,
        "map_revision_identity_sha256": metadata[
            "map_revision_identity_sha256"
        ],
        "frame_count": len(frames),
        "orphan_payload_count": len(stored_payloads - referenced_payloads),
        "labelled_frame_count": sum(
            row.annotation_status == "labelled" for row in annotations
        ),
        "no_target_frame_count": sum(
            row.annotation_status == "verified_no_target" for row in annotations
        ),
        "class_counts": {
            name: class_counts[name] for name in EQUIPMENT_CLASSES
        },
    }


def _write_dataset_yaml(root):
    names = "".join(
        f"  {index}: {name}\n" for index, name in enumerate(EQUIPMENT_CLASSES)
    )
    (root / "dataset.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\n"
        f"val: images/validation\nnames:\n{names}",
        encoding="utf-8",
    )


def _materialize_members(root, destination, frames, annotations):
    by_id = {item.frame_id: item for item in frames}
    members, split_counts = [], Counter()
    class_counts = {"train": Counter(), "validation": Counter()}
    no_target = Counter()
    modes = Counter()
    for index, annotation in enumerate(annotations):
        split = "validation" if index % 5 == 0 else "train"
        frame = by_id[annotation.frame_id]
        sample_id = f"sandbox-{index:06d}-{frame.payload_sha256[:12]}"
        image = destination / "images" / split / f"{sample_id}.png"
        label = destination / "labels" / split / f"{sample_id}.txt"
        source = root / frame.payload_relative_path
        if file_sha256(source) != frame.payload_sha256:
            raise ValueError(f"custom-map frame payload changed: {frame.frame_id}")
        modes[link_image(source, image)] += 1
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(yolo_label_text(annotation.to_record()), encoding="utf-8")
        split_counts[split] += 1
        class_counts[split].update(item.class_name for item in annotation.objects)
        no_target[split] += annotation.annotation_status == "verified_no_target"
        members.append({
            "sample_id": sample_id,
            "split": split,
            "frame_id": frame.frame_id,
            "payload_sha256": frame.payload_sha256,
            "label_sha256": file_sha256(label),
        })
    return members, split_counts, class_counts, no_target, modes


def register_sandbox_recording(recording_root, datasets_root, dataset_id):
    root, destination = Path(recording_root), Path(datasets_root) / dataset_id
    if not dataset_id or Path(dataset_id).name != dataset_id:
        raise ValueError("invalid workbench dataset identifier")
    if destination.exists():
        raise ValueError("workbench dataset identifier already exists")
    metadata, frames, annotations, source_identity = _recording_data(root)
    try:
        values = _materialize_members(root, destination, frames, annotations)
        members, counts, classes, no_target, modes = values
        if not counts["train"] or not counts["validation"]:
            raise ValueError("custom-map dataset requires train and validation frames")
        identity = object_sha256({
            "source_recording_identity_sha256": source_identity,
            "split_algorithm": "ordered_every_fifth_validation_v1",
            "members": members,
            "class_names": EQUIPMENT_CLASSES,
        })
        dataset = WorkbenchDataset(
            dataset_id=dataset_id,
            source_type="sandbox_custom_map",
            dataset_root=str(destination),
            dataset_identity_sha256=identity,
            source_identity_sha256=source_identity,
            split_counts=dict(counts),
            class_counts={
                split: {name: values[name] for name in EQUIPMENT_CLASSES}
                for split, values in classes.items()
            },
            no_target_counts=dict(no_target),
        )
        write_json(destination / "membership.json", members)
        write_json(destination / "dataset.json", dataset.to_record())
        write_json(destination / "registration_receipt.json", {
            "dataset_id": dataset_id,
            "dataset_identity_sha256": identity,
            "source_type": "sandbox_custom_map",
            "dataset_role": "development",
            "recording_id": metadata["recording_id"],
            "recording_identity_sha256": source_identity,
            "map_revision_identity_sha256": metadata[
                "map_revision_identity_sha256"
            ],
            "link_modes": dict(modes),
        })
        _write_dataset_yaml(destination)
        return dataset.to_record()
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        raise
