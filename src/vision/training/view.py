"""Materialize deterministic, split-safe YOLO training views."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, write_json
from src.vision.contracts.identity import DatasetIdentity, class_order_identity
from src.vision.contracts.training_identity import TrainingViewIdentity
from src.vision.training.yolo_dataset import materialize_yolo_partition


SAMPLING_ALGORITHM = "recording_proportional_even_v1"
TRAIN_CLASS_CAP = 5_000
TRAIN_NEGATIVE_CAP = 8_000
VALIDATION_CLASS_CAP = 2_000
VALIDATION_NEGATIVE_CAP = 4_000


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def proportional_quotas(counts, target):
    """Allocate an exact target proportionally with stable largest remainders."""
    counts = {key: int(value) for key, value in counts.items() if value > 0}
    target = min(int(target), sum(counts.values()))
    if target <= 0:
        return {key: 0 for key in counts}
    keys = sorted(counts)
    quotas = {key: 0 for key in keys}
    if target >= len(keys):
        quotas = {key: 1 for key in keys}
        target -= len(keys)
        remaining = {key: counts[key] - 1 for key in keys}
    else:
        chosen = sorted(keys, key=lambda key: (-counts[key], str(key)))[:target]
        return {key: int(key in chosen) for key in keys}
    available = sum(remaining.values())
    if target and available:
        raw = {key: target * remaining[key] / available for key in keys}
        for key in keys:
            quotas[key] += min(remaining[key], math.floor(raw[key]))
        left = target - sum(quotas.values()) + len(keys)
        order = sorted(
            keys,
            key=lambda key: (-(raw[key] - math.floor(raw[key])), str(key)),
        )
        for key in order:
            if left <= 0:
                break
            if quotas[key] < counts[key]:
                quotas[key] += 1
                left -= 1
    return quotas


def evenly_select(rows, count):
    def sequence(row):
        if "sequence_number" in row:
            return int(row["sequence_number"])
        return int(row["annotation"]["sequence_number"])

    rows = sorted(
        rows,
        key=lambda row: (
            sequence(row),
            row["sample_id"],
        ),
    )
    if count >= len(rows):
        return rows
    indices = [
        min(len(rows) - 1, ((2 * i + 1) * len(rows)) // (2 * count))
        for i in range(count)
    ]
    return [rows[index] for index in indices]


def _select_grouped(rows, cap, grouping):
    groups = defaultdict(list)
    for row in rows:
        groups[grouping(row)].append(row)
    quotas = proportional_quotas(
        {key: len(values) for key, values in groups.items()},
        min(cap, len(rows)),
    )
    return [
        row
        for key in sorted(groups, key=str)
        for row in evenly_select(groups[key], quotas[key])
    ]


def select_partition(rows, *, class_cap, negative_cap):
    selected = {}
    for class_name in EQUIPMENT_CLASSES:
        candidates = [
            row
            for row in rows
            if class_name
            in {obj["class_name"] for obj in row["annotation"]["objects"]}
        ]
        for row in _select_grouped(
            candidates,
            class_cap,
            lambda item: item["recording_id"],
        ):
            selected[row["sample_id"]] = row
    negatives = [
        row
        for row in rows
        if row["annotation"]["annotation_status"] == "verified_no_target"
    ]
    for row in _select_grouped(
        negatives,
        negative_cap,
        lambda item: (
            item["recording_id"],
            item["annotation"]["mission_phase"],
        ),
    ):
        selected[row["sample_id"]] = row
    return sorted(
        selected.values(),
        key=lambda row: (
            row["recording_id"],
            int(row["annotation"]["sequence_number"]),
            row["sample_id"],
        ),
    )


def materialize_training_view(collection_root, output_root, *, seed=7):
    if seed != 7:
        raise ValueError("visual training view v1 requires sampling seed 7")
    collection_root, output_root = Path(collection_root), Path(output_root)
    identity_root = collection_root / "identity"
    development = DatasetIdentity.from_record(
        json.loads((identity_root / "development_dataset_identity.json").read_text())
    )
    membership = {
        row["sample_id"]: row
        for row in _read_jsonl(identity_root / "development_membership.jsonl")
    }
    annotations = _read_jsonl(identity_root / "development_annotations.jsonl")
    rows = []
    for item in annotations:
        member = membership[item["sample_id"]]
        rows.append({**member, **item})
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    if len(train_rows) + len(validation_rows) != len(rows):
        raise ValueError("development identity contains a non-development split")
    train = select_partition(
        train_rows, class_cap=TRAIN_CLASS_CAP, negative_cap=TRAIN_NEGATIVE_CAP
    )
    validation = select_partition(
        validation_rows,
        class_cap=VALIDATION_CLASS_CAP,
        negative_cap=VALIDATION_NEGATIVE_CAP,
    )
    train_result = materialize_yolo_partition(
        "train", train, collection_root, output_root
    )
    validation_result = materialize_yolo_partition(
        "validation", validation, collection_root, output_root
    )
    full_validation_result = materialize_yolo_partition(
        "full_validation", validation_rows, collection_root, output_root
    )
    smoke_train_result = materialize_yolo_partition(
        "smoke_train",
        evenly_select(train, min(256, len(train))),
        collection_root,
        output_root,
    )
    smoke_validation_result = materialize_yolo_partition(
        "smoke_validation",
        evenly_select(validation, min(64, len(validation))),
        collection_root,
        output_root,
    )
    labels_path = output_root / "identity/labels_manifest.json"
    write_json(
        labels_path,
        {
            "train": train_result["labels_manifest"],
            "validation": validation_result["labels_manifest"],
            "full_validation": full_validation_result["labels_manifest"],
        },
    )
    identity = TrainingViewIdentity(
        source_development_dataset_identity=development.dataset_identity_sha256,
        sampling_algorithm=SAMPLING_ALGORITHM,
        sampling_seed=seed,
        train_membership_sha256=train_result["membership_sha256"],
        validation_membership_sha256=validation_result["membership_sha256"],
        full_validation_membership_sha256=full_validation_result[
            "membership_sha256"
        ],
        labels_manifest_sha256=file_sha256(labels_path),
        class_order_identity=class_order_identity(),
        train_frame_count=train_result["frame_count"],
        validation_frame_count=validation_result["frame_count"],
        full_validation_frame_count=full_validation_result["frame_count"],
        train_class_counts=train_result["class_counts"],
        validation_class_counts=validation_result["class_counts"],
        train_no_target_count=train_result["no_target_count"],
        validation_no_target_count=validation_result["no_target_count"],
    )
    identity_path = output_root / "identity/training_view_identity.json"
    write_json(identity_path, identity.to_record())
    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {output_root.resolve()}\n"
        "train: images/train\nval: images/validation\ntest: images/full_validation\n"
        "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(EQUIPMENT_CLASSES)),
        encoding="utf-8",
    )
    smoke_yaml_path = output_root / "dataset-smoke.yaml"
    smoke_yaml_path.write_text(
        f"path: {output_root.resolve()}\n"
        "train: images/smoke_train\nval: images/smoke_validation\n"
        "names:\n"
        + "".join(
            f"  {index}: {name}\n"
            for index, name in enumerate(EQUIPMENT_CLASSES)
        ),
        encoding="utf-8",
    )
    return {
        "identity": identity.to_record(),
        "identity_path": str(identity_path),
        "dataset_yaml": str(yaml_path),
        "smoke_dataset_yaml": str(smoke_yaml_path),
        "link_modes": {
            "train": train_result["link_modes"],
            "validation": validation_result["link_modes"],
            "full_validation": full_validation_result["link_modes"],
            "smoke_train": smoke_train_result["link_modes"],
            "smoke_validation": smoke_validation_result["link_modes"],
        },
    }
