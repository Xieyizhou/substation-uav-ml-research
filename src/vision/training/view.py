"""Materialize deterministic, split-safe YOLO training views."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path

from src.ml import EQUIPMENT_CLASSES
from src.vision.training.view_source import load_training_sources


SAMPLING_ALGORITHM = "recording_proportional_even_v1"
TRAIN_CLASS_CAP = 5_000
TRAIN_NEGATIVE_CAP = 8_000
VALIDATION_CLASS_CAP = 2_000
VALIDATION_NEGATIVE_CAP = 4_000


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
        annotation = row.get("annotation")
        if annotation and "sequence_number" in annotation:
            return int(annotation["sequence_number"])
        source_frame_id = row.get("source_frame_id")
        if source_frame_id:
            suffix = str(source_frame_id).rsplit("-", 1)[-1]
            if suffix.isdigit():
                return int(suffix)
        return int(hashlib.sha256(row["sample_id"].encode()).hexdigest()[:16], 16)

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
    development, source_validation, train_rows, validation_rows = (
        load_training_sources(collection_root)
    )
    selection_summary = None
    if source_validation is None:
        algorithm = SAMPLING_ALGORITHM
        train = select_partition(
            train_rows, class_cap=TRAIN_CLASS_CAP, negative_cap=TRAIN_NEGATIVE_CAP
        )
        validation = select_partition(
            validation_rows,
            class_cap=VALIDATION_CLASS_CAP,
            negative_cap=VALIDATION_NEGATIVE_CAP,
        )
    else:
        from src.vision.training.v2_sampling import (
            ALGORITHM,
            TRAIN_SIZE_TARGETS,
            VALIDATION_SIZE_TARGETS,
            select_v2_partition,
        )

        algorithm = ALGORITHM
        train, train_summary = select_v2_partition(
            train_rows, size_targets=TRAIN_SIZE_TARGETS, negative_cap=8_000
        )
        validation, validation_summary = select_v2_partition(
            validation_rows,
            size_targets=VALIDATION_SIZE_TARGETS,
            negative_cap=4_000,
        )
        selection_summary = {"train": train_summary, "validation": validation_summary}
    from src.vision.training.view_output import materialize_view_output

    return materialize_view_output(
        collection_root, output_root, development, source_validation,
        train, validation, validation_rows, algorithm, seed, selection_summary,
    )
