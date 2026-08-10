"""Load split-safe source rows for visual training views."""

from __future__ import annotations

import json
from pathlib import Path

from src.vision.contracts.identity import DatasetIdentity


V2_PROTOCOL = "visual-multiscenario-png-v2"


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _load_partition(identity_root, name):
    identity = DatasetIdentity.from_record(
        json.loads((identity_root / f"{name}_dataset_identity.json").read_text())
    )
    membership = {
        row["sample_id"]: row
        for row in _read_jsonl(identity_root / f"{name}_membership.jsonl")
    }
    annotations = _read_jsonl(identity_root / f"{name}_annotations.jsonl")
    if set(membership) != {row["sample_id"] for row in annotations}:
        raise ValueError(f"{name} membership and annotation samples differ")
    return identity, [{**membership[row["sample_id"]], **row} for row in annotations]


def load_training_sources(collection_root):
    identity_root = Path(collection_root) / "identity"
    development, rows = _load_partition(identity_root, "development")
    if development.dataset_version == V2_PROTOCOL:
        validation, validation_rows = _load_partition(identity_root, "validation")
        if validation.dataset_version != V2_PROTOCOL:
            raise ValueError("v2 development and validation protocols differ")
        if any(row["split"] != "development" for row in rows):
            raise ValueError("v2 development identity contains another split")
        if any(row["split"] != "validation" for row in validation_rows):
            raise ValueError("v2 validation identity contains another split")
        return development, validation, rows, validation_rows
    train = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    if len(train) + len(validation_rows) != len(rows):
        raise ValueError("development identity contains a non-development split")
    return development, None, train, validation_rows
