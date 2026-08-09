"""Materialize held-out YOLO labels only after a model package is frozen."""

from __future__ import annotations

import json
from pathlib import Path

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import object_sha256, write_json
from src.vision.contracts.identity import DatasetIdentity
from src.vision.training.yolo_dataset import materialize_yolo_partition
from src.vision.evaluation.yolo_package import validate_yolo_package


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _heldout_dataset_yaml(output_root):
    return (
        f"path: {Path(output_root).resolve()}\n"
        "train: disabled/heldout_not_for_training\n"
        "val: disabled/heldout_not_for_validation\n"
        "test: images/heldout_test\nnames:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(EQUIPMENT_CLASSES))
    )


def load_heldout_source(collection_root):
    collection_root = Path(collection_root)
    identity_root = collection_root / "identity"
    heldout = DatasetIdentity.from_record(
        json.loads((identity_root / "held_out_test_dataset_identity.json").read_text())
    )
    membership = {
        row["sample_id"]: row
        for row in _read_jsonl(identity_root / "held_out_test_membership.jsonl")
    }
    annotations = _read_jsonl(identity_root / "held_out_test_annotations.jsonl")
    rows = [{**membership[item["sample_id"]], **item} for item in annotations]
    expected_split = (
        "blind"
        if heldout.dataset_version == "visual-multiscenario-png-v2"
        else "test"
    )
    if any(row["split"] != expected_split for row in rows):
        raise ValueError(
            f"held-out identity contains a non-{expected_split} split"
        )
    return heldout, rows


def materialize_heldout_view(collection_root, package_root, output_root):
    collection_root, output_root = Path(collection_root), Path(output_root)
    package = validate_yolo_package(package_root)
    existing_receipt = output_root / "identity/heldout_access_receipt.json"
    if existing_receipt.is_file():
        existing = json.loads(existing_receipt.read_text())
        if (
            existing.get("model_package_identity_sha256")
            != package["package_identity_sha256"]
        ):
            raise ValueError(
                "held-out view was already unlocked for a different package"
            )
    heldout, rows = load_heldout_source(collection_root)
    result = materialize_yolo_partition(
        "heldout_test", rows, collection_root, output_root
    )
    write_json(
        output_root / "identity/held_out_test_dataset_identity.json",
        heldout.to_record(),
    )
    yaml = output_root / "dataset.yaml"
    yaml.write_text(_heldout_dataset_yaml(output_root), encoding="utf-8")
    receipt = {
        "heldout_access_schema_version": 1,
        "heldout_dataset_identity_sha256": heldout.dataset_identity_sha256,
        "model_package_identity_sha256": package["package_identity_sha256"],
        "allowed_model_sha256": sorted(
            export["sha256"] for export in package["exports"].values()
        ),
        "canonical_input_size": 640,
        "canonical_model_sha256": package["exports"]["640"]["sha256"],
        "frozen_confidence_threshold": package[
            "frozen_confidence_threshold"
        ],
        "membership_sha256": result["membership_sha256"],
        "frame_count": result["frame_count"],
    }
    receipt["heldout_access_identity_sha256"] = object_sha256(receipt)
    write_json(output_root / "identity/heldout_access_receipt.json", receipt)
    return {
        "receipt": receipt,
        "dataset_yaml": str(yaml),
        "link_modes": result["link_modes"],
    }
