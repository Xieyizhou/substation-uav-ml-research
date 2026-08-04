"""Lightweight acceptance receipts for resumable visual collection status."""

from __future__ import annotations

import json
from pathlib import Path

from src.ml.artifacts import object_sha256, write_json


RECEIPT_NAME = "identity/collection_validation.json"


def write_collection_validation_receipt(
    recording_directory,
    plan,
    validation,
):
    receipt = {
        "collection_validation_schema_version": 1,
        "accepted": True,
        "collection_plan_identity_sha256": plan[
            "collection_plan_identity_sha256"
        ],
        "scenario_id": validation["scenario_id"],
        "recording_id": validation["recording_id"],
        "recording_identity_sha256": validation[
            "recording_identity_sha256"
        ],
    }
    receipt["collection_validation_identity_sha256"] = object_sha256(receipt)
    path = Path(recording_directory) / RECEIPT_NAME
    write_json(path, receipt)
    return {**receipt, "path": str(path)}


def collection_validation_receipt_is_current(root, row, plan):
    root = Path(root)
    try:
        receipt = json.loads((root / RECEIPT_NAME).read_text(encoding="utf-8"))
        identity = json.loads(
            (root / "identity/recording_identity.json").read_text(
                encoding="utf-8"
            )
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    supplied = receipt.pop("collection_validation_identity_sha256", None)
    expected = {
        "collection_validation_schema_version": 1,
        "accepted": True,
        "collection_plan_identity_sha256": plan[
            "collection_plan_identity_sha256"
        ],
        "scenario_id": row["scenario_id"],
        "recording_id": row["recording_id"],
        "recording_identity_sha256": identity.get(
            "recording_identity_sha256"
        ),
    }
    return receipt == expected and supplied == object_sha256(receipt)
