"""Unlock a real-domain blind view only after the v3 package is frozen."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.contracts.real_domain import RealDomainDatasetIdentity
from src.vision.evaluation.yolo_package import validate_yolo_package


def _link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def materialize_real_blind_view(real_dataset_root, package_root, output_root):
    real_dataset_root, package_root = Path(real_dataset_root), Path(package_root)
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("real blind output must be empty and used only once")
    package = validate_yolo_package(package_root)
    gate_ref = package.get("real_domain_v3_gate")
    if not isinstance(gate_ref, dict):
        raise ValueError("real blind requires a package-bound v3 validation gate")
    gate_path = package_root / gate_ref.get("path", "")
    if file_sha256(gate_path) != gate_ref.get("sha256"):
        raise ValueError("package-bound v3 validation gate changed")
    gate = json.loads(gate_path.read_text())
    if gate.get("passed") is not True:
        raise ValueError("real blind remains sealed because the v3 gate failed")
    identity = RealDomainDatasetIdentity.from_record(json.loads(
        (real_dataset_root / "identity/real_domain_dataset_identity.json").read_text()
    ))
    rows = []
    with (real_dataset_root / "identity/samples.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            if row["partition"] == "blind":
                rows.append(row)
    for row in rows:
        _link(real_dataset_root / row["image_relative_path"],
              output_root / row["image_relative_path"])
        _link(real_dataset_root / row["label_relative_path"],
              output_root / row["label_relative_path"])
    membership = output_root / "identity/blind_membership.jsonl"
    membership.parent.mkdir(parents=True, exist_ok=True)
    membership.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ), encoding="utf-8")
    names = "".join(f"  {index}: {name}\n" for index, name in enumerate(EQUIPMENT_CLASSES))
    (output_root / "dataset.yaml").write_text(
        f"path: {output_root.resolve()}\ntest: images/blind\nnames:\n" + names,
        encoding="utf-8",
    )
    receipt = {
        "real_blind_access_receipt_schema_version": 1,
        "real_domain_dataset_identity_sha256": (
            identity.real_domain_dataset_identity_sha256
        ),
        "package_identity_sha256": package["package_identity_sha256"],
        "gate_identity_sha256": gate["gate_identity_sha256"],
        "membership_sha256": file_sha256(membership),
        "canonical_model_relative_path": package["exports"]["640"]["path"],
        "canonical_model_sha256": package["exports"]["640"]["sha256"],
        "frozen_confidence_threshold": package["frozen_confidence_threshold"],
        "frame_count": len(rows),
        "threshold_search_allowed": False,
    }
    receipt["receipt_identity_sha256"] = object_sha256(receipt)
    write_json(output_root / "identity/real_blind_access_receipt.json", receipt)
    return receipt
