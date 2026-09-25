#!/usr/bin/env python3
"""Read-only integrity check for a frozen local experiment baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256
from src.vision.evaluation.yolo_package import validate_yolo_package


def verify(path: Path) -> dict:
    record = json.loads(path.read_text())
    supplied = record.pop("baseline_identity_sha256")
    if object_sha256(record) != supplied:
        raise ValueError("baseline contract identity mismatch")
    for ref in record["pinned_files"]:
        source = ROOT / ref["path"]
        if file_sha256(source) != ref["sha256"]:
            raise ValueError(f"frozen file changed: {ref['path']}")
    package = validate_yolo_package(ROOT / record["model"]["package_root"])
    if package["package_identity_sha256"] != record["model"]["package_identity_sha256"]:
        raise ValueError("baseline model package changed")
    groups = {}
    for partition, spec in record["data"]["memberships"].items():
        rows = [json.loads(line) for line in (ROOT / spec["path"]).read_text().splitlines()]
        if len(rows) != spec["frame_count"]:
            raise ValueError(f"membership count mismatch: {partition}")
        groups[partition] = {
            key: {row[key] for row in rows}
            for key in ("sample_id", "recording_id", "payload_sha256")
        }
    overlap = {}
    for partition in ("validation", "full_validation"):
        overlap[partition] = {
            key: len(groups["development_replay_pool"][key] & values)
            for key, values in groups[partition].items()
        }
        if any(overlap[partition].values()):
            raise ValueError(f"development/evaluation membership overlap: {partition}")
    if not groups["validation"]["sample_id"] <= groups["full_validation"]["sample_id"]:
        raise ValueError("quick validation is no longer a subset of full validation")
    return {
        "baseline_identity_sha256": supplied,
        "integrity_passed": True,
        "pinned_files_verified": len(record["pinned_files"]),
        "model_package_valid": True,
        "development_evaluation_overlap": overlap,
        "validation_is_full_validation_subset": True,
        "limits": [
            "Membership metadata and model artifacts verified; dataset image/label bytes not fully rehashed.",
            "No new inference, semantic review, near-duplicate audit or promotion performed.",
            "This verifier is a manual preflight; training CLI does not invoke it automatically.",
        ],
        "training_data_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=ROOT / "config/perception/visual_experiment_baseline_v1.json")
    args = parser.parse_args()
    try:
        result = verify(args.contract)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"integrity_passed": False, "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
