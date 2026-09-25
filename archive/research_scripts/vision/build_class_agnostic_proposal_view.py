#!/usr/bin/env python3
"""Collapse a four-class visual view into a deterministic equipment proposal view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.artifacts import object_sha256
from src.vision.contracts.training_identity import TrainingViewIdentity


SOURCE_CLASSES = ("transformer", "switchgear", "capacitor_bank", "reactor")
PROPOSAL_CLASSES = ("equipment",)
SPLITS = ("train", "validation", "full_validation")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _collapse_label(source: Path, destination: Path) -> int:
    lines = []
    for raw in source.read_text().splitlines():
        values = raw.split()
        if not values:
            continue
        class_id = int(values[0])
        if class_id < 0 or class_id >= len(SOURCE_CLASSES):
            raise ValueError(f"unknown source class id {class_id} in {source}")
        if len(values) != 5:
            raise ValueError(f"invalid YOLO detection label in {source}")
        lines.append("0 " + " ".join(values[1:]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def _collapse_members(rows: list[dict]) -> list[dict]:
    collapsed = []
    for row in rows:
        source_classes = sorted(set(row.get("classes", [])))
        unknown = sorted(set(source_classes) - set(SOURCE_CLASSES))
        if unknown:
            raise ValueError(f"unknown source membership classes: {unknown}")
        collapsed.append(
            {
                **row,
                "classes": ["equipment"] if source_classes else [],
                "proposal_source_classes": source_classes,
                "proposal_label_transform": "four_class_to_equipment_v1",
            }
        )
    return collapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view-version", default="run15-equipment-proposal-v1")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("proposal training-view output must be absent or empty")

    source_identity = TrainingViewIdentity.from_record(
        json.loads((args.input / "identity/training_view_identity.json").read_text())
    )
    identity_dir = args.output / "identity"
    identity_dir.mkdir(parents=True, exist_ok=True)
    (args.output / "images").mkdir(exist_ok=True)
    (args.output / "labels").mkdir(exist_ok=True)

    memberships = {}
    box_counts = {}
    for split in SPLITS:
        membership_name = (
            "train_membership.jsonl"
            if split == "train"
            else f"{split}_membership.jsonl"
        )
        rows = _collapse_members(_read_jsonl(args.input / "identity" / membership_name))
        memberships[split] = rows
        _write_jsonl(identity_dir / membership_name, rows)

        box_count = 0
        for row in rows:
            source_image = (args.input / row["image_relative_path"]).resolve()
            destination_image = args.output / row["image_relative_path"]
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            if destination_image.exists():
                raise ValueError(
                    f"duplicate proposal-view image destination: {destination_image}"
                )
            os.link(source_image, destination_image)
            source_label = args.input / row["label_relative_path"]
            destination_label = args.output / row["label_relative_path"]
            box_count += _collapse_label(source_label, destination_label)
        box_counts[split] = box_count

    labels_manifest = {
        "schema_version": 1,
        "classes": list(PROPOSAL_CLASSES),
        "source_classes": list(SOURCE_CLASSES),
        "source_training_view_identity": source_identity.training_view_identity_sha256,
        "transform": "four_class_to_equipment_v1",
        "view_version": args.view_version,
        "box_counts": box_counts,
    }
    labels_manifest_path = identity_dir / "labels_manifest.json"
    labels_manifest_path.write_text(
        json.dumps(labels_manifest, indent=2, sort_keys=True) + "\n"
    )

    dataset = [
        f"path: {args.output.resolve()}",
        "train: images/train",
        "val: images/validation",
        "test: images/full_validation",
        "names:",
        "  0: equipment",
    ]
    (args.output / "dataset.yaml").write_text("\n".join(dataset) + "\n")

    train_membership = identity_dir / "train_membership.jsonl"
    validation_membership = identity_dir / "validation_membership.jsonl"
    full_membership = identity_dir / "full_validation_membership.jsonl"
    transform_identity = object_sha256(
        {
            "source": source_identity.training_view_identity_sha256,
            "transform": "four_class_to_equipment_v1",
            "view_version": args.view_version,
        }
    )
    identity = TrainingViewIdentity(
        source_development_dataset_identity=transform_identity,
        source_validation_dataset_identity=object_sha256(
            {
                "source": source_identity.source_validation_dataset_identity,
                "transform": "four_class_to_equipment_v1",
            }
        ),
        sampling_algorithm=f"identity_preserving_{args.view_version}",
        sampling_seed=7,
        train_membership_sha256=_sha256(train_membership),
        validation_membership_sha256=_sha256(validation_membership),
        full_validation_membership_sha256=_sha256(full_membership),
        labels_manifest_sha256=_sha256(labels_manifest_path),
        class_order_identity=object_sha256(list(PROPOSAL_CLASSES)),
        train_frame_count=len(memberships["train"]),
        validation_frame_count=len(memberships["validation"]),
        full_validation_frame_count=len(memberships["full_validation"]),
        train_class_counts={"equipment": box_counts["train"]},
        validation_class_counts={"equipment": box_counts["validation"]},
        train_no_target_count=sum(not row["classes"] for row in memberships["train"]),
        validation_no_target_count=sum(
            not row["classes"] for row in memberships["validation"]
        ),
    )
    (identity_dir / "training_view_identity.json").write_text(
        json.dumps(identity.to_record(), indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "training_view_identity": identity.training_view_identity_sha256,
                "train_frames": len(memberships["train"]),
                "validation_frames": len(memberships["validation"]),
                "full_validation_frames": len(memberships["full_validation"]),
                "box_counts": box_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
