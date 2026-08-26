#!/usr/bin/env python3
"""Append split-safe Complex hard negatives to the frozen v2.9 view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.artifacts import object_sha256
from src.vision.contracts.training_identity import TrainingViewIdentity


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-view", type=Path, required=True)
    parser.add_argument("--negative-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-oversample-factor", type=int, default=8)
    args = parser.parse_args()
    if args.negative_oversample_factor < 1:
        raise ValueError("negative oversample factor must be positive")
    if args.output.exists():
        raise ValueError("v2.10 output must not already exist")

    base_identity = json.loads(
        (args.base_view / "identity/training_view_identity.json").read_text()
    )
    negative_receipt = json.loads(
        (args.negative_view / "training-view-receipt.json").read_text()
    )
    if negative_receipt.get("status") != "complete":
        raise ValueError("negative hard view is not complete")

    shutil.copytree(args.base_view, args.output, copy_function=os.link, symlinks=True)
    # Ultralytics cache files contain the base view's resolved image list. They
    # must never cross a training-view identity boundary after new members are
    # appended, otherwise the added hard negatives are silently skipped.
    for cache_name in ("train.cache", "validation.cache"):
        (args.output / "labels" / cache_name).unlink(missing_ok=True)
    dataset_path = args.output / "dataset.yaml"
    dataset_text = dataset_path.read_text(encoding="utf-8")
    base_path = f"path: {args.base_view.resolve()}"
    if base_path not in dataset_text:
        raise ValueError("base dataset path is not bound to the base view")
    # copytree uses hard links for the large immutable view. Break the metadata
    # link before rebinding it so the frozen base dataset is not modified.
    dataset_path.unlink()
    dataset_path.write_text(
        dataset_text.replace(base_path, f"path: {args.output.resolve()}", 1),
        encoding="utf-8",
    )
    identity_dir = args.output / "identity"
    train = _read_jsonl(identity_dir / "train_membership.jsonl")
    validation = _read_jsonl(identity_dir / "validation_membership.jsonl")
    negatives = _read_jsonl(args.negative_view / "membership.jsonl")
    for row in negatives:
        destination_split = "train" if row["split"] == "development" else "validation"
        replicas = args.negative_oversample_factor if destination_split == "train" else 1
        for replica in range(1, replicas + 1):
            suffix = "" if replica == 1 else f"-replica-{replica}"
            basename = f"hard-negative-v2-10-{row['sample_id']}{suffix}"
            image_relative = Path("images") / destination_split / f"{basename}.png"
            label_relative = Path("labels") / destination_split / f"{basename}.txt"
            if replica == 1:
                with Image.open(args.negative_view / row["image_relative_path"]) as image:
                    image.save(args.output / image_relative, format="PNG", compress_level=6)
                first_image = args.output / image_relative
            else:
                os.link(first_image, args.output / image_relative)
            os.link(
                args.negative_view / row["label_relative_path"],
                args.output / label_relative,
            )
            member = {
                **row,
                "sample_id": row["sample_id"] if replica == 1 else f"{row['sample_id']}:negative:replica:{replica}",
                "source_role": "v2_10_complex_hard_negative" if replica == 1 else "v2_10_complex_hard_negative_oversample",
                "oversample_source_sample_id": row["sample_id"],
                "oversample_replica": replica,
                "image_relative_path": image_relative.as_posix(),
                "label_relative_path": label_relative.as_posix(),
            }
            (train if destination_split == "train" else validation).append(member)

    train.sort(key=lambda row: (row.get("source_role", ""), row["sample_id"]))
    validation.sort(key=lambda row: (row.get("source_role", ""), row["sample_id"]))
    train_path = identity_dir / "train_membership.jsonl"
    validation_path = identity_dir / "validation_membership.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(validation_path, validation)

    labels_manifest = {
        "schema_version": 1,
        "base_training_view_identity": base_identity["training_view_identity_sha256"],
        "negative_view_identity": negative_receipt["identity"],
        "negative_curated_identity": negative_receipt["curated_identity"],
        "negative_oversample_factor": args.negative_oversample_factor,
    }
    labels_path = identity_dir / "labels_manifest.json"
    labels_path.write_text(json.dumps(labels_manifest, indent=2, sort_keys=True) + "\n")

    identity = TrainingViewIdentity(
        source_development_dataset_identity=object_sha256({
            "base": base_identity["training_view_identity_sha256"],
            "negative": negative_receipt["identity"],
            "factor": args.negative_oversample_factor,
        }),
        source_validation_dataset_identity=object_sha256({
            "base": base_identity["source_validation_dataset_identity"],
            "negative": negative_receipt["curated_identity"],
        }),
        sampling_algorithm=f"v2_9_plus_complex_hard_negative_oversample_{args.negative_oversample_factor}",
        sampling_seed=7,
        train_membership_sha256=_sha256(train_path),
        validation_membership_sha256=_sha256(validation_path),
        full_validation_membership_sha256=base_identity["full_validation_membership_sha256"],
        labels_manifest_sha256=_sha256(labels_path),
        class_order_identity=base_identity["class_order_identity"],
        train_frame_count=len(train),
        validation_frame_count=len(validation),
        full_validation_frame_count=base_identity["full_validation_frame_count"],
        train_class_counts=base_identity["train_class_counts"],
        validation_class_counts=base_identity["validation_class_counts"],
        train_no_target_count=base_identity["train_no_target_count"] + 20 * args.negative_oversample_factor,
        validation_no_target_count=base_identity["validation_no_target_count"] + 10,
    )
    identity_path = identity_dir / "training_view_identity.json"
    identity_path.write_text(json.dumps(identity.to_record(), indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "complete",
        "training_view_identity": identity.training_view_identity_sha256,
        "train_frames": len(train),
        "validation_frames": len(validation),
        "full_validation_frames": identity.full_validation_frame_count,
        "negative_oversample_factor": args.negative_oversample_factor,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
