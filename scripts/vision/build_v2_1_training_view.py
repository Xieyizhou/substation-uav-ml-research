#!/usr/bin/env python3
"""Combine v2 development replay with the curated v2.1 hard-example view."""

import argparse
from collections import defaultdict
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


CLASSES = ("transformer", "switchgear", "capacitor_bank", "reactor")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _select_replay(rows, per_class):
    selected, used = [], set()
    for class_name in CLASSES:
        groups = defaultdict(list)
        for row in rows:
            if class_name in row["classes"] and row["sample_id"] not in used:
                groups[row["recording_id"]].append(row)
        for values in groups.values():
            values.sort(key=lambda row: hashlib.sha256((class_name + ":" + row["sample_id"]).encode()).hexdigest())
        group_ids = sorted(groups, key=lambda value: hashlib.sha256((class_name + ":" + value).encode()).hexdigest())
        chosen, offset = [], 0
        while len(chosen) < per_class and group_ids:
            next_groups = []
            for group_id in group_ids:
                if offset < len(groups[group_id]):
                    row = groups[group_id][offset]
                    if row["sample_id"] not in used:
                        chosen.append(row); used.add(row["sample_id"])
                        if len(chosen) == per_class:
                            break
                if offset + 1 < len(groups[group_id]):
                    next_groups.append(group_id)
            group_ids = next_groups
            offset += 1
        if len(chosen) != per_class:
            raise ValueError(f"v2 development replay lacks {per_class} exclusive {class_name} frames")
        selected.extend((class_name, row) for row in chosen)
    return selected


def _link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError(f"duplicate training-view destination: {destination.name}")
    os.link(source, destination)


def _materialize_hard_image(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError(f"duplicate training-view destination: {destination.name}")
    with Image.open(source) as image:
        image.save(destination, format="PNG", optimize=False, compress_level=6)


def _class_counts(label_paths):
    counts = {name: 0 for name in CLASSES}
    for path in label_paths:
        for line in path.read_text().splitlines():
            if line.strip():
                counts[CLASSES[int(line.split()[0])]] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-view", type=Path, required=True)
    parser.add_argument("--hard-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-per-class", type=int, default=1000)
    parser.add_argument("--view-version", choices=("v2.1", "v2.2"), default="v2.1")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("v2.1 training-view output must be absent or empty")
    identity_dir = args.output / "identity"; identity_dir.mkdir(parents=True, exist_ok=True)
    v2_identity = json.loads((args.v2_view / "identity/training_view_identity.json").read_text())
    hard_receipt = json.loads((args.hard_view / "training-view-receipt.json").read_text())
    if hard_receipt.get("status") != "complete" or hard_receipt.get("member_count", 0) <= 0:
        raise ValueError("hard-example view is not complete")
    hard_role = "v2_1_hard_example" if args.view_version == "v2.1" else "v2_2_hard_example"
    replay = _select_replay(_read_jsonl(args.v2_view / "identity/train_membership.jsonl"), args.replay_per_class)
    hard_rows = _read_jsonl(args.hard_view / "membership.jsonl")
    train_members, validation_members, train_labels, validation_labels = [], [], [], []
    for class_name, row in replay:
        token = hashlib.sha256(row["sample_id"].encode()).hexdigest()
        image_source, label_source = args.v2_view / row["image_relative_path"], args.v2_view / row["label_relative_path"]
        image_relative = Path("images/train") / f"replay-{token}.png"
        label_relative = Path("labels/train") / f"replay-{token}.txt"
        _link(image_source, args.output / image_relative); _link(label_source, args.output / label_relative)
        train_labels.append(args.output / label_relative)
        train_members.append({**row, "replay_quota_class": class_name, "source_role": "v2_development_replay", "image_relative_path": image_relative.as_posix(), "label_relative_path": label_relative.as_posix()})
    for row in hard_rows:
        destination_split = "train" if row["split"] == "development" else "validation"
        image_relative = Path("images") / destination_split / f"hard-{row['sample_id']}.png"
        label_relative = Path("labels") / destination_split / f"hard-{row['sample_id']}.txt"
        _materialize_hard_image(args.hard_view / row["image_relative_path"], args.output / image_relative)
        _link(args.hard_view / row["label_relative_path"], args.output / label_relative)
        member = {**row, "source_role": hard_role, "source_image_sha256": row["image_sha256"], "image_sha256": _sha256(args.output / image_relative), "image_relative_path": image_relative.as_posix(), "label_relative_path": label_relative.as_posix()}
        if destination_split == "train":
            train_members.append(member); train_labels.append(args.output / label_relative)
        else:
            validation_members.append(member); validation_labels.append(args.output / label_relative)
    train_members.sort(key=lambda row: (row["source_role"], row["sample_id"]))
    validation_members.sort(key=lambda row: row["sample_id"])
    train_membership = identity_dir / "train_membership.jsonl"
    validation_membership = identity_dir / "validation_membership.jsonl"
    _write_jsonl(train_membership, train_members); _write_jsonl(validation_membership, validation_members)
    full_membership = identity_dir / "full_validation_membership.jsonl"
    shutil.copy2(args.v2_view / "identity/full_validation_membership.jsonl", full_membership)
    labels_manifest = {
        "schema_version": 1, "classes": list(CLASSES), "hard_view_identity": hard_receipt["identity"],
        "v2_training_view_identity": v2_identity["training_view_identity_sha256"], "replay_per_class": args.replay_per_class,
    }
    labels_manifest_path = identity_dir / "labels_manifest.json"
    labels_manifest_path.write_text(json.dumps(labels_manifest, indent=2, sort_keys=True) + "\n")
    (args.output / "images").mkdir(exist_ok=True); (args.output / "labels").mkdir(exist_ok=True)
    os.symlink((args.v2_view / "images/full_validation").resolve(), args.output / "images/full_validation")
    os.symlink((args.v2_view / "labels/full_validation").resolve(), args.output / "labels/full_validation")
    dataset = [f"path: {args.output.resolve()}", "train: images/train", "val: images/validation", "test: images/full_validation", "names:"]
    dataset.extend(f"  {index}: {name}" for index, name in enumerate(CLASSES))
    (args.output / "dataset.yaml").write_text("\n".join(dataset) + "\n")
    source_development = object_sha256({"v2": v2_identity["training_view_identity_sha256"], "hard": hard_receipt["identity"], "replay_per_class": args.replay_per_class})
    identity = TrainingViewIdentity(
        source_development_dataset_identity=source_development,
        source_validation_dataset_identity=hard_receipt["curated_identity"],
        sampling_algorithm=f"v2_development_grouped_replay_{args.replay_per_class}_per_class_plus_hard_{args.view_version.replace('.', '_')}",
        sampling_seed=7,
        train_membership_sha256=_sha256(train_membership), validation_membership_sha256=_sha256(validation_membership),
        full_validation_membership_sha256=v2_identity["full_validation_membership_sha256"], labels_manifest_sha256=_sha256(labels_manifest_path),
        class_order_identity=v2_identity["class_order_identity"], train_frame_count=len(train_members), validation_frame_count=len(validation_members),
        full_validation_frame_count=v2_identity["full_validation_frame_count"], train_class_counts=_class_counts(train_labels),
        validation_class_counts=_class_counts(validation_labels), train_no_target_count=sum(not row["classes"] for row in train_members),
        validation_no_target_count=sum(not row["classes"] for row in validation_members),
    )
    (identity_dir / "training_view_identity.json").write_text(json.dumps(identity.to_record(), indent=2, sort_keys=True) + "\n")
    report = {"status": "complete", "training_view_identity": identity.training_view_identity_sha256, "train_frames": len(train_members), "validation_frames": len(validation_members), "replay_frames": len(replay), "hard_frames": len(hard_rows)}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
