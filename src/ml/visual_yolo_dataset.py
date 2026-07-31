"""YOLO filesystem and label materialization helpers."""

from __future__ import annotations

from collections import Counter
import errno
import hashlib
import json
import os
from pathlib import Path

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256
from src.ml.yolo_labels import BoundingBoxLabel


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            output.write("\n")


def link_image(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError as error:
        if error.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES}:
            raise
        destination.symlink_to(os.path.relpath(source, destination.parent))
        return "symlink"


def yolo_label_text(annotation):
    if annotation["annotation_status"] == "verified_no_target":
        if annotation["objects"]:
            raise ValueError("verified_no_target annotation contains objects")
        return ""
    if annotation["annotation_status"] != "labelled":
        raise ValueError("training view contains invalid truth")
    lines = []
    for obj in annotation["objects"]:
        if obj.get("validation_status") != "validated":
            raise ValueError("training view contains an unvalidated object")
        lines.append(
            BoundingBoxLabel(
                class_name=obj["class_name"],
                x_min=obj["bbox_xyxy"][0],
                y_min=obj["bbox_xyxy"][1],
                x_max=obj["bbox_xyxy"][2],
                y_max=obj["bbox_xyxy"][3],
                image_width=annotation["image_width"],
                image_height=annotation["image_height"],
            ).to_yolo()
        )
    if not lines:
        raise ValueError("labelled annotation has no objects")
    return "\n".join(lines) + "\n"


def materialize_yolo_partition(name, rows, collection_root, output_root):
    membership, labels_manifest = [], []
    link_modes, class_counts = Counter(), Counter()
    no_target = 0
    for row in rows:
        annotation = row["annotation"]
        basename = hashlib.sha256(row["sample_id"].encode("utf-8")).hexdigest()
        image_relative = Path("images") / name / f"{basename}.png"
        label_relative = Path("labels") / name / f"{basename}.txt"
        source = (
            collection_root
            / "recordings"
            / row["recording_id"]
            / row["payload_relative_path"]
        )
        if file_sha256(source) != row["payload_sha256"]:
            raise ValueError(f"payload hash mismatch for {row['sample_id']}")
        link_modes[link_image(source, output_root / image_relative)] += 1
        label_path = output_root / label_relative
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(yolo_label_text(annotation), encoding="utf-8")
        label_sha = file_sha256(label_path)
        classes = sorted({obj["class_name"] for obj in annotation["objects"]})
        class_counts.update(classes)
        no_target += annotation["annotation_status"] == "verified_no_target"
        membership.append(
            {
                "sample_id": row["sample_id"],
                "recording_id": row["recording_id"],
                "sequence_number": annotation["sequence_number"],
                "payload_sha256": row["payload_sha256"],
                "image_relative_path": image_relative.as_posix(),
                "label_relative_path": label_relative.as_posix(),
                "label_sha256": label_sha,
                "classes": classes,
                "annotation_status": annotation["annotation_status"],
            }
        )
        labels_manifest.append({"sample_id": row["sample_id"], "sha256": label_sha})
    membership_path = output_root / "identity" / f"{name}_membership.jsonl"
    _write_jsonl(membership_path, membership)
    return {
        "membership_sha256": file_sha256(membership_path),
        "labels_manifest": labels_manifest,
        "class_counts": {item: class_counts[item] for item in EQUIPMENT_CLASSES},
        "no_target_count": no_target,
        "frame_count": len(membership),
        "link_modes": dict(link_modes),
    }
