#!/usr/bin/env python3
"""Materialize a curated hard-example manifest as a split-safe YOLO view."""

import argparse
import hashlib
import json
import os
from pathlib import Path


CLASSES = ("transformer", "switchgear", "capacitor_bank", "reactor")
CLASS_IDS = {name: index for index, name in enumerate(CLASSES)}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _labels(objects, width=1920.0, height=1080.0):
    rows = []
    for item in objects:
        x1, y1, x2, y2 = map(float, item["bbox_xyxy"])
        x1, x2 = max(0.0, x1), min(width, x2)
        y1, y2 = max(0.0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("curated truth contains an empty bounding box")
        rows.append(f"{CLASS_IDS[item['class_name']]} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
    return "\n".join(rows) + ("\n" if rows else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if (
        manifest.get("status") != "complete"
        or not isinstance(manifest.get("selected_count"), int)
        or manifest["selected_count"] <= 0
        or any(manifest.get("shortfall", {}).values())
    ):
        raise ValueError("curated manifest has not passed its frozen quota gate")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output training view must be absent or empty")
    membership = []
    for row in manifest["selected"]:
        split = row["split"]
        if split not in {"development", "validation"}:
            raise ValueError("unsupported curated split")
        source = Path(row["collection"]) / row["rgb_path"]
        if _sha256(source) != row["image_sha256"]:
            raise ValueError("curated source image SHA256 mismatch")
        sample_id = hashlib.sha256(
            f"{row['collection_identity']}:{row['frame_id']}".encode()
        ).hexdigest()
        image_relative = Path("images") / split / f"{sample_id}.ppm"
        label_relative = Path("labels") / split / f"{sample_id}.txt"
        image_path, label_path = args.output / image_relative, args.output / label_relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if image_path.exists() or label_path.exists():
            raise ValueError("duplicate curated frame_id")
        os.link(source, image_path)
        label_text = _labels(row["objects"])
        label_path.write_text(label_text)
        membership.append({
            "sample_id": sample_id, "source_frame_id": row["frame_id"], "map_id": row["map_id"], "seed": row["seed"], "split": split,
            "image_relative_path": image_relative.as_posix(), "label_relative_path": label_relative.as_posix(),
            "image_sha256": row["image_sha256"], "label_sha256": hashlib.sha256(label_text.encode()).hexdigest(),
            "perceptual_hash": row["perceptual_hash"], "classes": sorted({item["class_name"] for item in row["objects"]}),
            "annotation_status": "labelled" if row["objects"] else "verified_no_target",
            "source_collection_identity": row["collection_identity"],
        })
    membership.sort(key=lambda row: row["sample_id"])
    membership_path = args.output / "membership.jsonl"
    membership_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in membership))
    data_yaml = [f"path: {args.output.resolve()}", "train: images/development", "val: images/validation", "names:"]
    data_yaml.extend(f"  {index}: {name}" for index, name in enumerate(CLASSES))
    (args.output / "data.yaml").write_text("\n".join(data_yaml) + "\n")
    split_counts = {split: sum(row["split"] == split for row in membership) for split in ("development", "validation")}
    receipt = {
        "schema_version": 1, "status": "complete", "curated_identity": manifest["identity"],
        "member_count": len(membership), "split_counts": split_counts,
        "membership_sha256": _sha256(membership_path), "classes": list(CLASSES),
    }
    receipt["identity"] = hashlib.sha256(_canonical(receipt).encode()).hexdigest()
    (args.output / "training-view-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
