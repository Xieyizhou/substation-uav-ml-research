#!/usr/bin/env python3
"""Materialize synchronized RGB and normalized Gazebo truth as a YOLO view."""

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

from src.ml import EQUIPMENT_CLASSES
from src.vision.training.hard_example_view import audit_members, load_protocol
from src.vision.training.labels import BoundingBoxLabel


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _nearest_truth(timestamp, truth_rows, max_skew_seconds):
    if not truth_rows:
        raise ValueError("truth stream is empty")
    row = min(truth_rows, key=lambda item: (abs(float(item["simulation_timestamp"]) - timestamp), item["message_id"]))
    skew = abs(float(row["simulation_timestamp"]) - timestamp)
    if skew > max_skew_seconds:
        raise ValueError(f"RGB-truth skew {skew * 1000:.3f} ms exceeds limit")
    if row.get("validation_status") != "valid":
        raise ValueError(f"invalid truth frame: {row.get('invalid_reasons', [])}")
    return row, skew


def _label_text(objects, width, height):
    lines = []
    for item in objects:
        if item.get("validation_status") != "validated":
            raise ValueError("truth contains an unvalidated object")
        x_min, y_min, x_max, y_max = item["bbox_xyxy"]
        lines.append(BoundingBoxLabel(item["class_name"], x_min, y_min, x_max, y_max, width, height).to_yolo())
    return "\n".join(lines) + ("\n" if lines else "")


def materialize(collection, truth_path, output, protocol_path, max_skew_ms):
    receipt = json.loads((collection / "collection-receipt.json").read_text())
    protocol = load_protocol(protocol_path)
    truth_rows = _load_jsonl(truth_path)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for member in receipt["members"]:
        timestamp = float(member["rgb_timestamp"])
        truth, skew = _nearest_truth(timestamp, truth_rows, max_skew_ms / 1000.0)
        source = collection / member["rgb_path"]
        digest = _sha256(source)
        if digest != member["image_sha256"]:
            raise ValueError(f"image hash mismatch: {member['frame_id']}")
        split = member["split"]
        basename = hashlib.sha256(member["frame_id"].encode()).hexdigest()
        image_relative = Path("images") / split / f"{basename}{source.suffix.lower()}"
        label_relative = Path("labels") / split / f"{basename}.txt"
        (output / image_relative).parent.mkdir(parents=True, exist_ok=True)
        (output / label_relative).parent.mkdir(parents=True, exist_ok=True)
        os.link(source, output / image_relative)
        objects = truth["objects"]
        (output / label_relative).write_text(_label_text(objects, member["image_width"], member["image_height"]), encoding="utf-8")
        rows.append({
            "sample_id": member["frame_id"], "seed": member["seed"], "split": split,
            "source_partition": split, "image_sha256": digest,
            "perceptual_hash": member["perceptual_hash"],
            "image_relative_path": image_relative.as_posix(),
            "label_relative_path": label_relative.as_posix(),
            "classes": sorted({item["class_name"] for item in objects}),
            "annotation_status": "labelled" if objects else "verified_no_target",
            "truth_message_id": truth["message_id"], "truth_skew_ms": skew * 1000.0,
        })
    audit = audit_members(rows, protocol)
    manifest_path = output / "membership.jsonl"
    manifest_path.write_text("".join(_canonical(row) + "\n" for row in sorted(rows, key=lambda item: item["sample_id"])), encoding="utf-8")
    data_yaml = "path: .\ntrain: images/development\nval: images/validation\nnames:\n" + "".join(f"  {index}: {name}\n" for index, name in enumerate(EQUIPMENT_CLASSES))
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    result = {
        "schema_version": 1, "protocol_id": protocol["protocol_id"],
        "collection_identity": receipt["identity"], "truth_stream_sha256": _sha256(truth_path),
        "frame_count": len(rows), "target_frame_count": sum(row["annotation_status"] == "labelled" for row in rows),
        "no_target_frame_count": sum(row["annotation_status"] == "verified_no_target" for row in rows),
        "max_truth_skew_ms": max((row["truth_skew_ms"] for row in rows), default=0.0),
        "membership_sha256": _sha256(manifest_path), "audit": audit,
    }
    result["identity"] = hashlib.sha256(_canonical(result).encode()).hexdigest()
    (output / "materialization-receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "config/perception/visual_hard_examples_v2_1.json")
    parser.add_argument("--max-skew-ms", type=float, default=33.334)
    args = parser.parse_args()
    if args.max_skew_ms <= 0:
        parser.error("max-skew-ms must be positive")
    result = materialize(args.collection, args.truth, args.output, args.protocol, args.max_skew_ms)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
