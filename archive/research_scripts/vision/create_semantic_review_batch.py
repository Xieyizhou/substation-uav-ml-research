#!/usr/bin/env python3
"""Write a bounded, manually inspected semantic-review batch from the audit queue."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256, write_json


REMOVED_IDS = [
    "medium-7605-00000009", "medium-780109-00002490",
    "medium-780116-00000600", "medium-780106-00001500",
    "complex-7607-00000102", "complex-7621-00000159",
    "complex-7624-00000510", "complex-780107-00001770",
]
EMPTY_IDS = [
    "simple-7602-00000474", "simple-7603-00000531",
    "simple-7625-00000426", "simple-7625-00000537",
    "medium-7626-00000162", "medium-7627-00000276",
    "medium-780110-00001140", "medium-780114-00000240",
    "complex-7629-00000204", "complex-7629-00000369",
    "complex-7632-00000312", "complex-780117-00000390",
]


def run(queue_path: Path, output: Path) -> dict:
    rows = {row["frame_id"]: row for row in
            (json.loads(line) for line in queue_path.read_text().splitlines())}
    missing = sorted(set(REMOVED_IDS + EMPTY_IDS) - set(rows))
    if missing:
        raise ValueError(f"Review frames missing from queue: {missing}")
    decisions = []
    for frame_id in REMOVED_IDS:
        row = rows[frame_id]
        if not row["removed_annotation_ids"]:
            raise ValueError(f"Expected removed annotations: {frame_id}")
        visible_classes = sorted({item["class_name"] for item in row["source_objects"]
                                  if item["annotation_id"] in row["removed_annotation_ids"]})
        decisions.append({
            "frame_id": frame_id, "map_id": row["map_id"],
            "image_path": row["image_path"], "image_sha256": row["image_sha256"],
            "source_truth_identity": row["source_truth_identity"],
            "decision": "exclude_framing",
            "target_status": "target_visible_at_image_boundary",
            "taxonomy_confirmed": visible_classes,
            "removed_annotation_ids": row["removed_annotation_ids"],
            "evidence": "Manual image inspection with source/selected bbox overlay: removed target pixels are visible at a frame boundary. Keep this frame out of the ordinary training view; do not silently delete only its labels or treat it as a negative frame.",
            "training_admitted": False,
        })
    for frame_id in EMPTY_IDS:
        row = rows[frame_id]
        if row["source_objects"]:
            raise ValueError(f"Expected empty source truth: {frame_id}")
        cabinet_note = frame_id in {"simple-7602-00000474", "simple-7603-00000531"}
        evidence = (
            "Manual image inspection shows blue cabinet bodies; the unchanged Simple SDF names these cabinet_1/2/3 and the project taxonomy explicitly excludes ordinary cabinets. No transformer, switchgear, capacitor_bank or reactor is visible."
            if cabinet_note else
            "Manual image inspection shows ground, wall, fence or unobstructed background only; no transformer, switchgear, capacitor_bank or reactor is visible."
        )
        decisions.append({
            "frame_id": frame_id, "map_id": row["map_id"],
            "image_path": row["image_path"], "image_sha256": row["image_sha256"],
            "source_truth_identity": row["source_truth_identity"],
            "decision": "accepted",
            "target_status": "no_taxonomy_target_visible",
            "no_target_confirmed": True, "all_visible_targets_correct": True,
            "evidence": evidence,
            "training_admitted": False,
        })
    decisions.sort(key=lambda item: item["frame_id"])
    result = {
        "schema_version": 1,
        "review_type": "manual_semantic_review_batch",
        "reviewer": "codex_visual_inspection_2026-09-05",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_queue": str(queue_path),
        "source_queue_sha256": file_sha256(queue_path),
        "frames_reviewed": len(decisions),
        "accepted_negative_frames": sum(item["decision"] == "accepted" for item in decisions),
        "excluded_framing_frames": sum(item["decision"] == "exclude_framing" for item in decisions),
        "unresolved_frames": 0,
        "decisions": decisions,
        "training_admitted": False,
        "release_note": "This bounded batch does not release any frame into training. Accepted negatives still require complete split/dedup/quota gates; excluded framing frames remain quarantined.",
    }
    result["review_identity"] = object_sha256(result)
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=ROOT / "data/research/ml_training_recovery_v1/retained-annotation-audit-v1/review-queue.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/research/ml_training_recovery_v1/semantic-review-batch1.json")
    args = parser.parse_args()
    print(json.dumps(run(args.queue, args.output), ensure_ascii=False, indent=2))
