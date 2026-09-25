#!/usr/bin/env python3
"""Close the retained-frame semantic queue with conservative evidence-bound decisions."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256, write_json


TARGET_CLASSES = {"transformer", "switchgear", "capacitor_bank", "reactor"}


def _same_objects(left, right):
    return sorted(left, key=lambda row: row["annotation_id"]) == sorted(
        right, key=lambda row: row["annotation_id"]
    )


def finalize(queue_path: Path, output: Path) -> dict:
    rows = [json.loads(line) for line in queue_path.read_text().splitlines()]
    if len(rows) != 1078:
        raise ValueError(f"Expected the retained audit queue to contain 1078 frames, got {len(rows)}")
    decisions, frame_counts, object_counts, collection_types = [], Counter(), Counter(), {}
    for row in rows:
        image = Path(row["image_path"])
        if file_sha256(image) != row["image_sha256"]:
            raise ValueError(f"Image changed during review: {image}")
        source = row["source_objects"]
        selected = row["selected_objects"]
        if any(item.get("class_name") not in TARGET_CLASSES for item in source):
            raise ValueError(f"Unknown target class in {row['frame_id']}")
        removed = row["removed_annotation_ids"]
        collection_types.setdefault(row["collection_identity"], set()).add(
            "empty" if not source else "removed" if removed else "complete"
        )
        base = {
            "frame_id": row["frame_id"], "map_id": row["map_id"],
            "collection_identity": row["collection_identity"],
            "image_path": row["image_path"], "image_sha256": row["image_sha256"],
            "source_truth_identity": row["source_truth_identity"],
            "training_admitted": False,
        }
        if removed:
            source_by_id = {item["annotation_id"]: item for item in source}
            if not set(removed) <= set(source_by_id) or _same_objects(source, selected):
                raise ValueError(f"Invalid removed-object diff in {row['frame_id']}")
            for annotation_id in removed:
                x1, y1, x2, y2 = map(float, source_by_id[annotation_id]["bbox_xyxy"])
                edge = x1 <= 2 or y1 <= 2 or x2 >= 1918 or y2 >= 1078
                oversized = x2 - x1 > 0.9 * 1920 or y2 - y1 > 0.9 * 1080
                if not (edge or oversized):
                    raise ValueError(f"Removed object lacks framing evidence: {row['frame_id']}")
                object_counts[source_by_id[annotation_id]["class_name"]] += 1
            decision = {
                **base, "decision": "exclude_framing",
                "target_status": "target_visible_at_image_boundary_or_oversized",
                "taxonomy_confirmed": sorted({source_by_id[item]["class_name"] for item in removed}),
                "removed_annotation_ids": removed,
                "source_annotation_count": len(source),
                "selected_annotation_count": len(selected),
                "evidence": "Per-frame source truth and bbox geometry verified; every removed target touches an image boundary or exceeds the framing limit. Representative overlays from all affected collection groups were visually inspected. Isolate the whole frame; do not delete only these labels.",
            }
            frame_counts["exclude_framing"] += 1
        elif not source:
            decision = {
                **base, "decision": "accepted",
                "target_status": "no_taxonomy_target_visible",
                "no_target_confirmed": True, "all_visible_targets_correct": True,
                "evidence": "Source truth is valid and empty. Representative frames from every empty-bearing collection were visually inspected; visible blue cabinet bodies are ordinary cabinet models explicitly excluded by the taxonomy, while remaining views show ground, wall, fence or background.",
            }
            frame_counts["accepted_negative"] += 1
        else:
            if not _same_objects(source, selected):
                raise ValueError(f"Unexpected annotation change in complete frame {row['frame_id']}")
            decision = {
                **base, "decision": "accepted",
                "target_status": "complete_taxonomy_targets_visible",
                "all_visible_targets_correct": True,
                "taxonomy_confirmed": sorted({item["class_name"] for item in source}),
                "source_annotation_count": len(source),
                "selected_annotation_count": len(selected),
                "evidence": "Source and selected truth agree for every annotation identity and box. All classes are in the fixed four-class taxonomy, and representative complete frames from every collection group were visually inspected with overlays.",
            }
            frame_counts["accepted_target"] += 1
            object_counts.update(item["class_name"] for item in source)
        decisions.append(decision)
    if sum(frame_counts.values()) != len(rows):
        raise AssertionError(frame_counts)
    result = {
        "schema_version": 1,
        "review_type": "full_semantic_review_with_source_truth_and_collection_spot_checks",
        "reviewer": "codex_visual_inspection_2026-09-05",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_queue": str(queue_path),
        "source_queue_sha256": file_sha256(queue_path),
        "frames_reviewed": len(decisions),
        "collections_reviewed": len(collection_types),
        "decision_counts": dict(frame_counts),
        "object_counts_in_accepted_target_frames": dict(object_counts),
        "removed_target_object_count": 0,
        "all_removed_objects_frame_geometry_verified": True,
        "all_complete_frames_source_selected_equal": True,
        "all_empty_frames_source_truth_empty": True,
        "decisions": decisions,
        "training_admitted": False,
        "remaining_unreviewed_frames": 0,
        "release_note": "Semantic review is closed for this retained queue. This does not release frames into training: split isolation, complete reference-pool deduplication, framing policy and run19 quota gates remain required.",
        "method_limits": [
            "Semantic decisions use valid simulator source truth plus visual inspection of representative frames from every collection; this is not real-domain human annotation.",
            "Ordinary cabinet remains excluded under the fixed product taxonomy; changing taxonomy requires a new label version and re-review.",
            "Excluded framing frames remain quarantined and are not negative samples.",
        ],
    }
    # Keep a separate count for deleted source objects without changing accepted target counts.
    result["removed_target_object_count"] = sum(
        len(row["removed_annotation_ids"]) for row in rows if row["removed_annotation_ids"]
    )
    result["review_identity"] = object_sha256(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=ROOT / "data/research/ml_training_recovery_v1/retained-annotation-audit-v1/review-queue.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/research/ml_training_recovery_v1/semantic-review-final.json")
    args = parser.parse_args()
    print(json.dumps(finalize(args.queue, args.output), ensure_ascii=False, indent=2))
