#!/usr/bin/env python3
"""Attribute proposal misses to detector, localization, NMS, cap, or matching."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, write_json
from src.vision.evaluation.detection_metrics import box_iou
from src.vision.evaluation.yolo_evaluation import collect_predictions


def _membership(dataset: Path, partition: str) -> dict[str, dict]:
    path = dataset / "identity" / f"{partition}_membership.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {Path(row["image_relative_path"]).stem: row for row in rows}


def _nms_trace(predictions: list[dict], threshold: float) -> list[dict]:
    kept = []
    for raw_index, row in sorted(
        enumerate(predictions), key=lambda item: -item[1]["confidence"]
    ):
        suppressors = [
            (box_iou(row["bbox"], other["bbox"]), other)
            for other in kept
            if box_iou(row["bbox"], other["bbox"]) >= threshold
        ]
        if not suppressors:
            kept.append({**row, "raw_index": raw_index})
    return kept


def _best_iou(truth: dict, proposals: list[dict]) -> tuple[float, int]:
    return max(
        ((box_iou(truth["bbox"], row["bbox"]), index) for index, row in enumerate(proposals)),
        default=(0.0, -1),
    )


def _greedy_matches(truths: list[dict], proposals: list[dict]) -> dict[int, int]:
    used = set()
    matches = {}
    for truth_index, truth in enumerate(truths):
        choices = [
            (box_iou(truth["bbox"], row["bbox"]), index)
            for index, row in enumerate(proposals)
            if index not in used
        ]
        overlap, proposal_index = max(choices, default=(0.0, -1))
        if overlap >= 0.5:
            used.add(proposal_index)
            matches[truth_index] = proposal_index
    return matches


def _reason(raw_iou: float, nms_iou: float, capped_iou: float) -> str:
    if capped_iou >= 0.5:
        return "shared_proposal_or_matching_order_collision"
    if nms_iou >= 0.5:
        return "max_proposals_truncated"
    if raw_iou >= 0.5:
        return "nms_suppressed"
    if raw_iou >= 0.1:
        return "localization_iou_below_gate"
    return "detector_miss"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--partition", default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-proposals", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    memberships = _membership(args.dataset, args.partition)
    frames = collect_predictions(
        args.model,
        args.dataset,
        args.partition,
        device=args.device,
        imgsz=args.imgsz,
        confidence=0.001,
        batch=1,
    )
    failures = []
    reason_counts = Counter()
    class_reason_counts = defaultdict(Counter)
    frame_summaries = []
    for frame in frames:
        member = memberships.get(frame["sample_id"])
        if member is None:
            raise ValueError(f"missing membership for {frame['sample_id']}")
        raw = [
            row for row in frame["predictions"]
            if row["confidence"] >= args.threshold
        ]
        nms = _nms_trace(raw, args.nms_iou)
        capped = nms[: args.max_proposals]
        matches = _greedy_matches(frame["truth"], capped)
        frame_failure_count = 0
        for truth_index, truth in enumerate(frame["truth"]):
            if truth_index in matches:
                continue
            raw_iou, raw_index = _best_iou(truth, raw)
            nms_iou, nms_index = _best_iou(truth, nms)
            capped_iou, capped_index = _best_iou(truth, capped)
            reason = _reason(raw_iou, nms_iou, capped_iou)
            reason_counts[reason] += 1
            class_reason_counts[truth["class_name"]][reason] += 1
            frame_failure_count += 1
            failures.append(
                {
                    "sample_id": member["sample_id"],
                    "recording_id": member.get("recording_id"),
                    "source_frame_id": member.get("source_frame_id"),
                    "source_collection_identity": member.get("source_collection_identity"),
                    "map_id": member.get("map_id"),
                    "seed": member.get("seed"),
                    "truth_index": truth_index,
                    "class_name": truth["class_name"],
                    "truth_bbox": truth["bbox"],
                    "reason": reason,
                    "raw_best_iou": raw_iou,
                    "raw_best_index": raw_index,
                    "raw_best_prediction": raw[raw_index] if raw_index >= 0 else None,
                    "nms_best_iou": nms_iou,
                    "nms_best_index": nms_index,
                    "capped_best_iou": capped_iou,
                    "capped_best_index": capped_index,
                    "raw_count": len(raw),
                    "nms_count": len(nms),
                    "capped_count": len(capped),
                }
            )
        frame_summaries.append(
            {
                "sample_id": member["sample_id"],
                "recording_id": member.get("recording_id"),
                "source_frame_id": member.get("source_frame_id"),
                "map_id": member.get("map_id"),
                "seed": member.get("seed"),
                "truth_count": len(frame["truth"]),
                "raw_count": len(raw),
                "nms_count": len(nms),
                "capped_count": len(capped),
                "failure_count": frame_failure_count,
            }
        )

    record = {
        "schema_version": 1,
        "model": str(args.model),
        "model_sha256": file_sha256(args.model),
        "dataset": str(args.dataset),
        "partition": args.partition,
        "threshold": args.threshold,
        "nms_iou": args.nms_iou,
        "max_proposals": args.max_proposals,
        "frame_count": len(frames),
        "failure_count": len(failures),
        "reason_counts": dict(sorted(reason_counts.items())),
        "class_reason_counts": {
            name: dict(sorted(counts.items()))
            for name, counts in sorted(class_reason_counts.items())
        },
        "failure_frames": [row for row in frame_summaries if row["failure_count"]],
        "failures": failures,
    }
    record["identity_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(args.output, record)
    print(
        json.dumps(
            {
                "identity_sha256": record["identity_sha256"],
                "failure_count": record["failure_count"],
                "reason_counts": record["reason_counts"],
                "class_reason_counts": record["class_reason_counts"],
                "failure_frame_count": len(record["failure_frames"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
