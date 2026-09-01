#!/usr/bin/env python3
"""Join proposal failures to frozen dataset lineage without rerunning inference."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


RECORDING_RE = re.compile(
    r"^visual-v2-(?P<partition>development|validation)-"
    r"(?P<condition>.+)-(?P<seed>\d+)-r(?P<run>\d+)$"
)
AREA_BANDS = (
    ("lt_0_003", 0.0, 0.003),
    ("0_003_to_0_01", 0.003, 0.01),
    ("0_01_to_0_03", 0.01, 0.03),
    ("gte_0_03", 0.03, float("inf")),
)


def _load_membership(path: Path) -> dict[str, dict]:
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {row["sample_id"]: row for row in rows}


def _lineage(recording_id: str) -> dict:
    match = RECORDING_RE.match(recording_id)
    if match is None:
        raise ValueError(f"unsupported recording_id: {recording_id}")
    row = match.groupdict()
    row["seed"] = int(row["seed"])
    row["run"] = int(row["run"])
    return row


def _area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _area_band(area: float) -> str:
    for name, lower, upper in AREA_BANDS:
        if lower <= area < upper:
            return name
    raise AssertionError(area)


def _nested_counts(counter: Counter[tuple[str, ...]]) -> list[dict]:
    return [
        {"keys": list(keys), "failure_count": count}
        for keys, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    diagnostic = json.loads(args.diagnostic.read_text())
    membership = _load_membership(args.membership)
    by_recording = Counter()
    by_condition = Counter()
    by_condition_class_reason = Counter()
    by_class_reason_area = Counter()
    failure_frames: dict[str, set[str]] = defaultdict(set)
    enriched = []
    for failure in diagnostic["failures"]:
        member = membership.get(failure["sample_id"])
        if member is None:
            raise ValueError(f"missing membership: {failure['sample_id']}")
        recording_id = member["recording_id"]
        lineage = _lineage(recording_id)
        area = _area(failure["truth_bbox"])
        band = _area_band(area)
        class_name = failure["class_name"]
        reason = failure["reason"]
        by_recording[(recording_id, class_name, reason, band)] += 1
        by_condition[(lineage["condition"], class_name, reason, band)] += 1
        by_condition_class_reason[(lineage["condition"], class_name, reason)] += 1
        by_class_reason_area[(class_name, reason, band)] += 1
        failure_frames[recording_id].add(failure["sample_id"])
        enriched.append(
            {
                "sample_id": failure["sample_id"],
                "recording_id": recording_id,
                **lineage,
                "class_name": class_name,
                "reason": reason,
                "truth_bbox_area_fraction": area,
                "area_band": band,
            }
        )

    recording_summary = []
    for recording_id, sample_ids in failure_frames.items():
        lineage = _lineage(recording_id)
        recording_summary.append(
            {
                "recording_id": recording_id,
                **lineage,
                "failure_frame_count": len(sample_ids),
                "failure_count": sum(
                    count for keys, count in by_recording.items() if keys[0] == recording_id
                ),
            }
        )
    recording_summary.sort(key=lambda row: (-row["failure_count"], row["recording_id"]))

    record = {
        "schema_version": 1,
        "source_diagnostic": str(args.diagnostic),
        "source_diagnostic_identity": diagnostic["identity_sha256"],
        "membership": str(args.membership),
        "failure_count": len(enriched),
        "failure_frame_count": len({row["sample_id"] for row in enriched}),
        "recording_count": len(failure_frames),
        "recording_summary": recording_summary,
        "condition_class_reason_area_counts": _nested_counts(by_condition),
        "condition_class_reason_counts": _nested_counts(by_condition_class_reason),
        "class_reason_area_counts": _nested_counts(by_class_reason_area),
        "recording_class_reason_area_counts": _nested_counts(by_recording),
        "failures": enriched,
        "run18_decision": {
            "branch": "balanced_small_dense_full_label_collection",
            "start_weights": "run15",
            "primary_area_bands": ["lt_0_003", "0_003_to_0_01"],
            "preserve_all_visible_labels": True,
            "frozen_runtime_max_proposals": 16,
            "rationale": [
                "localization and detector misses require additional small-object views",
                "proposal truncation requires dense full-label views, not a higher runtime cap",
                "all four classes fail, so a single-class calibration would risk forgetting",
            ],
        },
    }
    identity_payload = dict(record)
    record["identity_sha256"] = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "identity_sha256": record["identity_sha256"],
        "failure_count": record["failure_count"],
        "recording_count": record["recording_count"],
        "top_recordings": record["recording_summary"][:10],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
