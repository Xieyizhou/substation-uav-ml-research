"""Freeze the reviewed 192 positives and isolated 48 negatives as one dataset."""
import json
import sys
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256, write_json

V1 = ROOT / "data/research/ml_training_recovery_v1/visual-augmentation-240-v1"
NEGATIVE = ROOT / "data/research/ml_training_recovery_v1/hard-negative-isolated-v2/frozen-intake-ledger.json"
NEGATIVE_REVIEW = ROOT / "data/research/ml_training_recovery_v1/hard-negative-isolated-v2/semantic-review.json"
OUT = ROOT / "data/research/ml_training_recovery_v1/visual-augmentation-240-v2"
REFERENCE_INDEXES = (
    ROOT / "data/research/ml_training_recovery_v1/pixel-dedup-v1/pixel-fingerprints.jsonl",
    ROOT / "data/research/ml_training_recovery_v1/reference-group-audit-v1/supplemental-protected-fingerprints.jsonl",
)


def load_json(path):
    return json.loads(path.read_text())


def load_fingerprints():
    file_hashes, pixel_hashes, inputs = set(), set(), {}
    for path in REFERENCE_INDEXES:
        inputs[str(path)] = file_sha256(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("image_sha256"):
                file_hashes.add(row["image_sha256"])
            if row.get("pixel_sha256"):
                pixel_hashes.add(row["pixel_sha256"])
    return file_hashes, pixel_hashes, inputs


def pixel_hash(path):
    import hashlib
    with Image.open(path) as image:
        return hashlib.sha256(image.convert("RGB").tobytes() + f"{image.width}x{image.height}".encode()).hexdigest()


def freeze():
    planned_path = V1 / "intake-ledger.json"
    review_path = V1 / "positive-semantic-review.json"
    planned, review, negative, negative_review = map(load_json, (planned_path, review_path, NEGATIVE, NEGATIVE_REVIEW))
    if review["status"] != "reviewed" or review["accepted"] != 192 or review["held"]:
        raise ValueError("Positive review is not complete")
    if negative["status"] != "frozen" or len(negative["entries"]) != 48:
        raise ValueError("Isolated hard-negative ledger is not frozen and complete")

    planned_positive = {
        (row["recording_group"], row["pose_id"]): row
        for row in planned["entries"]
        if not row["recording_group"].startswith("hard-negative")
    }
    reviewed_positive = {(row["run_id"], row["view_id"]): row for row in review["frames"]}
    if planned_positive.keys() != reviewed_positive.keys():
        missing = planned_positive.keys() ^ reviewed_positive.keys()
        raise ValueError(f"Planned/reviewed positive membership differs: {sorted(missing)[:3]}")

    entries = []
    for key in sorted(planned_positive):
        entry = dict(planned_positive[key])
        decision = reviewed_positive[key]
        entry.update(
            admission_status="admitted",
            capture_status="captured",
            image_path=decision["image_path"],
            image_sha256=decision["image_sha256"],
            pixel_sha256=decision["pixel_sha256"],
            label_sha256=decision["label_sha256"],
            instance_present=True,
            visibility_status="visible_reviewed",
            truncation_status="not_truncated_reviewed",
            review_decision="accepted",
            review_nature=decision["review_nature"],
            review_reason=decision["reason"],
            reviewed_at="2026-09-07T00:00:00+08:00",
            exact_duplicate_of=None,
            pixel_duplicate_of=None,
        )
        entries.append(entry)
    negative_frames = {(row["variant"], row["view_id"]): row for row in negative_review["frames"]}
    for row in negative["entries"]:
        entry = dict(row)
        reviewed = negative_frames[(entry["lighting_id"], entry["pose_id"])]
        entry["image_path"] = reviewed["image_path"]
        entry["pixel_sha256"] = pixel_hash(reviewed["image_path"])
        entries.append(entry)

    if len(entries) != 240 or any(row["admission_status"] != "admitted" for row in entries):
        raise ValueError("Unified admission is incomplete")
    if any(row["review_decision"] != "accepted" for row in entries):
        raise ValueError("Unified review decisions are incomplete")
    files = [row["image_sha256"] for row in entries]
    pixels = [row["pixel_sha256"] for row in entries]
    if len(set(files)) != 240 or len(set(pixels)) != 240:
        raise ValueError("Unexpected exact or decoded-pixel duplicate inside unified dataset")

    reference_files, reference_pixels, reference_inputs = load_fingerprints()
    file_overlap = sorted(set(files) & reference_files)
    pixel_overlap = sorted(set(pixels) & reference_pixels)
    if file_overlap or pixel_overlap:
        raise ValueError("Candidate members overlap protected/reference fingerprints")

    counts = Counter(row["recording_group"] for row in entries)
    subset_counts = Counter(
        "hard_negative" if row["equipment_instance_id"] is None else
        "regular_positive" if row["recording_group"] == "regular-positive" else
        "appearance_lighting_positive"
        for row in entries
    )
    ledger = {
        "schema_version": 2,
        "dataset_version": "visual-augmentation-240-v2",
        "status": "frozen",
        "supersedes": "visual-augmentation-240-v1",
        "supersession_reason": "The v1 hard-negative subset was replaced by isolated no-target scenes; historical files remain unchanged.",
        "counts": dict(sorted(subset_counts.items())) | {"total": len(entries)},
        "independent_pose_groups": {
            "appearance_lighting_positive": len({r["pose_id"] for r in entries if r["recording_group"].startswith("appearance-")}),
            "regular_positive": len({r["pose_id"] for r in entries if r["recording_group"] == "regular-positive"}),
            "hard_negative": len({r["pose_id"] for r in entries if r["equipment_instance_id"] is None}),
        },
        "run_counts": dict(sorted(counts.items())),
        "source_identities": {
            "positive_review": review["identity"],
            "isolated_negative_ledger": negative["identity"],
            "isolated_negative_review": negative_review["identity"],
            "planned_v1_ledger": planned["identity"],
        },
        "input_hashes": {
            str(planned_path): file_sha256(planned_path),
            str(review_path): file_sha256(review_path),
            str(NEGATIVE): file_sha256(NEGATIVE),
            str(NEGATIVE_REVIEW): file_sha256(NEGATIVE_REVIEW),
            **reference_inputs,
        },
        "deduplication": {
            "within_dataset_file_overlaps": 0,
            "within_dataset_pixel_overlaps": 0,
            "protected_reference_file_overlaps": 0,
            "protected_reference_pixel_overlaps": 0,
            "designed_lineage_variants_preserved": True,
        },
        "entries": entries,
        "training_admitted": False,
        "promotable": False,
    }
    ledger["identity"] = object_sha256(ledger)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "frozen-intake-ledger.json", ledger)
    return ledger


if __name__ == "__main__":
    result = freeze()
    print(json.dumps({k: result[k] for k in ("dataset_version", "status", "counts", "independent_pose_groups", "deduplication", "identity")}, indent=2))
