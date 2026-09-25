"""Audit remaining frames against history, pilot, and unrelated batch peers."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.training.hard_example_curator import dhash64
from scripts.vision.prepare_visual_bridge_supplement_v2 import BASE


def hamming(a, b):
    return (int(a, 16) ^ int(b, 16)).bit_count()


def main():
    review_path = BASE / "remaining-positive-v1/review-v1/semantic-review.json"
    review = json.loads(review_path.read_text())
    if review["status"] != "reviewed" or review["accepted"] != 42 or review["held"]:
        raise ValueError("Explicit remaining-frame review is incomplete")

    current_paths = {Path(row["image_path"]).resolve() for row in review["frames"]}
    candidates = []
    for receipt_path in ROOT.joinpath("data/research").rglob("collection-receipt.json"):
        if "protected" in str(receipt_path):
            continue
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in receipt.get("views", []):
            path = Path(row.get("rgb_path", "")).resolve()
            if not path.is_file() or path in current_paths:
                continue
            candidates.append(
                {
                    "path": str(path),
                    "image_sha256": row.get("image_sha256") or file_sha256(path),
                    "perceptual_hash": row.get("perceptual_hash") or dhash64(path),
                    "pair_id": None,
                    "scope": "prior_or_pilot",
                }
            )

    current = []
    for row in review["frames"]:
        current.append(
            {
                "path": row["image_path"],
                "image_sha256": row["image_sha256"],
                "perceptual_hash": dhash64(Path(row["image_path"])),
                "pair_id": row["pair_id"],
                "scope": "remaining_batch",
                "view_id": row["view_id"],
            }
        )

    decisions = []
    for row, item in zip(review["frames"], current):
        # Designed siblings share a pair_id and are deliberately excluded from
        # independence checks. All other current rows and all prior rows remain.
        comparison = candidates + [other for other in current if other["pair_id"] != item["pair_id"]]
        exact = [other for other in comparison if other["image_sha256"] == item["image_sha256"]]
        nearest = min(comparison, key=lambda other: hamming(item["perceptual_hash"], other["perceptual_hash"])) if comparison else None
        distance = hamming(item["perceptual_hash"], nearest["perceptual_hash"]) if nearest else None
        status = (
            "hold_exact_duplicate"
            if exact
            else "hold_near_duplicate"
            if distance is not None and distance <= 2
            else "accepted_no_independent_duplicate"
        )
        decisions.append(
            {
                "view_id": row["view_id"],
                "pair_id": row["pair_id"],
                "variant": row["variant"],
                "expected_category": row["expected_category"],
                "image_sha256": item["image_sha256"],
                "perceptual_hash": item["perceptual_hash"],
                "exact_matches": [other["path"] for other in exact],
                "nearest_path": nearest["path"] if nearest else None,
                "nearest_scope": nearest["scope"] if nearest else None,
                "nearest_hamming_distance": distance,
                "status": status,
                "designed_sibling_group": row["pair_id"],
                "training_admitted": False,
                "promotable": False,
            }
        )

    status = "passed" if all(row["status"] == "accepted_no_independent_duplicate" for row in decisions) else "held"
    result = {
        "schema_version": 1,
        "status": status,
        "frames": len(decisions),
        "decisions": decisions,
        "designed_sibling_rule": "Variants sharing pair_id are one lineage group and are excluded only from mutual independence checks.",
        "inputs": {
            str(review_path): file_sha256(review_path),
            str(Path(__file__)): file_sha256(Path(__file__)),
        },
        "unseen_scene_status": "sealed_not_evaluated",
        "training_admitted": False,
        "promotable": False,
    }
    result["identity"] = object_sha256(result)
    write_json(BASE / "remaining-positive-v1/review-v1/dedup-audit.json", result)
    distances = [row["nearest_hamming_distance"] for row in decisions if row["nearest_hamming_distance"] is not None]
    print(json.dumps({
        "status": status,
        "frames": len(decisions),
        "status_counts": dict(Counter(row["status"] for row in decisions)),
        "minimum_independent_hamming_distance": min(distances) if distances else None,
        "identity": result["identity"],
    }, indent=2))


if __name__ == "__main__":
    main()
