"""Audit a reviewed negative batch while preserving paired-light lineage."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.training.hard_example_curator import dhash64
from scripts.vision.prepare_visual_bridge_negative_v2 import BASE


def hamming(a, b):
    return (int(a, 16) ^ int(b, 16)).bit_count()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review = json.loads(args.review.read_text())
    if review.get("held") or not review.get("frames") or any(row["decision"] != "accepted" for row in review["frames"]):
        raise ValueError("Explicit review is incomplete")
    current_paths = {Path(row["image_path"]).resolve() for row in review["frames"]}
    prior = []
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
            prior.append({"path": str(path), "image_sha256": row.get("image_sha256") or file_sha256(path),
                          "perceptual_hash": row.get("perceptual_hash") or dhash64(path), "pair_id": None, "scope": "prior"})
    current = [{"path": row["image_path"], "image_sha256": row["image_sha256"],
                "perceptual_hash": dhash64(Path(row["image_path"])), "pair_id": row["pair_id"], "scope": "current"}
               for row in review["frames"]]
    decisions = []
    for row, item in zip(review["frames"], current):
        comparison = prior + [peer for peer in current if peer["pair_id"] != item["pair_id"]]
        exact = [peer for peer in comparison if peer["image_sha256"] == item["image_sha256"]]
        nearest = min(comparison, key=lambda peer: hamming(item["perceptual_hash"], peer["perceptual_hash"])) if comparison else None
        distance = hamming(item["perceptual_hash"], nearest["perceptual_hash"]) if nearest else None
        status = "hold_exact_duplicate" if exact else "hold_near_duplicate" if distance is not None and distance <= 2 else "accepted_no_independent_duplicate"
        decisions.append({"view_id": row["view_id"], "pair_id": row["pair_id"], "subject_family": row["subject_family"],
                          "image_sha256": item["image_sha256"], "perceptual_hash": item["perceptual_hash"],
                          "exact_matches": [peer["path"] for peer in exact], "nearest_path": nearest["path"] if nearest else None,
                          "nearest_hamming_distance": distance, "status": status, "training_admitted": False, "promotable": False})
    status = "passed" if all(row["status"] == "accepted_no_independent_duplicate" for row in decisions) else "held"
    result = {"schema_version": 1, "status": status, "frames": len(decisions), "decisions": decisions,
              "designed_sibling_rule": "The two lighting variants sharing pair_id are one lineage and excluded only from mutual independence checks.",
              "inputs": {str(args.review): file_sha256(args.review), str(Path(__file__)): file_sha256(Path(__file__))},
              "unseen_scene_status": "sealed_not_evaluated", "training_admitted": False, "promotable": False}
    result["identity"] = object_sha256(result)
    write_json(args.output, result)
    print(json.dumps({"status": status, "status_counts": dict(Counter(row["status"] for row in decisions)),
                      "minimum_independent_hamming_distance": min(row["nearest_hamming_distance"] for row in decisions),
                      "identity": result["identity"]}, indent=2))


if __name__ == "__main__":
    main()
