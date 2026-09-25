"""Audit exact and dHash duplication for scale-stratified coverage frames."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.training.hard_example_curator import dhash64


BASE = ROOT / "data/research/ml_training_recovery_v1/scale-stratified-v1"


def main():
    review_path = BASE / "semantic-review.json"
    review = json.loads(review_path.read_text())
    if review.get("accepted") != 36 or review.get("held"):
        raise ValueError("Scale-stratified review is not closed")

    prior = {}
    for receipt_path in ROOT.joinpath("data/research").rglob("collection-receipt.json"):
        if "protected" in str(receipt_path) or "scale-stratified-v1" in str(receipt_path):
            continue
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in receipt.get("views", []):
            if row.get("status") == "captured" and row.get("image_sha256") and row.get("rgb_path"):
                prior.setdefault(row["image_sha256"], row)

    prior_fp = []
    for row in prior.values():
        path = Path(row["rgb_path"])
        if file_sha256(path) != row["image_sha256"]:
            raise ValueError(f"Historical image changed: {path}")
        prior_fp.append((row.get("view_id"), int(row.get("perceptual_hash") or dhash64(path), 16)))

    seen = set()
    decisions = []
    for row in sorted(review["frames"], key=lambda item: item["view_id"]):
        path = Path(row["rgb_path"])
        value = int(dhash64(path), 16)
        nearest = min(((value ^ fp).bit_count(), view_id) for view_id, fp in prior_fp)
        exact_prior = row["image_sha256"] in prior
        exact_internal = row["image_sha256"] in seen
        status = (
            "exact_duplicate_prior" if exact_prior
            else "exact_duplicate_scale_stratified" if exact_internal
            else "near_similarity_review" if nearest[0] <= 2
            else "retained"
        )
        decisions.append(
            {
                **row,
                "perceptual_hash": f"{value:016x}",
                "nearest_prior_hamming_distance": nearest[0],
                "nearest_prior_view_id": nearest[1],
                "exact_prior_duplicate": exact_prior,
                "exact_scale_stratified_duplicate": exact_internal,
                "status": status,
                "training_admitted": False,
            }
        )
        seen.add(row["image_sha256"])

    result = {
        "status": "scale_stratified_dedup_audit_complete",
        "frames": len(decisions),
        "status_counts": dict(Counter(row["status"] for row in decisions)),
        "minimum_prior_hamming_distance": min(row["nearest_prior_hamming_distance"] for row in decisions),
        "decisions": decisions,
        "inputs": {str(review_path): file_sha256(review_path), str(Path(__file__)): file_sha256(Path(__file__))},
        "training_admitted": False,
        "promotable": False,
        "limits": [
            "Non-protected historical receipt scope only.",
            "dHash distance <=2 is a quarantine trigger, not duplicate proof.",
        ],
    }
    result["identity"] = object_sha256(result)
    write_json(BASE / "dedup-audit.json", result)
    print(json.dumps({key: result[key] for key in ("status", "frames", "status_counts", "minimum_prior_hamming_distance", "training_admitted", "promotable", "identity")}, indent=2))


if __name__ == "__main__":
    main()
