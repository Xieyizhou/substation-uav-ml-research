"""Audit exact and conservative dHash duplication for the expansion frames."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.training.hard_example_curator import dhash64


BASE = ROOT / "data/research/ml_training_recovery_v1/stratified-expansion-v1"


def hamming(left, right):
    return (left ^ right).bit_count()


def audit():
    expansion = []
    inputs = {str(BASE / "semantic-review.json"): file_sha256(BASE / "semantic-review.json"), str(Path(__file__)): file_sha256(Path(__file__))}
    review = json.loads((BASE / "semantic-review.json").read_text())
    if review.get("accepted") != 36 or review.get("held") != 0:
        raise ValueError("Expansion semantic review is not closed")
    for row in review["frames"]:
        path = Path(row["rgb_path"])
        if file_sha256(path) != row["image_sha256"]:
            raise ValueError(f"Image changed: {path}")
        expansion.append(row)

    prior_by_hash = {}
    for receipt_path in ROOT.joinpath("data/research").rglob("collection-receipt.json"):
        if "protected" in str(receipt_path) or "stratified-expansion-v1" in str(receipt_path):
            continue
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        inputs[str(receipt_path)] = file_sha256(receipt_path)
        for row in receipt.get("views", []):
            if row.get("status") == "captured" and row.get("image_sha256") and row.get("rgb_path"):
                prior_by_hash.setdefault(row["image_sha256"], row)

    prior = []
    for row in prior_by_hash.values():
        path = Path(row["rgb_path"])
        if file_sha256(path) != row["image_sha256"]:
            raise ValueError(f"Historical image changed: {path}")
        prior.append({"view_id": row.get("view_id"), "image_sha256": row["image_sha256"], "perceptual_hash": int(row.get("perceptual_hash") or dhash64(path), 16)})

    seen_hashes = set()
    decisions = []
    for row in sorted(expansion, key=lambda item: item["view_id"]):
        value = int(dhash64(row["rgb_path"]), 16)
        exact_prior = row["image_sha256"] in prior_by_hash
        exact_internal = row["image_sha256"] in seen_hashes
        distances = [(hamming(value, candidate["perceptual_hash"]), candidate) for candidate in prior]
        nearest_distance, nearest = min(distances, key=lambda pair: (pair[0], pair[1]["view_id"] or ""))
        status = "exact_duplicate_prior" if exact_prior else "exact_duplicate_expansion" if exact_internal else "near_similarity_review" if nearest_distance <= 2 else "retained"
        decisions.append({
            **row,
            "perceptual_hash": f"{value:016x}",
            "exact_prior_duplicate": exact_prior,
            "exact_expansion_duplicate": exact_internal,
            "nearest_prior_hamming_distance": nearest_distance,
            "nearest_prior_view_id": nearest["view_id"],
            "status": status,
            "training_admitted": False,
        })
        seen_hashes.add(row["image_sha256"])

    report = {
        "status": "expansion_dedup_audit_complete",
        "frames": len(decisions),
        "prior_unique_captured_hashes": len(prior),
        "status_counts": dict(Counter(row["status"] for row in decisions)),
        "minimum_prior_hamming_distance": min(row["nearest_prior_hamming_distance"] for row in decisions),
        "decisions": decisions,
        "inputs": inputs,
        "training_admitted": False,
        "promotable": False,
        "limits": [
            "Historical scope is limited to non-protected collection receipts available in the workspace.",
            "dHash distance <=2 is a quarantine trigger, not proof of duplicate content.",
            "No frame is released into formal training by this audit.",
        ],
    }
    report["identity"] = object_sha256(report)
    write_json(BASE / "dedup-audit.json", report)
    print(json.dumps({k: report[k] for k in ("status", "frames", "status_counts", "minimum_prior_hamming_distance", "training_admitted", "promotable", "identity")}, indent=2))
    return report


if __name__ == "__main__":
    audit()
