"""Admit the 72 reviewed bridge candidates to one frozen development comparison only."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.training.hard_example_curator import dhash64
from scripts.vision.verify_pixel_duplicates import rgb_digest

BASE = ROOT / "data/research/ml_training_recovery_v1/visual-bridge-supplement-v2"
POSITIVE = BASE / "frozen-positive-ledger.json"
NEGATIVE = BASE / "negative-redesign-v2/frozen-negative-ledger.json"
COMPLETION = BASE / "supplement-completion.json"
PROTECTED = ROOT / "data/research/ml_training_recovery_v1/reference-group-audit-v1/supplemental-protected-fingerprints.jsonl"


def main():
    output = BASE / "development-admission.json"
    if output.exists():
        result = json.loads(output.read_text())
        for path, digest in result["inputs"].items():
            if file_sha256(Path(path)) != digest:
                raise ValueError(f"Frozen admission input changed: {path}")
        return result
    positive = json.loads(POSITIVE.read_text())
    negative = json.loads(NEGATIVE.read_text())
    completion = json.loads(COMPLETION.read_text())
    if completion["total_frames"] != 72 or positive["frame_count"] != 48 or negative["frame_count"] != 24:
        raise ValueError("Supplement completion membership changed")
    references = [json.loads(line) for line in PROTECTED.read_text().splitlines() if line]
    protected_file = {row["image_sha256"] for row in references}
    protected_pixel = {row["pixel_sha256"] for row in references}
    protected_dhash = [int(row["perceptual_hash"], 16) for row in references]
    entries = []
    for subset, rows in (("bridge_positive", positive["frames"]), ("hard_negative", negative["frames"])):
        for row in rows:
            path = Path(row["image_path"])
            if file_sha256(path) != row["image_sha256"]:
                raise ValueError(f"Candidate image changed: {path}")
            pixel, size = rgb_digest(path.read_bytes())
            perceptual = int(dhash64(path), 16)
            minimum = min((perceptual ^ value).bit_count() for value in protected_dhash)
            reasons = []
            if row["image_sha256"] in protected_file: reasons.append("protected_file_hash")
            if pixel in protected_pixel: reasons.append("protected_pixel_hash")
            if minimum <= 2: reasons.append("protected_near_hash")
            if row["review"]["decision"] != "accepted" or row["duplicate_status"].startswith("hold_"):
                reasons.append("review_or_dedup_incomplete")
            if subset == "bridge_positive":
                if not row["instance_checks"]["category_present"] or not row["instance_checks"]["planned_instance_present"]:
                    reasons.append("planned_instance_gate_failed")
            elif row["truth_object_count"] != 0 or row["review"]["target_exclusion_status"] != "no_target_visible_reviewed":
                reasons.append("negative_truth_or_visual_exclusion_failed")
            entries.append({
                "member_id": f"bridge:{row['view_id']}", "view_id": row["view_id"], "subset": subset,
                "pair_id": row["pair_id"], "derivation_group": row["derivation_group"], "map_id": row["map_id"],
                "image_path": row["image_path"], "image_sha256": row["image_sha256"], "pixel_sha256": pixel,
                "perceptual_hash": f"{perceptual:016x}", "image_size": list(size), "label_sha256": row["label_sha256"],
                "protected_minimum_hamming_distance": minimum, "exclusion_reasons": reasons,
                "development_training_eligible": not reasons, "formal_training_admitted": False,
                "training_admitted": False, "promotable": False,
            })
    if len(entries) != 72 or any(not row["development_training_eligible"] for row in entries):
        raise ValueError("One or more supplement members failed development admission")
    result = {
        "schema_version": 1, "status": "eligible_for_frozen_development_training", "frame_count": 72,
        "subset_counts": dict(Counter(row["subset"] for row in entries)), "protected_reference_count": len(references),
        "protected_overlap": {"file_sha256": 0, "pixel_sha256": 0, "dhash_distance_le_2": 0},
        "entries": entries,
        "scope": "Eligibility applies only to the frozen visual-bridge A/B/C/D development comparison.",
        "inputs": {str(path): file_sha256(path) for path in (POSITIVE, NEGATIVE, COMPLETION, PROTECTED, Path(__file__))},
        "unseen_scene_status": "sealed_not_evaluated", "formal_training_admitted": False,
        "training_admitted": False, "promotable": False,
    }
    result["identity"] = object_sha256(result)
    write_json(output, result)
    return result


if __name__ == "__main__":
    value = main()
    print(json.dumps({k: value[k] for k in ("status", "frame_count", "subset_counts", "protected_reference_count", "protected_overlap", "identity")}, indent=2))
