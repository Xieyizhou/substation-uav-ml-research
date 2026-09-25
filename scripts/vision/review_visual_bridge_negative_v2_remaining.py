"""Record explicit AI-assisted decisions for the remaining negative frames."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.prepare_visual_bridge_negative_v2 import BASE

SHEET_HASHES = (
    "b4d4d5d8d6f6942c44c3b30312e69fa0e791db675db336bf0594fc13a8c50bdf",
    "cc4d024c875352f05a09a39a84421246cbf5107ccafaf7cbf9399663e5a309e7",
    "eaa7a4050db516d9bf9774a81b45436f73e2249836e4973af4d9ad85d669434b",
)


def main():
    manifest_path = BASE / "remaining-v1/review-v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "pending_explicit_review" or len(manifest["frames"]) != 12:
        raise ValueError("Unexpected remaining-negative review manifest")
    if tuple(row["sha256"] for row in manifest["sheets"]) != SHEET_HASHES:
        raise ValueError("Contact sheets differ from inspected evidence")
    for sheet in manifest["sheets"]:
        if file_sha256(Path(sheet["path"])) != sheet["sha256"]:
            raise ValueError("Inspected contact sheet changed")
    frames = []
    for row in manifest["frames"]:
        if file_sha256(Path(row["image_path"])) != row["image_sha256"] or row["truth_object_count"] != 0:
            raise ValueError("Frame or empty truth changed after sheet construction")
        detail = "; foreground ordinary infrastructure does not obscure the planned subject" if row["subject_family"] == "complex_control_building" else ""
        frames.append({**row, "decision": "accepted",
                       "reason": f"Hash-bound full frame inspected: {row['subject_family']} is visible{detail}, image quality is usable, no four-class target equipment is visible, and synchronized full_2d truth is empty.",
                       "image_quality": "usable_reviewed", "target_exclusion_status": "no_target_visible_reviewed",
                       "reviewer": "codex_visual_inspection_2026-09-07", "reviewed_at": "2026-09-07T13:35:00+08:00"})
    result = {"schema_version": 1, "status": "reviewed", "accepted": 12, "held": 0, "review_nature": "AI-assisted",
              "frames": frames, "inputs": {str(manifest_path): file_sha256(manifest_path), str(Path(__file__)): file_sha256(Path(__file__))},
              "unseen_scene_status": "sealed_not_evaluated", "training_admitted": False, "promotable": False}
    result["identity"] = object_sha256(result)
    write_json(BASE / "remaining-v1/review-v1/semantic-review.json", result)
    print(json.dumps({"status": result["status"], "accepted": 12, "held": 0, "identity": result["identity"]}, indent=2))


if __name__ == "__main__":
    main()
