"""Record explicit AI-assisted decisions for the inspected negative pilot."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.prepare_visual_bridge_negative_v2 import BASE

SHEET_HASHES = (
    "4a8b677427b151f2b8db555ca2755e19696282b8b39a669367f992b17cbf2f67",
    "478ccf872fdf9da3167509e281dc26fab9fff08fc2a147edd28cf79f5473a78b",
    "9ce102764e216ec86703c72028df5ca4090326ccd713fa04136c005944e50ff6",
)


def main():
    manifest_path = BASE / "pilot-v1/review-v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "pending_explicit_review" or len(manifest["frames"]) != 12:
        raise ValueError("Unexpected negative-pilot review manifest")
    if tuple(row["sha256"] for row in manifest["sheets"]) != SHEET_HASHES:
        raise ValueError("Contact sheets differ from inspected evidence")
    for sheet in manifest["sheets"]:
        if file_sha256(Path(sheet["path"])) != sheet["sha256"]:
            raise ValueError("Inspected contact sheet changed")
    frames = []
    for row in manifest["frames"]:
        if file_sha256(Path(row["image_path"])) != row["image_sha256"] or row["truth_object_count"] != 0:
            raise ValueError("Frame or empty truth changed after sheet construction")
        frames.append({
            **row,
            "decision": "accepted",
            "reason": f"Hash-bound full frame inspected: {row['subject_family']} is clearly visible, image quality is usable, and no four-class target equipment is visible; synchronized full_2d truth is empty.",
            "image_quality": "usable_reviewed",
            "target_exclusion_status": "no_target_visible_reviewed",
            "reviewer": "codex_visual_inspection_2026-09-07",
            "reviewed_at": "2026-09-07T13:10:00+08:00",
        })
    result = {"schema_version": 1, "status": "reviewed_pilot_only", "accepted": 12, "held": 0,
              "review_nature": "AI-assisted", "frames": frames,
              "inputs": {str(manifest_path): file_sha256(manifest_path), str(Path(__file__)): file_sha256(Path(__file__))},
              "unseen_scene_status": "sealed_not_evaluated", "training_admitted": False, "promotable": False}
    result["identity"] = object_sha256(result)
    write_json(BASE / "pilot-v1/review-v1/semantic-review.json", result)
    print(json.dumps({"status": result["status"], "accepted": 12, "held": 0, "identity": result["identity"]}, indent=2))


if __name__ == "__main__":
    main()
