"""Record the explicit AI-assisted visual decisions for the seven reviewed sheets."""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.prepare_visual_bridge_supplement_v2 import BASE


# These are the exact seven sheets inspected in this review session. Binding the
# decision to their hashes prevents a rebuilt or edited sheet from inheriting it.
REVIEWED_SHEET_HASHES = (
    "fba99d8542137f9434a9dbead4475d164737e3d34fbccba8a4ce254f4441fed0",
    "3fa81e844d2b3b0b034225e25e9ae3f3978363c19190f3156c594ad51dc9bc05",
    "7b53733b1473727ad0496ee76f2f6efc77ad1777d29909e12b4beebf3a19e8e3",
    "835dde12dd01f686c3dea685bf898dbfb9c7c5391ded7b028e38689023c5961e",
    "92e9ef0e09e0106274a4ddf0f35ebf16d6e49a4b887409a8aba8cf53aae98329",
    "ddf94b96a0db79e87d43c61e01c7a0db5a04d38172a78e82b5ba5ea14ed01f62",
    "1eeb7990a3af367e0d7e0afe04ba3d5f474b7b74f45a24387feb0ba480ff7ecb",
)


def reason_for(row):
    subject = "capacitor bank" if row["expected_category"] == "capacitor_bank" else "reactor"
    condition = {
        "original": "original appearance",
        "neutral_bridge": "neutral material",
        "background_bridge": "shifted background",
    }[row["variant"]]
    return (
        f"Reviewed on the hash-bound contact sheet: planned {subject} is fully visible at far scale "
        f"under {condition}; the image is usable and the target box follows the complete device body."
    )


def main():
    manifest_path = BASE / "remaining-positive-v1/review-v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "pending_explicit_review" or len(manifest["frames"]) != 42:
        raise ValueError("Remaining review manifest is not the inspected 42-frame batch")
    if tuple(row["sha256"] for row in manifest["sheets"]) != REVIEWED_SHEET_HASHES:
        raise ValueError("Review-sheet membership or order differs from the inspected evidence")
    for sheet in manifest["sheets"]:
        if file_sha256(Path(sheet["path"])) != sheet["sha256"]:
            raise ValueError("A reviewed contact sheet changed after inspection")

    frames = []
    for row in manifest["frames"]:
        for key in ("image", "overlay", "crop"):
            if file_sha256(Path(row[f"{key}_path"])) != row[f"{key}_sha256"]:
                raise ValueError(f"Review evidence changed for {row['view_id']}")
        frames.append(
            {
                **row,
                "decision": "accepted",
                "reason": reason_for(row),
                "image_quality": "usable_reviewed",
                "visibility_status": "visible_reviewed",
                "truncation_status": "not_truncated_reviewed",
                "reviewer": "codex_visual_inspection_2026-09-07",
                "reviewed_at": "2026-09-07T12:10:00+08:00",
            }
        )

    paired_geometry = {}
    grouped = defaultdict(list)
    for row in frames:
        grouped[row["pair_id"]].append(row)
    for pair_id, rows in grouped.items():
        if len(rows) != 3 or {row["variant"] for row in rows} != {
            "original", "neutral_bridge", "background_bridge"
        }:
            raise ValueError(f"Incomplete reviewed pair {pair_id}")
        reference = rows[0]["target_bbox_xyxy"]
        delta = max(abs(a - b) for row in rows for a, b in zip(row["target_bbox_xyxy"], reference))
        if delta > 1:
            raise ValueError(f"Target geometry changed by {delta}px in {pair_id}")
        paired_geometry[pair_id] = {
            "variants": sorted(row["variant"] for row in rows),
            "maximum_target_bbox_delta_px": delta,
        }

    result = {
        "schema_version": 1,
        "status": "reviewed",
        "review_nature": "AI-assisted",
        "accepted": 42,
        "held": 0,
        "frames": frames,
        "paired_geometry": paired_geometry,
        "maximum_target_bbox_delta_px": max(
            row["maximum_target_bbox_delta_px"] for row in paired_geometry.values()
        ),
        "inputs": {
            str(manifest_path): file_sha256(manifest_path),
            str(Path(__file__)): file_sha256(Path(__file__)),
        },
        "unseen_scene_status": "sealed_not_evaluated",
        "training_admitted": False,
        "promotable": False,
    }
    result["identity"] = object_sha256(result)
    write_json(BASE / "remaining-positive-v1/review-v1/semantic-review.json", result)
    print(json.dumps({k: result[k] for k in ("status", "accepted", "held", "maximum_target_bbox_delta_px", "identity")}, indent=2))


if __name__ == "__main__":
    main()
