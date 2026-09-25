"""Build hash-bound contact sheets for the 12-frame negative gate pilot."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.canonical.plan import read_record
from scripts.vision.prepare_visual_bridge_negative_v2 import BASE


def main():
    progress_path = BASE / "pilot-v1/capture-progress.json"
    progress = json.loads(progress_path.read_text())
    if progress["status"] != "complete_pending_review" or sum(row["captured"] for row in progress["runs"]) != 12:
        raise ValueError("Negative gate pilot is incomplete")
    out = BASE / "pilot-v1/review-v1"
    out.mkdir(parents=True, exist_ok=False)
    frames = []
    inputs = {str(progress_path): file_sha256(progress_path)}
    tiles = []
    for run in progress["runs"]:
        receipt_path = Path(run["receipt_path"])
        receipt = read_record(receipt_path)
        plan_path = receipt_path.parent.parent / "plan/plan.json"
        plan = read_record(plan_path)
        planned = {row["view_id"]: row for row in plan["calibration_views"]}
        inputs[str(receipt_path)] = file_sha256(receipt_path)
        inputs[str(plan_path)] = file_sha256(plan_path)
        for view in receipt["views"]:
            if view["status"] != "captured" or view["truth"]["objects"] or view["target_checks"]["observed_instance_labels"]:
                raise ValueError(f"Pilot frame is not valid empty truth: {view['view_id']}")
            row = planned[view["view_id"]]
            image_path = Path(view["rgb_path"])
            image = Image.open(image_path).convert("RGB")
            tile = image.resize((768, 432))
            canvas = Image.new("RGB", (768, 468), (255, 255, 255))
            canvas.paste(tile, (0, 36))
            ImageDraw.Draw(canvas).text((8, 10), f"{run['run_id']} | {row['subject_family']} | {view['view_id'][:12]}", fill=(0, 0, 0))
            tiles.append(canvas)
            frames.append({
                "run_id": run["run_id"], "view_id": view["view_id"], "pair_id": row["pair_id"],
                "derivation_group": row["derivation_group"], "map_id": view["map_id"],
                "subject_family": row["subject_family"], "subject_instance": row["object_id"],
                "lighting_id": row["lighting_id"], "image_path": str(image_path),
                "image_sha256": file_sha256(image_path), "truth_sha256": object_sha256(view["truth"]),
                "truth_object_count": 0, "skew_ms": view["skew_ms"], "decision": "pending",
                "reason": None, "review_nature": "AI-assisted", "training_admitted": False, "promotable": False,
            })
    sheets = []
    for start in range(0, len(tiles), 4):
        sheet = Image.new("RGB", (1536, 936), (230, 230, 230))
        for offset, tile in enumerate(tiles[start:start + 4]):
            sheet.paste(tile, ((offset % 2) * 768, (offset // 2) * 468))
        path = out / f"review-sheet-{start // 4 + 1:02}.png"
        sheet.save(path)
        sheets.append({"path": str(path), "sha256": file_sha256(path), "frame_indexes": [start + 1, min(start + 4, len(tiles))]})
    result = {"schema_version": 1, "status": "pending_explicit_review", "frames": frames, "sheets": sheets,
              "inputs": inputs, "unseen_scene_status": "sealed_not_evaluated", "training_admitted": False, "promotable": False}
    result["identity"] = object_sha256(result)
    write_json(out / "manifest.json", result)
    print(json.dumps({"status": result["status"], "frames": len(frames), "sheets": sheets, "identity": result["identity"]}, indent=2))


if __name__ == "__main__":
    main()
