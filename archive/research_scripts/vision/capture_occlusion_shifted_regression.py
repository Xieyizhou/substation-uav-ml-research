"""Capture the development-only occlusion-shifted regression plans."""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import object_sha256, write_json
from src.vision.canonical.collect import collect
from scripts.vision.prepare_occlusion_shifted_regression import BASE, prepare


async def main():
    manifest = prepare()
    runs = []
    for run in manifest["runs"]:
        capture_dir = BASE / run["map_id"] / "capture"
        receipt_path = capture_dir / "collection-receipt.json"
        receipt = (json.loads(receipt_path.read_text()) if receipt_path.exists()
                   else await collect(run["plan_path"], capture_dir, mode="calibration"))
        captured = sum(row.get("status") == "captured" for row in receipt.get("views", []))
        rejected = sum(row.get("status") == "rejected" for row in receipt.get("views", []))
        runs.append({
            "map_id": run["map_id"], "plan_path": run["plan_path"],
            "plan_identity": run["plan_identity"], "receipt_path": str(receipt_path),
            "collection_identity": receipt.get("identity"), "status": receipt.get("status"),
            "requested": len(run["selected"]), "captured": captured, "rejected": rejected,
        })
        print(json.dumps(runs[-1], ensure_ascii=False), flush=True)
    result = {
        "status": "complete_pending_review" if all(row["captured"] == row["requested"] for row in runs) else "incomplete",
        "runs": runs, "training_admitted": False, "promotable": False,
    }
    result["identity"] = object_sha256(result)
    write_json(BASE / "capture-progress.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "complete_pending_review" else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
