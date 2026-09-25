"""Capture the frozen stratified coverage-expansion plans."""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import object_sha256, write_json
from src.vision.canonical.collect import collect
from scripts.vision.prepare_stratified_expansion import BASE, prepare


async def capture():
    manifest = prepare()
    runs = []
    for run in manifest["runs"]:
        map_dir = BASE / run["map_id"]
        capture_dir = map_dir / "capture"
        receipt_path = capture_dir / "collection-receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
        else:
            receipt = await collect(run["plan_path"], capture_dir, mode="calibration")
        captured = sum(row.get("status") == "captured" for row in receipt.get("views", []))
        runs.append({
            "map_id": run["map_id"],
            "plan_path": run["plan_path"],
            "plan_identity": run["plan_identity"],
            "receipt_path": str(receipt_path),
            "collection_identity": receipt.get("identity"),
            "status": receipt.get("status"),
            "requested": len(run["selected"]),
            "captured": captured,
        })
        print(json.dumps(runs[-1], ensure_ascii=False), flush=True)
    result = {
        "status": "complete_pending_review" if all(r["captured"] == r["requested"] for r in runs) else "incomplete",
        "runs": runs,
        "training_admitted": False,
        "promotable": False,
    }
    result["identity"] = object_sha256(result)
    write_json(BASE / "capture-progress.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "complete_pending_review" else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(capture()))
