"""Capture the corrected simple-scene diagnostic plan."""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import object_sha256, write_json
from src.vision.canonical.collect import collect
from scripts.vision.prepare_simple_annotation_repair import BASE, prepare


async def capture():
    manifest = prepare()
    capture_dir = BASE / "capture"
    if capture_dir.exists():
        raise FileExistsError(f"Capture directory already exists: {capture_dir}")
    receipt = await collect(manifest["plan_path"], capture_dir, mode="calibration")
    result = {
        "status": receipt.get("status"),
        "plan_identity": manifest["plan_identity"],
        "collection_identity": receipt.get("identity"),
        "captured": sum(v.get("status") == "captured" for v in receipt.get("views", [])),
        "requested": len(manifest["selected"]),
        "error": receipt.get("error"),
        "training_admitted": False,
        "promotable": False,
    }
    result["identity"] = object_sha256(result)
    write_json(BASE / "capture-progress.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "complete_pending_review" else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(capture()))
