"""Capture the 12-frame gate pilot from the canonical-only negative matrix."""
import asyncio
import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import object_sha256, write_json
from src.vision.canonical.collect import collect
from src.vision.canonical.plan import read_record, write_record
from scripts.vision.prepare_visual_bridge_negative_v2 import BASE, prepare


async def main():
    matrix = prepare()
    pilot = BASE / "pilot-v1"
    runs = []
    for spec in matrix["runs"]:
        source = Path(spec["plan_path"]).parent
        target = pilot / "runs" / spec["run_id"] / "plan"
        if not target.exists():
            shutil.copytree(source, target)
            plan = read_record(target / "plan.json")
            record = {key: copy.deepcopy(value) for key, value in plan.items() if key != "identity"}
            chosen = []
            for family in sorted({row["subject_family"] for row in record["calibration_views"]}):
                chosen.append(next(row for row in record["calibration_views"] if row["subject_family"] == family))
            record["calibration_views"] = chosen
            record["parent_plan_identity"] = plan["identity"]
            (target / "plan.json").unlink()
            written = write_record(target / "plan.json", record)
        else:
            written = read_record(target / "plan.json")
        output = target.parent / "capture"
        receipt = await collect(target / "plan.json", output, mode="calibration")
        captured = sum(row.get("status") == "captured" for row in receipt["views"])
        runs.append({
            "run_id": spec["run_id"], "plan_identity": written["identity"],
            "receipt_path": str(output / "collection-receipt.json"), "receipt_identity": receipt["identity"],
            "status": receipt["status"], "captured": captured, "requested": 2, "error": receipt.get("error"),
        })
        if receipt["status"] != "complete_pending_review":
            break
    status = "complete_pending_review" if len(runs) == 6 and all(row["captured"] == 2 for row in runs) else "incomplete"
    result = {"status": status, "matrix_identity": matrix["identity"], "runs": runs,
              "training_admitted": False, "promotable": False, "unseen_scene_status": "sealed_not_evaluated"}
    result["identity"] = object_sha256(result)
    write_json(pilot / "capture-progress.json", result)
    print(json.dumps(result, indent=2))
    return 0 if status == "complete_pending_review" else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
