"""Capture the 12 non-pilot frames after explicit pilot review and dedup."""
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
    review = json.loads((BASE / "pilot-v1/review-v1/semantic-review.json").read_text())
    dedup = json.loads((BASE / "pilot-v1/review-v1/dedup-audit.json").read_text())
    if review["status"] != "reviewed_pilot_only" or review["accepted"] != 12 or review["held"] or dedup["status"] != "passed":
        raise ValueError("Accepted and deduplicated 12-frame pilot is required")
    pilot_ids = {row["view_id"] for row in review["frames"]}
    runs = []
    for spec in matrix["runs"]:
        source = Path(spec["plan_path"]).parent
        target = BASE / "remaining-v1/runs" / spec["run_id"] / "plan"
        if not target.exists():
            shutil.copytree(source, target)
            plan = read_record(target / "plan.json")
            record = {key: copy.deepcopy(value) for key, value in plan.items() if key != "identity"}
            record["calibration_views"] = [row for row in record["calibration_views"] if row["view_id"] not in pilot_ids]
            if len(record["calibration_views"]) != 2:
                raise ValueError("Each continuation run must contain two frames")
            record["parent_plan_identity"] = plan["identity"]
            record["excluded_reviewed_pilot_view_ids"] = sorted(pilot_ids)
            (target / "plan.json").unlink()
            written = write_record(target / "plan.json", record)
        else:
            written = read_record(target / "plan.json")
        output = target.parent / "capture"
        receipt = await collect(target / "plan.json", output, mode="calibration")
        captured = sum(row.get("status") == "captured" for row in receipt["views"])
        runs.append({"run_id": spec["run_id"], "plan_identity": written["identity"], "receipt_path": str(output / "collection-receipt.json"),
                     "receipt_identity": receipt["identity"], "status": receipt["status"], "captured": captured, "requested": 2,
                     "error": receipt.get("error")})
        if receipt["status"] != "complete_pending_review":
            break
    status = "complete_pending_review" if len(runs) == 6 and all(row["captured"] == 2 for row in runs) else "incomplete"
    result = {"status": status, "matrix_identity": matrix["identity"], "pilot_review_identity": review["identity"],
              "pilot_dedup_identity": dedup["identity"], "runs": runs, "training_admitted": False, "promotable": False,
              "unseen_scene_status": "sealed_not_evaluated"}
    result["identity"] = object_sha256(result)
    write_json(BASE / "remaining-v1/capture-progress.json", result)
    print(json.dumps(result, indent=2))
    return 0 if status == "complete_pending_review" else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
