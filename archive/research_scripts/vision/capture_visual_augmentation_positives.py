"""Capture the 184 non-pilot positive members of the frozen augmentation matrix."""
import asyncio
import copy
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.ml.artifacts import object_sha256,write_json
from src.vision.canonical.collect import collect
from src.vision.canonical.plan import read_record,write_record
from scripts.vision.prepare_visual_augmentation_batch import BASE,prepare

PILOT_RUNS={"appearance-mat_cool_gray-light_normal","appearance-mat_warm_oxide-light_cool_low"}

async def main():
    matrix=prepare();review=json.loads((BASE/"pilot-v1/positive-semantic-review.json").read_text())
    if review.get("status")!="reviewed_pilot_only" or review.get("accepted")!=8 or review.get("held"):raise ValueError("Reviewed positive pilot is required")
    pilot_ids={row["view_id"] for row in review["frames"]};specs=[r for r in matrix["runs"] if r["subset"] in {"appearance_lighting_positive","regular_positive"}];results=[]
    for spec in specs:
        source=Path(spec["plan_path"]).parent;target=BASE/"positive-capture-v1"/"runs"/spec["run_id"]/"plan"
        if not target.exists():
            shutil.copytree(source,target);plan=read_record(target/"plan.json");record={k:copy.deepcopy(v) for k,v in plan.items() if k!="identity"}
            if spec["run_id"] in PILOT_RUNS:
                record["calibration_views"]=[row for row in record["calibration_views"] if row["view_id"] not in pilot_ids];record["excluded_reviewed_pilot_view_ids"]=sorted(pilot_ids)
            record["parent_plan_identity"]=plan["identity"];(target/"plan.json").unlink();written=write_record(target/"plan.json",record)
        else:written=read_record(target/"plan.json")
        requested=len(written["calibration_views"]);out=target.parent/"capture";receipt=await collect(target/"plan.json",out,mode="calibration")
        results.append({"run_id":spec["run_id"],"subset":spec["subset"],"continuation_plan_identity":written["identity"],"receipt_path":str(out/"collection-receipt.json"),"receipt_identity":receipt["identity"],"status":receipt["status"],"captured":sum(r.get("status")=="captured" for r in receipt["views"]),"requested":requested,"error":receipt.get("error")})
        if receipt["status"]!="complete_pending_review":break
    status="complete_pending_review" if len(results)==len(specs) and sum(r["captured"] for r in results)==184 else "incomplete"
    report={"status":status,"matrix_identity":matrix["identity"],"positive_pilot_review_identity":review["identity"],"captured_nonpilot":sum(r["captured"] for r in results),"expected_nonpilot":184,"runs":results,"training_admitted":False,"promotable":False};report["identity"]=object_sha256(report);write_json(BASE/"positive-capture-v1/capture-progress.json",report);print(json.dumps(report,indent=2));return 0 if status=="complete_pending_review" else 2

if __name__=="__main__":raise SystemExit(asyncio.run(main()))
