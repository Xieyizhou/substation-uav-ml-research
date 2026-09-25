"""Capture the 42 remaining positive bridge frames after pilot admission."""
import asyncio
import copy
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import object_sha256,write_json
from src.vision.canonical.collect import collect
from src.vision.canonical.plan import read_record,write_record
from scripts.vision.prepare_visual_bridge_supplement_v2 import BASE,prepare


async def main():
    matrix=prepare();pilot=json.loads((BASE/"pilot-v1/completion.json").read_text())
    if pilot["status"]!="pilot_passed_ready_for_remaining_positive_capture" or pilot["matrix_identity"]!=matrix["identity"]:
        raise ValueError("Admitted pilot is required")
    accepted=set(pilot["accepted_view_ids"]);runs=[]
    for spec in matrix["positive"]["runs"]:
        source=Path(spec["plan_path"]).parent;target=BASE/"remaining-positive-v1"/"runs"/spec["variant"]/"plan"
        if not target.exists():
            shutil.copytree(source,target);plan=read_record(target/"plan.json");record={k:copy.deepcopy(value) for k,value in plan.items() if k!="identity"}
            record["calibration_views"]=[row for row in record["calibration_views"] if row["view_id"] not in accepted]
            if len(record["calibration_views"])!=14: raise ValueError("Continuation plan must contain 14 views")
            record["parent_plan_identity"]=plan["identity"];record["excluded_reviewed_pilot_view_ids"]=sorted(accepted&{row["view_id"] for row in plan["calibration_views"]})
            (target/"plan.json").unlink();plan=write_record(target/"plan.json",record)
        else: plan=read_record(target/"plan.json")
        out=target.parent/"capture";receipt=await collect(target/"plan.json",out,mode="calibration")
        runs.append({"variant":spec["variant"],"continuation_plan_identity":plan["identity"],"receipt_path":str(out/"collection-receipt.json"),
            "receipt_identity":receipt["identity"],"status":receipt["status"],"captured":sum(row.get("status")=="captured" for row in receipt["views"]),
            "requested":14,"error":receipt.get("error")})
        if receipt["status"]!="complete_pending_review": break
    status="complete_pending_review" if len(runs)==3 and all(row["captured"]==14 for row in runs) else "incomplete"
    result={"status":status,"matrix_identity":matrix["identity"],"pilot_completion_identity":pilot["identity"],"captured":sum(row["captured"] for row in runs),
        "expected":42,"runs":runs,"unseen_scene_status":"sealed_not_evaluated","training_admitted":False,"promotable":False}
    result["identity"]=object_sha256(result);write_json(BASE/"remaining-positive-v1/capture-progress.json",result);print(json.dumps(result,indent=2));return 0 if status=="complete_pending_review" else 2


if __name__=="__main__": raise SystemExit(asyncio.run(main()))
