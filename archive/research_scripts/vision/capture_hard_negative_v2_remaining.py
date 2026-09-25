"""Capture the 40 non-pilot members of the frozen hard-negative v2 matrix."""
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
from scripts.vision.prepare_hard_negative_v2 import BASE,prepare

async def main():
    matrix=prepare();pilot=json.loads((BASE/"pilot-v1/semantic-review.json").read_text())
    if pilot.get("status")!="reviewed_pilot_only" or pilot.get("accepted")!=8 or pilot.get("held"):
        raise ValueError("Reviewed eight-frame pilot is required")
    pilot_ids={row["view_id"] for row in pilot["frames"]};results=[]
    for spec in matrix["runs"]:
        source=Path(spec["plan_path"]).parent;target=BASE/"remaining-v1"/"runs"/spec["run_id"]/"plan"
        if not target.exists():
            shutil.copytree(source,target);plan=read_record(target/"plan.json");record={k:copy.deepcopy(v) for k,v in plan.items() if k!="identity"}
            record["calibration_views"]=[row for row in record["calibration_views"] if row["view_id"] not in pilot_ids]
            if len(record["calibration_views"])!=20:raise ValueError("Continuation is not exactly 20 views")
            record["parent_plan_identity"]=plan["identity"];record["excluded_reviewed_pilot_view_ids"]=sorted(pilot_ids)
            (target/"plan.json").unlink();written=write_record(target/"plan.json",record)
        else:written=read_record(target/"plan.json")
        out=target.parent/"capture";receipt=await collect(target/"plan.json",out,mode="calibration")
        results.append({"run_id":spec["run_id"],"continuation_plan_identity":written["identity"],"receipt_path":str(out/"collection-receipt.json"),"receipt_identity":receipt["identity"],"status":receipt["status"],"captured":sum(r.get("status")=="captured" for r in receipt["views"]),"requested":20,"error":receipt.get("error")})
        if receipt["status"]!="complete_pending_review":break
    status="complete_pending_review" if len(results)==2 and all(r["captured"]==20 for r in results) else "incomplete"
    report={"status":status,"matrix_identity":matrix["identity"],"reviewed_pilot_identity":pilot["identity"],"runs":results,"training_admitted":False,"promotable":False};report["identity"]=object_sha256(report);write_json(BASE/"remaining-v1/capture-progress.json",report);print(json.dumps(report,indent=2));return 0 if status=="complete_pending_review" else 2

if __name__=="__main__":raise SystemExit(asyncio.run(main()))
