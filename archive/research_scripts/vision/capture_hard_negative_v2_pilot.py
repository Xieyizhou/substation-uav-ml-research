"""Capture four frozen negative poses under both lights (8-frame gate pilot)."""
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
    matrix=prepare();pilot=BASE/"pilot-v1";runs=[]
    for spec in matrix["runs"]:
        source=Path(spec["plan_path"]).parent;target=pilot/"runs"/spec["run_id"]/"plan"
        if not target.exists():
            shutil.copytree(source,target);plan=read_record(target/"plan.json");record={k:copy.deepcopy(v) for k,v in plan.items() if k!="identity"};record["calibration_views"]=record["calibration_views"][:4];record["parent_plan_identity"]=plan["identity"];(target/"plan.json").unlink();write_record(target/"plan.json",record)
        out=target.parent/"capture";receipt=await collect(target/"plan.json",out,mode="calibration")
        runs.append({"run_id":spec["run_id"],"receipt_path":str(out/"collection-receipt.json"),"receipt_identity":receipt["identity"],"status":receipt["status"],"captured":sum(r.get("status")=="captured" for r in receipt["views"]),"requested":4,"error":receipt.get("error")})
        if receipt["status"]!="complete_pending_review":break
    status="complete_pending_review" if len(runs)==2 and all(r["captured"]==4 for r in runs) else "incomplete"
    result={"status":status,"matrix_identity":matrix["identity"],"runs":runs,"training_admitted":False,"promotable":False};result["identity"]=object_sha256(result);write_json(pilot/"capture-progress.json",result);print(json.dumps(result,indent=2));return 0 if status=="complete_pending_review" else 2

if __name__=="__main__":raise SystemExit(asyncio.run(main()))
