"""Capture the corrected six-frame full_2d positive bridge pilot."""
import asyncio
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import object_sha256,write_json
from src.vision.canonical.collect import collect
from scripts.vision.prepare_visual_bridge_supplement_v2 import BASE,prepare


async def main():
    matrix=prepare();runs=[]
    for spec in matrix["positive_pilot"]["runs"]:
        out=Path(spec["plan_path"]).parent.parent/"capture";receipt=await collect(spec["plan_path"],out,mode="calibration")
        runs.append({"variant":spec["variant"],"receipt_path":str(out/"collection-receipt.json"),"receipt_identity":receipt["identity"],
            "status":receipt["status"],"captured":sum(row.get("status")=="captured" for row in receipt["views"]),"requested":2,"error":receipt.get("error")})
        if receipt["status"]!="complete_pending_review": break
    status="complete_pending_review" if len(runs)==3 and all(row["captured"]==2 for row in runs) else "incomplete"
    result={"status":status,"matrix_identity":matrix["identity"],"runs":runs,"unseen_scene_status":"sealed_not_evaluated",
        "training_admitted":False,"promotable":False};result["identity"]=object_sha256(result)
    write_json(BASE/"pilot-v1/capture-progress.json",result);print(json.dumps(result,indent=2));return 0 if status=="complete_pending_review" else 2


if __name__=="__main__": raise SystemExit(asyncio.run(main()))
