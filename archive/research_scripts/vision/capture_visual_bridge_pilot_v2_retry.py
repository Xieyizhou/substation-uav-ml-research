"""Retry the corrected bridge pilot after a sandbox transport failure."""
import asyncio
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from src.vision.canonical.collect import collect
from scripts.vision.prepare_visual_bridge_supplement_v2 import BASE,prepare


async def main():
    matrix=prepare();prior_path=BASE/"pilot-v1/capture-progress.json";prior=json.loads(prior_path.read_text())
    if prior["status"]!="incomplete" or prior["runs"][0]["captured"]!=0 or "Required topics not discovered" not in prior["runs"][0]["error"]:
        raise ValueError("Retry is only valid for the recorded zero-frame transport startup failure")
    runs=[]
    for spec in matrix["positive_pilot"]["runs"]:
        out=Path(spec["plan_path"]).parent.parent/"capture-retry-1";receipt=await collect(spec["plan_path"],out,mode="calibration")
        runs.append({"variant":spec["variant"],"receipt_path":str(out/"collection-receipt.json"),"receipt_identity":receipt["identity"],
            "status":receipt["status"],"captured":sum(row.get("status")=="captured" for row in receipt["views"]),"requested":2,"error":receipt.get("error")})
        if receipt["status"]!="complete_pending_review": break
    status="complete_pending_review" if len(runs)==3 and all(row["captured"]==2 for row in runs) else "incomplete"
    result={"status":status,"matrix_identity":matrix["identity"],"retry_of":str(prior_path),"retry_of_sha256":file_sha256(prior_path),
        "retry_reason":"sandbox denied Gazebo Transport socket binding; prior attempt captured zero frames","runs":runs,
        "unseen_scene_status":"sealed_not_evaluated","training_admitted":False,"promotable":False};result["identity"]=object_sha256(result)
    write_json(BASE/"pilot-v1/capture-retry-1-progress.json",result);print(json.dumps(result,indent=2));return 0 if status=="complete_pending_review" else 2


if __name__=="__main__": raise SystemExit(asyncio.run(main()))
