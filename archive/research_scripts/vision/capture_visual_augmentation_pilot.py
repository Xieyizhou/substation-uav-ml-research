"""Capture the frozen workflow pilot, stopping on the first rejected run."""
import asyncio
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.ml.artifacts import object_sha256,write_json
from src.vision.canonical.collect import collect
from scripts.vision.prepare_visual_augmentation_pilot import PILOT,prepare

async def main():
    manifest=prepare();results=[]
    for run in manifest["runs"]:
        out=PILOT/"runs"/run["run_id"]/"capture"
        receipt=await collect(run["plan_path"],out,mode="calibration")
        results.append({"run_id":run["run_id"],"receipt_path":str(out/"collection-receipt.json"),"receipt_identity":receipt["identity"],"status":receipt["status"],"captured":sum(r.get("status")=="captured" for r in receipt["views"]),"requested":run["frame_count"],"error":receipt.get("error")})
        if receipt["status"]!="complete_pending_review":break
    status="complete_pending_review" if len(results)==len(manifest["runs"]) and all(r["captured"]==r["requested"] for r in results) else "incomplete"
    report={"status":status,"pilot_manifest_identity":manifest["identity"],"runs":results,"training_admitted":False,"promotable":False};report["identity"]=object_sha256(report);write_json(PILOT/"capture-progress.json",report);print(json.dumps(report,indent=2));return 0 if status=="complete_pending_review" else 2

if __name__=="__main__":raise SystemExit(asyncio.run(main()))
