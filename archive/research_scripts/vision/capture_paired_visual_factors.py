"""Sequentially capture all four frozen paired visual-factor variants."""
import asyncio
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.ml.artifacts import object_sha256, write_json
from src.vision.canonical.collect import collect
from scripts.vision.prepare_paired_visual_factors import BASE, VARIANTS, prepare


async def main():
    protocol=prepare(); runs=[]
    for variant in VARIANTS:
        plan=BASE/variant/"plan/plan.json"; output=BASE/variant/"capture"
        if output.exists():
            prior=output/"collection-receipt.json"
            if not prior.exists(): raise ValueError(f"Partial output without receipt: {output}")
            retry=BASE/variant/"capture-resumed"
            receipt=await collect(plan,retry,mode="calibration",resume_from=prior)
            receipt_path=retry/"collection-receipt.json"
        else:
            receipt=await collect(plan,output,mode="calibration")
            receipt_path=output/"collection-receipt.json"
        captured=sum(row.get("status")=="captured" for row in receipt["views"])
        runs.append({"variant":variant,"receipt_path":str(receipt_path),"receipt_identity":receipt["identity"],
                     "status":receipt["status"],"captured":captured,"requested":12})
        if receipt["status"]!="complete_pending_review" or captured!=12: break
    result={"status":"complete_pending_review" if len(runs)==4 and all(r["captured"]==12 for r in runs) else "incomplete",
            "protocol_identity":protocol["identity"],"runs":runs,"training_admitted":False,"promotable":False}
    result["identity"]=object_sha256(result); write_json(BASE/"capture-progress.json",result)
    print(json.dumps(result,indent=2)); return 0 if result["status"]=="complete_pending_review" else 2


if __name__=="__main__": raise SystemExit(asyncio.run(main()))
