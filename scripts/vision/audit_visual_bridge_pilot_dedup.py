"""Audit exact and perceptual overlap while preserving designed bridge siblings."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from src.vision.training.hard_example_curator import dhash64
from scripts.vision.prepare_visual_bridge_supplement_v2 import BASE


def hamming(a,b): return (int(a,16)^int(b,16)).bit_count()


def main():
    review_path=BASE/"pilot-v1/review-v1/semantic-review.json";review=json.loads(review_path.read_text())
    if review["status"]!="reviewed" or review["accepted"]!=6: raise ValueError("Explicit pilot review is incomplete")
    current_paths={Path(row["image_path"]).resolve() for row in review["frames"]};prior=[]
    for receipt_path in ROOT.joinpath("data/research").rglob("collection-receipt.json"):
        if "protected" in str(receipt_path) or BASE in receipt_path.parents: continue
        try: receipt=json.loads(receipt_path.read_text())
        except (OSError,json.JSONDecodeError): continue
        for row in receipt.get("views",[]):
            path=Path(row.get("rgb_path","")).resolve()
            if not path.is_file() or path in current_paths: continue
            prior.append({"path":str(path),"image_sha256":row.get("image_sha256") or file_sha256(path),
                "perceptual_hash":row.get("perceptual_hash") or dhash64(path)})
    current=[{"path":row["image_path"],"image_sha256":row["image_sha256"],"perceptual_hash":dhash64(Path(row["image_path"])),
        "sibling_group":row["expected_category"]} for row in review["frames"]]
    decisions=[]
    for row,item in zip(review["frames"],current):
        comparison=prior+[peer for peer in current if peer["sibling_group"]!=item["sibling_group"]]
        phash=item["perceptual_hash"];exact=[p for p in comparison if p["image_sha256"]==row["image_sha256"]]
        nearest=min(comparison,key=lambda p:hamming(phash,p["perceptual_hash"])) if comparison else None
        distance=hamming(phash,nearest["perceptual_hash"]) if nearest else None
        status="hold_prior_exact_duplicate" if exact else "hold_prior_near_duplicate" if distance is not None and distance<=2 else "accepted_no_prior_duplicate"
        decisions.append({"view_id":row["view_id"],"variant":row["variant"],"expected_category":row["expected_category"],
            "image_sha256":row["image_sha256"],"perceptual_hash":phash,"prior_exact_matches":[p["path"] for p in exact],
            "nearest_prior_path":nearest["path"] if nearest else None,"nearest_prior_hamming_distance":distance,"status":status,
            "designed_sibling_group":row["expected_category"],"training_admitted":False,"promotable":False})
    status="passed" if all(row["status"]=="accepted_no_prior_duplicate" for row in decisions) else "held"
    result={"schema_version":1,"status":status,"frames":len(decisions),"decisions":decisions,
        "designed_sibling_rule":"The three variants of one bridge pose are retained as one lineage group and are not compared as independent samples.",
        "inputs":{str(review_path):file_sha256(review_path),str(Path(__file__)):file_sha256(Path(__file__))},
        "unseen_scene_status":"sealed_not_evaluated","training_admitted":False,"promotable":False}
    result["identity"]=object_sha256(result);write_json(BASE/"pilot-v1/review-v1/dedup-audit.json",result)
    print(json.dumps({"status":status,"frames":len(decisions),"status_counts":dict(__import__('collections').Counter(row['status'] for row in decisions)),
        "minimum_prior_hamming_distance":min(row["nearest_prior_hamming_distance"] for row in decisions if row["nearest_prior_hamming_distance"] is not None),"identity":result["identity"]},indent=2))


if __name__=="__main__": main()
