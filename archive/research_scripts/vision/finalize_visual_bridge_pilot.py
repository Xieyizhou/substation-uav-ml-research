"""Finalize only explicit bridge-pilot review and dedup decisions."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256,object_sha256,write_json
from scripts.vision.prepare_visual_bridge_supplement_v2 import BASE,prepare


def finalize():
    matrix=prepare();capture_path=BASE/"pilot-v1/capture-retry-1-progress.json";review_path=BASE/"pilot-v1/review-v1/semantic-review.json";dedup_path=BASE/"pilot-v1/review-v1/dedup-audit.json"
    capture=json.loads(capture_path.read_text());review=json.loads(review_path.read_text());dedup=json.loads(dedup_path.read_text())
    if capture["status"]!="complete_pending_review" or sum(row["captured"] for row in capture["runs"])!=6: raise ValueError("Pilot capture incomplete")
    if review["status"]!="reviewed" or review["accepted"]!=6 or review["held"]: raise ValueError("Pilot review incomplete")
    if dedup["status"]!="passed" or any(row["status"]!="accepted_no_prior_duplicate" for row in dedup["decisions"]): raise ValueError("Pilot dedup incomplete")
    for row in review["frames"]:
        if file_sha256(Path(row["image_path"]))!=row["image_sha256"]: raise ValueError("Reviewed pilot image changed")
    result={"status":"pilot_passed_ready_for_remaining_positive_capture","matrix_identity":matrix["identity"],"accepted_frames":6,
        "accepted_view_ids":sorted(row["view_id"] for row in review["frames"]),"capture_sha256":file_sha256(capture_path),
        "review_identity":review["identity"],"review_sha256":file_sha256(review_path),"dedup_identity":dedup["identity"],"dedup_sha256":file_sha256(dedup_path),
        "negative_redesign_status":"pending_plan_materialization_and_source_validation","unseen_scene_status":"sealed_not_evaluated",
        "training_admitted":False,"promotable":False};result["identity"]=object_sha256(result);write_json(BASE/"pilot-v1/completion.json",result);return result


if __name__=="__main__": print(json.dumps(finalize(),indent=2))
