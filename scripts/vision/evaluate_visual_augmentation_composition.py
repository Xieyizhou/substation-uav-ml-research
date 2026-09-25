"""Evaluate E/F families and apply the acceptance policy frozen before training."""
import json
import operator
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from src.ml.artifacts import file_sha256,object_sha256,write_json
from scripts.vision.analyze_recovery_paired_calibration import iou
from scripts.vision.evaluate_paired_visual_factors import match,checked_rows,summarize
from scripts.vision.evaluate_visual_augmentation_abcd import predict,paired_truth
from scripts.vision.prepare_paired_visual_factors import VARIANTS
from scripts.vision.prepare_visual_augmentation_composition import OUT,prepare

BASE=ROOT/"data/research/ml_training_recovery_v1"
NEGATIVE_REVIEW=BASE/"hard-negative-isolated-v2/semantic-review.json"
REPORT=OUT/"development-evaluation.json"
COMPLETION=OUT/"completion.json"
SEEDS=(7,17,27)


def weight_grid(protocol):
    result={};inputs={}
    for arm in "EF":
        for seed in SEEDS:
            completion=OUT/f"arm-{arm}-seed-{seed}/completion.json";row=json.loads(completion.read_text())
            if row["status"]!="complete" or not row["exposure_verified"] or row["protocol_identity"]!=protocol["identity"]:
                raise ValueError("Training grid is incomplete or stale")
            weight=Path(row["weights"])
            if file_sha256(weight)!=row["weights_sha256"]: raise ValueError("Training weight changed")
            result[f"{arm}_{seed}"]=weight;inputs[str(completion)]=file_sha256(completion);inputs[str(weight)]=file_sha256(weight)
    return result,inputs


def metric_value(aggregate,arm,name):
    variant,metric,stat=name.split(".")
    return aggregate[arm][variant][metric][stat]


def apply_policy(aggregate,policy):
    operations={">=":operator.ge,"<=":operator.le}
    decisions={}
    for arm in "EF":
        checks=[]
        for rule in policy["required_all"]:
            actual=metric_value(aggregate,arm,rule["metric"]);passed=operations[rule["op"]](actual,rule["value"])
            checks.append({**rule,"actual":actual,"passed":passed})
        decisions[arm]={"passed":all(row["passed"] for row in checks),"checks":checks}
    selected="E" if decisions["E"]["passed"] else "F" if decisions["F"]["passed"] else None
    return decisions,selected


def main():
    from ultralytics import YOLO
    protocol=prepare();review_rows,paired_review_path=checked_rows();paired,receipt_inputs=paired_truth(review_rows)
    negatives=json.loads(NEGATIVE_REVIEW.read_text())
    if negatives["status"]!="reviewed" or negatives["accepted"]!=48 or negatives["held"]: raise ValueError("Negative review incomplete")
    grid,training_inputs=weight_grid(protocol)
    inputs={str(Path(__file__)):file_sha256(Path(__file__)),str(OUT/"protocol.json"):file_sha256(OUT/"protocol.json"),
        str(paired_review_path):file_sha256(paired_review_path),str(NEGATIVE_REVIEW):file_sha256(NEGATIVE_REVIEW),**receipt_inputs,**training_inputs}
    results={}
    for name,weight in grid.items():
        model=YOLO(str(weight));scored=[]
        for row,truth in paired:
            predictions=predict(model,row["image_path"]);matches,used_p,_=match(predictions,truth);target=row["target_bbox_xyxy"]
            planned=any(p["class_name"]==row["expected_category"] and iou(p["bbox_xyxy"],target)>=.5 for p in predictions)
            scored.append({"pair_id":row["pair_id"],"view_id":row["view_id"],"variant":row["variant"],
                "category":row["expected_category"],"object_id":row["expected_object_id"],"planned_instance_hit":planned,
                "truth":truth,"predictions":predictions,"matches":matches,"unmatched_prediction_count":len(predictions)-len(used_p)})
        negative_rows=[]
        for row in negatives["frames"]:
            if row["decision"]!="accepted" or file_sha256(Path(row["image_path"]))!=row["image_sha256"]: raise ValueError("Negative review stale")
            predictions=predict(model,row["image_path"])
            negative_rows.append({"view_id":row["view_id"],"variant":row["variant"],"prediction_count":len(predictions),
                "predictions":predictions,"frame_has_prediction":bool(predictions)})
        summary={variant:summarize([row for row in scored if row["variant"]==variant]) for variant in VARIANTS}
        negative_summary={"frames":48,"frames_with_predictions":sum(x["frame_has_prediction"] for x in negative_rows),
            "frame_false_positive_rate":sum(x["frame_has_prediction"] for x in negative_rows)/48,
            "unmatched_predictions":sum(x["prediction_count"] for x in negative_rows)}
        results[name]={"weights_path":str(weight),"weights_sha256":file_sha256(weight),"paired_rows":scored,
            "paired_summary":summary,"negative_rows":negative_rows,"negative_summary":negative_summary}
    aggregate={}
    for arm in "EF":
        aggregate[arm]={}
        for variant in VARIANTS:
            vals=[results[f"{arm}_{seed}"]["paired_summary"][variant] for seed in SEEDS];aggregate[arm][variant]={}
            for metric in ("planned_instance_hit_rate","instance_recall","matched_precision","unmatched_predictions"):
                numbers=[row[metric] for row in vals]
                aggregate[arm][variant][metric]={"mean":statistics.mean(numbers),"stdev":statistics.stdev(numbers),"min_seed":min(numbers),"max_seed":max(numbers)}
        neg=[results[f"{arm}_{seed}"]["negative_summary"] for seed in SEEDS];aggregate[arm]["no_target"]={}
        for metric in ("frame_false_positive_rate","unmatched_predictions"):
            numbers=[row[metric] for row in neg]
            aggregate[arm]["no_target"][metric]={"mean":statistics.mean(numbers),"stdev":statistics.stdev(numbers),"min_seed":min(numbers),"max_seed":max(numbers)}
    decisions,selected=apply_policy(aggregate,protocol["acceptance_policy"])
    status="development_candidate_family_selected" if selected else "development_complete_no_candidate"
    report={"status":status,"protocol":{"input_size":640,"device":"cpu","confidence":.37,"class_aware_nms_iou":.7,"max_det":300,"match_iou":.5},
        "results":results,"aggregate":aggregate,"acceptance_policy":protocol["acceptance_policy"],"acceptance_decisions":decisions,
        "candidate_family":selected,"candidate_includes_all_seeds":bool(selected),"inputs":inputs,"unseen_scene_status":"sealed_not_evaluated",
        "training_admitted":False,"promotable":False,"limits":["Fixed 48-frame paired set and isolated negatives are viewed development data.",
        "Family selection uses every frozen rule and all three seeds; no best-seed selection.","No threshold tuning, protected-label access, unseen-scene evaluation, or promotion."]}
    report["identity"]=object_sha256(report);write_json(REPORT,report)
    completion={"status":status,"protocol_identity":protocol["identity"],"development_evaluation_identity":report["identity"],
        "candidate_family":selected,"candidate_includes_all_seeds":bool(selected),"unseen_scene_status":"sealed_not_evaluated",
        "training_admitted":False,"promotable":False}
    completion["identity"]=object_sha256(completion);write_json(COMPLETION,completion)
    print(json.dumps({"status":status,"identity":report["identity"],"aggregate":aggregate,"acceptance_decisions":decisions,"candidate_family":selected},indent=2))


if __name__=="__main__": main()
