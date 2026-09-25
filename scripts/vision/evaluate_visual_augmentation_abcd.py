"""Evaluate frozen A/B/C/D weights on paired factors and isolated negatives."""
import json
import statistics
import sys
from pathlib import Path

from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from src.ml.artifacts import file_sha256,object_sha256,write_json
from scripts.vision.analyze_recovery_paired_calibration import iou
from scripts.vision.evaluate_paired_visual_factors import match,checked_rows,summarize
from scripts.vision.prepare_paired_visual_factors import BASE as PAIRED,VARIANTS

BASE=ROOT/"data/research/ml_training_recovery_v1"
TRAINING=BASE/"visual-augmentation-training-v2"
NEGATIVE_REVIEW=BASE/"hard-negative-isolated-v2/semantic-review.json"
OUT=TRAINING/"development-evaluation.json"
SEEDS=(7,17,27)


def predict(model,path):
    with Image.open(path) as image:
        result=model.predict(image.convert("RGB"),imgsz=640,conf=.37,iou=.7,rect=False,device="cpu",verbose=False,agnostic_nms=False,max_det=300)[0]
    return [{"bbox_xyxy":box,"class_name":model.names[int(cls)],"confidence":conf} for box,cls,conf in zip(result.boxes.xyxy.tolist(),result.boxes.cls.tolist(),result.boxes.conf.tolist())]


def weights():
    result={"frozen_v2_11":ROOT/"models/equipment/visual-yolo11n-baseline-v2.11-candidate-package-v1/weights/best.pt"}
    for arm in "ABCD":
        for seed in SEEDS:
            completion=TRAINING/f"arm-{arm}-seed-{seed}/completion.json"
            row=json.loads(completion.read_text())
            if row["status"]!="complete" or not row["exposure_verified"]: raise ValueError("Training grid is incomplete")
            path=Path(row["weights"])
            if file_sha256(path)!=row["weights_sha256"]: raise ValueError("Training weight changed")
            result[f"{arm}_{seed}"]=path
    return result


def paired_truth(rows):
    progress=json.loads((PAIRED/"capture-progress.json").read_text()); receipts={}
    result=[]
    for row in rows:
        variant=row["variant"]
        if variant not in receipts:
            path=next(Path(x["receipt_path"]) for x in progress["runs"] if x["variant"]==variant)
            receipts[variant]=(path,json.loads(path.read_text()))
        captured=next(x for x in receipts[variant][1]["views"] if x["view_id"]==row["view_id"])
        if object_sha256(captured["truth"])!=row["truth_sha256"]: raise ValueError("Paired truth changed")
        result.append((row,captured["truth"]["objects"]))
    return result,{str(path):file_sha256(path) for path,_ in receipts.values()}


def main():
    from ultralytics import YOLO
    review_rows,paired_review_path=checked_rows(); paired,receipt_inputs=paired_truth(review_rows)
    negatives=json.loads(NEGATIVE_REVIEW.read_text())
    if negatives["status"]!="reviewed" or negatives["accepted"]!=48 or negatives["held"]: raise ValueError("Negative review incomplete")
    inputs={str(Path(__file__)):file_sha256(Path(__file__)),str(paired_review_path):file_sha256(paired_review_path),str(NEGATIVE_REVIEW):file_sha256(NEGATIVE_REVIEW),**receipt_inputs}
    results={}
    for name,weight in weights().items():
        inputs[str(weight)]=file_sha256(weight); model=YOLO(str(weight)); scored=[]
        for row,truth in paired:
            predictions=predict(model,row["image_path"]); matches,used_p,_=match(predictions,truth)
            target=row["target_bbox_xyxy"]
            planned=any(p["class_name"]==row["expected_category"] and iou(p["bbox_xyxy"],target)>=.5 for p in predictions)
            scored.append({"pair_id":row["pair_id"],"view_id":row["view_id"],"variant":row["variant"],"category":row["expected_category"],"object_id":row["expected_object_id"],"planned_instance_hit":planned,"truth":truth,"predictions":predictions,"matches":matches,"unmatched_prediction_count":len(predictions)-len(used_p)})
        negative_rows=[]
        for row in negatives["frames"]:
            if row["decision"]!="accepted" or file_sha256(Path(row["image_path"]))!=row["image_sha256"]: raise ValueError("Negative review stale")
            predictions=predict(model,row["image_path"])
            negative_rows.append({"view_id":row["view_id"],"variant":row["variant"],"prediction_count":len(predictions),"predictions":predictions,"frame_has_prediction":bool(predictions)})
        summary={variant:summarize([row for row in scored if row["variant"]==variant]) for variant in VARIANTS}
        negative_summary={"frames":48,"frames_with_predictions":sum(x["frame_has_prediction"] for x in negative_rows),"frame_false_positive_rate":sum(x["frame_has_prediction"] for x in negative_rows)/48,"unmatched_predictions":sum(x["prediction_count"] for x in negative_rows)}
        deltas=[]
        for pair_id in sorted({row["pair_id"] for row in scored}):
            group={row["variant"]:row for row in scored if row["pair_id"]==pair_id}; original=group["original"]
            for variant in VARIANTS[1:]:
                row=group[variant]; deltas.append({"pair_id":pair_id,"category":row["category"],"variant":variant,"planned_hit_delta":int(row["planned_instance_hit"])-int(original["planned_instance_hit"]),"matched_instance_delta":len(row["matches"])-len(original["matches"]),"unmatched_prediction_delta":row["unmatched_prediction_count"]-original["unmatched_prediction_count"]})
        results[name]={"weights_path":str(weight),"weights_sha256":file_sha256(weight),"paired_rows":scored,"paired_summary":summary,"paired_deltas":deltas,"negative_rows":negative_rows,"negative_summary":negative_summary}
    aggregate={}
    for arm in "ABCD":
        aggregate[arm]={}
        for variant in VARIANTS:
            vals=[results[f"{arm}_{seed}"]["paired_summary"][variant] for seed in SEEDS]
            aggregate[arm][variant]={metric:{"mean":statistics.mean(row[metric] for row in vals),"stdev":statistics.stdev(row[metric] for row in vals)} for metric in ("planned_instance_hit_rate","instance_recall","matched_precision","unmatched_predictions")}
        neg=[results[f"{arm}_{seed}"]["negative_summary"] for seed in SEEDS]
        aggregate[arm]["no_target"]={metric:{"mean":statistics.mean(row[metric] for row in neg),"stdev":statistics.stdev(row[metric] for row in neg)} for metric in ("frame_false_positive_rate","unmatched_predictions")}
    report={"status":"development_evaluation_complete","protocol":{"input_size":640,"device":"cpu","confidence":.37,"class_aware_nms_iou":.7,"max_det":300,"match_iou":.5},"results":results,"aggregate":aggregate,"inputs":inputs,"unseen_scene_status":"sealed_not_evaluated","training_admitted":False,"promotable":False,"limits":["Fixed 48-frame paired set is viewed development regression, not blind test.","Only three poses per class; conclusions are directional for this scene.","The isolated no-target set contains 24 poses under two lights; it is development data.","No threshold tuning, protected-label access, model selection on unseen scenes, or promotion."]}
    report["identity"]=object_sha256(report);write_json(OUT,report)
    print(json.dumps({"status":report["status"],"identity":report["identity"],"aggregate":aggregate},indent=2))


if __name__=="__main__":main()
