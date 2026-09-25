"""Evaluate four frozen weights on reviewed paired visual-factor frames."""
import json
import statistics
import sys
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.analyze_recovery_paired_calibration import iou
from scripts.vision.prepare_paired_visual_factors import BASE, VARIANTS

WEIGHTS={"frozen_v2_11":ROOT/"models/equipment/visual-yolo11n-baseline-v2.11-candidate-package-v1/weights/best.pt",
         **{f"uniform_{seed}":ROOT/f"data/research/ml_training_recovery_v1/expanded-stratified-v1/uniform-{seed}/weights/last.pt" for seed in (7,17,27)}}


def match(predictions,truth):
    candidates=sorted(((iou(p["bbox_xyxy"],t["bbox_xyxy"]),pi,ti) for pi,p in enumerate(predictions) for ti,t in enumerate(truth) if p["class_name"]==t["class_name"]),reverse=True)
    used_p=set(); used_t=set(); matches=[]
    for overlap,pi,ti in candidates:
        if overlap<.5 or pi in used_p or ti in used_t: continue
        used_p.add(pi); used_t.add(ti); matches.append({"prediction_index":pi,"truth_index":ti,"iou":overlap,"class_name":truth[ti]["class_name"]})
    return matches,used_p,used_t


def checked_rows():
    review_path=BASE/"semantic-review.json"; review=json.loads(review_path.read_text())
    if review.get("status")!="reviewed" or review.get("accepted")!=48 or review.get("held") or review.get("complete_pairs")!=12:
        raise ValueError("Review is not a complete 48-frame decision set")
    rows=[]
    for row in review["frames"]:
        if row.get("decision")!="accepted" or file_sha256(row["image_path"])!=row["image_sha256"]:
            raise ValueError("Review decision missing or image hash stale")
        rows.append(row)
    return rows,review_path


def summarize(rows):
    truth=sum(len(r["truth"]) for r in rows); predictions=sum(len(r["predictions"]) for r in rows); matched=sum(len(r["matches"]) for r in rows)
    return {"frames":len(rows),"planned_instance_hit_rate":sum(r["planned_instance_hit"] for r in rows)/len(rows),
            "instance_recall":matched/truth if truth else None,"matched_precision":matched/predictions if predictions else None,
            "unmatched_predictions":sum(r["unmatched_prediction_count"] for r in rows),"truth_instances":truth,"matched_instances":matched,"predictions":predictions}


def main():
    from ultralytics import YOLO
    review_rows,review_path=checked_rows(); inputs={str(review_path):file_sha256(review_path),str(Path(__file__)):file_sha256(Path(__file__))}; results={}
    for model_name,weight in WEIGHTS.items():
        inputs[str(weight)]=file_sha256(weight); model=YOLO(str(weight)); scored=[]
        for row in review_rows:
            with Image.open(row["image_path"]) as image:
                result=model.predict(image.convert("RGB"),imgsz=640,conf=.37,iou=.7,rect=False,device="cpu",verbose=False,agnostic_nms=False,max_det=300)[0]
            predictions=[{"bbox_xyxy":box,"class_name":model.names[int(cls)],"confidence":conf} for box,cls,conf in zip(result.boxes.xyxy.tolist(),result.boxes.cls.tolist(),result.boxes.conf.tolist())]
            # Recover the complete full-image truth from the hash-bound capture row.
            variant=row["variant"]; progress=json.loads((BASE/"capture-progress.json").read_text()); receipt_path=next(Path(r["receipt_path"]) for r in progress["runs"] if r["variant"]==variant)
            inputs[str(receipt_path)]=file_sha256(receipt_path)
            receipt=json.loads(receipt_path.read_text()); captured=next(r for r in receipt["views"] if r["view_id"]==row["view_id"])
            if object_sha256(captured["truth"])!=row["truth_sha256"]:
                raise ValueError("Review truth hash is stale")
            truth=captured["truth"]["objects"]
            matches,used_p,_=match(predictions,truth); target=row["target_bbox_xyxy"]
            planned=any(p["class_name"]==row["expected_category"] and iou(p["bbox_xyxy"],target)>=.5 for p in predictions)
            scored.append({"pair_id":row["pair_id"],"view_id":row["view_id"],"variant":variant,"category":row["expected_category"],"object_id":row["expected_object_id"],
                           "planned_instance_hit":planned,"truth":truth,"predictions":predictions,"matches":matches,
                           "unmatched_prediction_indices":[i for i in range(len(predictions)) if i not in used_p],"unmatched_prediction_count":len(predictions)-len(used_p)})
        results[model_name]={"weights_path":str(weight),"weights_sha256":file_sha256(weight),"rows":scored,
                             "summary":{v:summarize([r for r in scored if r["variant"]==v]) for v in VARIANTS},
                             "by_class":{c:{v:summarize([r for r in scored if r["category"]==c and r["variant"]==v]) for v in VARIANTS} for c in sorted({r["category"] for r in scored})}}
        deltas=[]
        for pair_id in sorted({r["pair_id"] for r in scored}):
            group={r["variant"]:r for r in scored if r["pair_id"]==pair_id}; base=group["original"]
            for variant in VARIANTS[1:]:
                row=group[variant]; deltas.append({"pair_id":pair_id,"category":row["category"],"variant":variant,
                    "planned_hit_delta":int(row["planned_instance_hit"])-int(base["planned_instance_hit"]),
                    "matched_instance_delta":len(row["matches"])-len(base["matches"]),
                    "unmatched_prediction_delta":row["unmatched_prediction_count"]-base["unmatched_prediction_count"]})
        results[model_name]["paired_deltas"]=deltas
    volatility={}
    for variant in VARIANTS:
        values=[results[f"uniform_{seed}"]["summary"][variant]["planned_instance_hit_rate"] for seed in (7,17,27)]
        volatility[variant]={"uniform_seed_mean":statistics.mean(values),"uniform_seed_stdev":statistics.stdev(values),"values":dict(zip((7,17,27),values))}
    report={"status":"complete","protocol":{"input_size":640,"device":"cpu","confidence":.37,"class_aware_nms_iou":.7,"max_det":300,"match_iou":.5},
            "results":results,"development_weight_volatility":volatility,"inputs":inputs,"training_admitted":False,"promotable":False,
            "limits":["No no-target frames: no no-target false-positive rate is reported.","Three poses per class support directional diagnosis in this complex scene only.","No protected labels, threshold tuning, training or promotion."]}
    report["identity"]=object_sha256(report); write_json(BASE/"evaluation.json",report)
    print(json.dumps({name:value["summary"] for name,value in results.items()},indent=2))


if __name__=="__main__": main()
