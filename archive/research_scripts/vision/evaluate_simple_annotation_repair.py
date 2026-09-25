"""Evaluate diagnostic weights on the repaired simple-scene captures."""
import json
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.analyze_recovery_paired_calibration import iou


def prior_image_hashes(base):
    hashes = set()
    for receipt_path in base.rglob("collection-receipt.json"):
        if "simple-annotation-repair-v2" in str(receipt_path):
            continue
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        hashes.update(v.get("image_sha256") for v in receipt.get("views", []) if v.get("status") == "captured")
    return hashes


def main():
    from ultralytics import YOLO

    base = ROOT / "data/research/ml_training_recovery_v1"
    repair = base / "simple-annotation-repair-v2"
    receipt_path = repair / "capture/collection-receipt.json"
    plan_path = repair / "plan/plan.json"
    review_path = repair / "semantic-review.json"
    receipt = json.loads(receipt_path.read_text())
    review = json.loads(review_path.read_text())
    if review.get("status") != "reviewed" or review.get("held") != 0:
        raise ValueError("Repaired capture is not fully reviewed")
    known = prior_image_hashes(base)
    rows = []
    for row in receipt["views"]:
        image_hash = file_sha256(Path(row["rgb_path"]))
        if image_hash != row["image_sha256"]:
            raise ValueError(f"RGB hash mismatch: {row['view_id']}")
        rows.append({
            "view_id": row["view_id"],
            "path": row["rgb_path"],
            "category": row["expected_category"],
            "truth": row["truth"]["objects"],
            "image_sha256": image_hash,
            "prior_diagnostic_pixel_overlap": image_hash in known,
        })
    weights = {
        "positive_only": base / "memorization-v1/fit-batchmatched/weights/last.pt",
        "negative_augmented": base / "negative-ab-v1/fit/weights/last.pt",
    }
    results = {}
    for name, weight in weights.items():
        model = YOLO(str(weight))
        scored = []
        for row in rows:
            with Image.open(row["path"]) as image:
                result = model.predict(image.convert("RGB"), imgsz=640, conf=.37, iou=.7, rect=False, device="cpu", verbose=False, agnostic_nms=False, max_det=300)[0]
            preds = [{"bbox_xyxy": box, "class_name": model.names[int(cls)], "confidence": conf} for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())]
            truth = row["truth"]
            used = set()
            hits = 0
            for pred in preds:
                overlap, index = max(((iou(pred["bbox_xyxy"], obj["bbox_xyxy"]), i) for i, obj in enumerate(truth) if i not in used and obj["class_name"] == pred["class_name"]), default=(0, -1))
                if overlap >= .5:
                    used.add(index)
                    hits += 1
            expected_present = any(obj["class_name"] == row["category"] for obj in truth)
            expected_hit = expected_present and any(pred["class_name"] == row["category"] and max((iou(pred["bbox_xyxy"], obj["bbox_xyxy"]) for obj in truth if obj["class_name"] == row["category"]), default=0) >= .5 for pred in preds)
            scored.append({**row, "truth_count": len(truth), "hits": hits, "predictions": len(preds), "expected_class_present": expected_present, "expected_hit": expected_hit, "high_confidence_predictions": sum(float(p["confidence"]) >= .5 for p in preds)})
        results[name] = {"weights_sha256": file_sha256(weight), "rows": scored}
    summary = {}
    for name, result in results.items():
        independent = [r for r in result["rows"] if not r["prior_diagnostic_pixel_overlap"]]
        summary[name] = {
            "frames": len(independent),
            "overlap_frames": sum(r["prior_diagnostic_pixel_overlap"] for r in result["rows"]),
            "truth_objects": sum(r["truth_count"] for r in independent),
            "matched_hits": sum(r["hits"] for r in independent),
            "predictions": sum(r["predictions"] for r in independent),
            "expected_class_hits": sum(r["expected_class_present"] and r["expected_hit"] for r in independent),
            "expected_class_frames": sum(r["expected_class_present"] for r in independent),
            "high_confidence_predictions": sum(r["high_confidence_predictions"] for r in independent),
            "by_category": {category: {"frames": sum(r["category"] == category for r in independent), "expected_class_hits": sum(r["category"] == category and r["expected_hit"] for r in independent), "truth_objects": sum(sum(o["class_name"] == category for o in r["truth"]) for r in independent), "predictions": sum(r["predictions"] for r in independent if r["category"] == category)} for category in ("switchgear", "transformer")},
        }
    report = {
        "status": "simple_annotation_repair_evaluation_complete",
        "results": results,
        "summary": summary,
        "inputs": {str(receipt_path): file_sha256(receipt_path), str(plan_path): file_sha256(plan_path), str(review_path): file_sha256(review_path), str(Path(__file__)): file_sha256(Path(__file__))},
        "confidence": .37,
        "matching_iou": .5,
        "training_admitted": False,
        "promotable": False,
        "limits": ["Development-only corrected annotation-mode check; no protected validation.", "No threshold selection, model promotion or training admission.", "The repaired scene shares the canonical simple world family with prior diagnostics; exact pixel overlaps are excluded from summary."],
    }
    report["identity"] = object_sha256(report)
    write_json(repair / "evaluation.json", report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
