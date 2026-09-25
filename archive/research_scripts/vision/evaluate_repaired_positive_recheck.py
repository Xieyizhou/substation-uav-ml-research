"""Compare the repaired-positive diagnostic weight on existing development rechecks."""
import json
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.analyze_recovery_paired_calibration import iou


def score(model, path, truth, expected):
    with Image.open(path) as image:
        result = model.predict(image.convert("RGB"), imgsz=640, conf=.37, iou=.7, rect=False, device="cpu", verbose=False, agnostic_nms=False, max_det=300)[0]
    preds = [{"bbox_xyxy": box, "class_name": model.names[int(cls)], "confidence": conf} for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())]
    used = set(); hits = 0
    for pred in preds:
        overlap, index = max(((iou(pred["bbox_xyxy"], obj["bbox_xyxy"]), i) for i, obj in enumerate(truth) if i not in used and obj["class_name"] == pred["class_name"]), default=(0, -1))
        if overlap >= .5:
            used.add(index); hits += 1
    expected_hit = expected in ("transformer", "switchgear", "capacitor_bank", "reactor") and any(p["class_name"] == expected and max((iou(p["bbox_xyxy"], obj["bbox_xyxy"]) for obj in truth if obj["class_name"] == expected), default=0) >= .5 for p in preds)
    return {"predictions": len(preds), "matched_hits": hits, "truth_count": len(truth), "expected_class_hit": expected_hit, "high_confidence_predictions": sum(float(p["confidence"]) >= .5 for p in preds)}


def main():
    from ultralytics import YOLO
    base = ROOT / "data/research/ml_training_recovery_v1"
    manifest = base / "paired-calibration-v1/plan-manifest.json"
    matrix_progress = base / "control-matrix-capture-v1/progress.json"
    cross_eval = json.loads((base / "cross-scene-recheck-v1/evaluation.json").read_text())
    datasets = {"held_targets": [], "matrix_controls": [], "cross_scene": []}
    m = json.loads(manifest.read_text())
    for run in m["runs"]:
        if run["mode"] != "full_2d": continue
        receipt = json.loads((base / "paired-calibration-v1" / run["name"] / "capture/collection-receipt.json").read_text())
        for row in receipt["views"]:
            if row["view_id"] in run["held_target_view_ids"]:
                datasets["held_targets"].append({"path": row["rgb_path"], "truth": row["truth"]["objects"], "expected": row["expected_category"], "view_id": row["view_id"]})
    for run in json.loads(matrix_progress.read_text())["runs"]:
        receipt = json.loads(Path(run["receipt_path"]).read_text())
        for row in receipt["views"]:
            datasets["matrix_controls"].append({"path": row["rgb_path"], "truth": row["truth"]["objects"], "expected": row["expected_category"], "view_id": row["view_id"]})
    cross_rows = next(iter(cross_eval["results"].values()))["rows"]
    datasets["cross_scene"] = [{"path": r["path"], "truth": r["truth"], "expected": r["expected"], "view_id": r["view_id"]} for r in cross_rows if not r["prior_diagnostic_pixel_overlap"]]
    assert len(datasets["held_targets"]) == 19 and len(datasets["matrix_controls"]) == 80 and len(datasets["cross_scene"]) == 25
    weights = {
        "positive_only": base / "memorization-v1/fit-batchmatched/weights/last.pt",
        "negative_augmented": base / "negative-ab-v1/fit/weights/last.pt",
        "repaired_positive": base / "repaired-positive-v1/fit/weights/last.pt",
    }
    inputs = {str(manifest): file_sha256(manifest), str(matrix_progress): file_sha256(matrix_progress), str(base / "cross-scene-recheck-v1/evaluation.json"): file_sha256(base / "cross-scene-recheck-v1/evaluation.json"), str(Path(__file__)): file_sha256(Path(__file__))}
    results = {}
    for name, weight in weights.items():
        model = YOLO(str(weight)); groups = {}
        for group, rows in datasets.items():
            scored = []
            for row in rows:
                p = Path(row["path"]); inputs[str(p)] = file_sha256(p)
                scored.append({**row, "score": score(model, p, row["truth"], row["expected"])})
            groups[group] = scored
        results[name] = {"weights_sha256": file_sha256(weight), "groups": groups}
    summary = {}
    for name, result in results.items():
        summary[name] = {}
        for group, rows in result["groups"].items():
            expected = [r for r in rows if r["expected"] in ("transformer", "switchgear", "capacitor_bank", "reactor")]
            expected_hits = sum(bool(r["score"]["expected_class_hit"]) for r in expected)
            summary[name][group] = {"frames": len(rows), "truth_objects": sum(r["score"]["truth_count"] for r in rows), "matched_hits": sum(r["score"]["matched_hits"] for r in rows), "predictions": sum(r["score"]["predictions"] for r in rows), "expected_class_hits": expected_hits, "expected_class_frames": len(expected), "high_confidence_predictions": sum(r["score"]["high_confidence_predictions"] for r in rows)}
    report = {"status": "repaired_positive_recheck_complete", "results": results, "summary": summary, "inputs": inputs, "training_admitted": False, "promotable": False, "limits": ["Development-only recheck on correlated simulated scenes; not protected validation.", "Repaired-positive weight was trained on the repaired simple captures; this report uses held, control and cross-scene rows only.", "No threshold selection or model promotion."]}
    report["identity"] = object_sha256(report)
    write_json(base / "repaired-positive-v1/independent-recheck.json", report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
