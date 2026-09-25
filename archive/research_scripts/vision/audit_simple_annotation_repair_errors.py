"""Break down errors on the repaired simple-scene diagnostic set."""
import json
import sys
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.analyze_recovery_paired_calibration import iou


def main():
    from ultralytics import YOLO

    base = ROOT / "data/research/ml_training_recovery_v1"
    repair = base / "simple-annotation-repair-v2"
    receipt_path = repair / "capture/collection-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    weights = {"positive_only": base / "memorization-v1/fit-batchmatched/weights/last.pt", "negative_augmented": base / "negative-ab-v1/fit/weights/last.pt"}
    inputs = {str(receipt_path): file_sha256(receipt_path), str(Path(__file__)): file_sha256(Path(__file__))}
    report_results = {}
    visuals = ROOT / "outputs/research/ml_training_recovery_v1/simple-annotation-repair-errors-v1"
    visuals.mkdir(parents=True, exist_ok=True)
    for name, weight in weights.items():
        model = YOLO(str(weight))
        errors = []
        for row in receipt["views"]:
            path = Path(row["rgb_path"]); inputs[str(path)] = file_sha256(path)
            truth = row["truth"]["objects"]
            with Image.open(path) as image:
                result = model.predict(image.convert("RGB"), imgsz=640, conf=.37, iou=.7, rect=False, device="cpu", verbose=False, agnostic_nms=False, max_det=300)[0]
            preds = [{"bbox_xyxy": box, "class_name": model.names[int(cls)], "confidence": conf} for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())]
            used = set()
            for pred in preds:
                overlap, index = max(((iou(pred["bbox_xyxy"], obj["bbox_xyxy"]), i) for i, obj in enumerate(truth) if i not in used and obj["class_name"] == pred["class_name"]), default=(0, -1))
                if overlap >= .5:
                    used.add(index)
                    continue
                best, best_index = max(((iou(pred["bbox_xyxy"], obj["bbox_xyxy"]), i) for i, obj in enumerate(truth)), default=(0, -1))
                category = "wrong_class_on_target" if best >= .5 else "localization_overlap" if best >= .1 else "background_candidate"
                errors.append({"type": "unmatched_prediction", "category": category, "view_id": row["view_id"], "expected_category": row["expected_category"], "image_path": str(path), "prediction": pred, "best_iou": best, "best_truth_index": best_index})
            for index, obj in enumerate(truth):
                if index not in used:
                    errors.append({"type": "missed_truth", "category": "missed_truth", "view_id": row["view_id"], "expected_category": row["expected_category"], "image_path": str(path), "truth": obj, "truth_index": index})
        counts = Counter(f"{e['type']}:{e['category']}:{e['expected_category']}" for e in errors)
        report_results[name] = {"errors": errors, "counts": dict(counts), "weights_sha256": file_sha256(weight)}
        selected = [e for e in errors if e["type"] == "missed_truth"]
        if selected:
            page = Image.new("RGB", (1280, 420 * min(6, len(selected))), "#202020")
            draw = ImageDraw.Draw(page)
            for i, error in enumerate(selected[:6]):
                with Image.open(error["image_path"]) as image:
                    page.paste(image.resize((640, 360)), ((i % 2) * 640, (i // 2) * 420 + 40))
                draw.text(((i % 2) * 640 + 4, (i // 2) * 420 + 4), f"{error['expected_category']} missed {error['truth']['class_name']}", fill="white")
                box = error["truth"]["bbox_xyxy"]; x, y = (i % 2) * 640, (i // 2) * 420 + 40
                a, b, c, d = box; draw.rectangle((x + a / 3, y + b / 3, x + c / 3, y + d / 3), outline="lime", width=3)
            visual_path = visuals / f"{name}-missed-truth.jpg"; page.save(visual_path, quality=95); inputs[str(visual_path)] = file_sha256(visual_path)
    report = {"status": "simple_annotation_repair_error_audit_complete", "results": report_results, "inputs": inputs, "training_admitted": False, "promotable": False, "limits": ["Errors are development diagnostics; visual categories are candidate explanations until reviewed.", "No threshold selection or model promotion."]}
    report["identity"] = object_sha256(report)
    write_json(repair / "error-audit.json", report)
    print(json.dumps({name: result["counts"] for name, result in report_results.items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
