"""Evaluate uniform and stratified recovery runs on hash-disjoint development sets."""
import json
import sys
import statistics
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.analyze_recovery_paired_calibration import iou


TARGETS = {"transformer", "switchgear", "capacitor_bank", "reactor"}


def score(model, path, truth, expected):
    with Image.open(path) as image:
        result = model.predict(
            image.convert("RGB"), imgsz=640, conf=.37, iou=.7, rect=False,
            device="cpu", verbose=False, agnostic_nms=False, max_det=300,
        )[0]
    preds = [
        {"bbox_xyxy": box, "class_name": model.names[int(cls)], "confidence": conf}
        for box, cls, conf in zip(
            result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
        )
    ]
    used = set()
    hits = 0
    for pred in preds:
        overlap, index = max(
            (
                (iou(pred["bbox_xyxy"], obj["bbox_xyxy"]), i)
                for i, obj in enumerate(truth)
                if i not in used and obj["class_name"] == pred["class_name"]
            ),
            default=(0, -1),
        )
        if overlap >= .5:
            used.add(index)
            hits += 1
    expected_hit = expected in TARGETS and any(
        p["class_name"] == expected
        and max((iou(p["bbox_xyxy"], obj["bbox_xyxy"]) for obj in truth if obj["class_name"] == expected), default=0) >= .5
        for p in preds
    )
    return {
        "predictions": len(preds),
        "matched_hits": hits,
        "truth_count": len(truth),
        "expected_class_hit": expected_hit,
        "high_confidence_predictions": sum(float(p["confidence"]) >= .5 for p in preds),
    }


def load_datasets(base):
    manifest_path = base / "paired-calibration-v1/plan-manifest.json"
    matrix_progress_path = base / "control-matrix-capture-v1/progress.json"
    cross_eval_path = base / "cross-scene-recheck-v1/evaluation.json"
    datasets = {"matrix_controls": [], "cross_scene": []}
    manifest = json.loads(manifest_path.read_text())
    for run in manifest["runs"]:
        if run["mode"] != "full_2d":
            continue
        receipt = json.loads((base / "paired-calibration-v1" / run["name"] / "capture/collection-receipt.json").read_text())
        # The held target set is intentionally excluded: most of it overlaps the
        # original training images. The report records that limitation explicitly.
    matrix_progress = json.loads(matrix_progress_path.read_text())
    for run in matrix_progress["runs"]:
        receipt = json.loads(Path(run["receipt_path"]).read_text())
        for row in receipt["views"]:
            datasets["matrix_controls"].append({
                "path": row["rgb_path"], "truth": row["truth"]["objects"],
                "expected": row["expected_category"], "view_id": row["view_id"],
            })
    cross_eval = json.loads(cross_eval_path.read_text())
    cross_rows = next(iter(cross_eval["results"].values()))["rows"]
    datasets["cross_scene"] = [
        {"path": r["path"], "truth": r["truth"], "expected": r["expected"], "view_id": r["view_id"]}
        for r in cross_rows if not r["prior_diagnostic_pixel_overlap"]
    ]
    assert len(datasets["matrix_controls"]) == 80
    assert len(datasets["cross_scene"]) == 25
    return datasets, {str(manifest_path): file_sha256(manifest_path), str(matrix_progress_path): file_sha256(matrix_progress_path), str(cross_eval_path): file_sha256(cross_eval_path)}


def main():
    from ultralytics import YOLO

    base = ROOT / "data/research/ml_training_recovery_v1"
    protocol_path = base / "stratified-sampling-v1/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    datasets, inputs = load_datasets(base)
    inputs[str(protocol_path)] = file_sha256(protocol_path)
    inputs[str(Path(__file__))] = file_sha256(Path(__file__))

    weights = {
        "positive_only": base / "memorization-v1/fit-batchmatched/weights/last.pt",
        "negative_augmented": base / "negative-ab-v1/fit/weights/last.pt",
    }
    for arm in protocol["arms"]:
        for seed in protocol["seeds"]:
            weights[f"{arm}-{seed}"] = base / "stratified-sampling-v1" / f"{arm}-{seed}" / "weights/last.pt"

    original_training = [
        item["row"]["rgb_path"]
        for item in json.loads((base / "memorization-v1/protocol.json").read_text())["selected"]
    ]
    repaired_receipt = json.loads((base / "simple-annotation-repair-v2/capture/collection-receipt.json").read_text())
    repaired_training = [row["rgb_path"] for row in repaired_receipt["views"]]
    training_hashes = {file_sha256(Path(p)) for p in original_training + repaired_training}
    overlap = {
        group: sum(file_sha256(Path(row["path"])) in training_hashes for row in rows)
        for group, rows in datasets.items()
    }
    inputs.update({str(Path(p)): file_sha256(Path(p)) for rows in datasets.values() for row in rows for p in [row["path"]]})

    results = {}
    for name, weight in weights.items():
        model = YOLO(str(weight))
        groups = {}
        for group, rows in datasets.items():
            scored = []
            for row in rows:
                scored.append({**row, "score": score(model, Path(row["path"]), row["truth"], row["expected"])})
            groups[group] = scored
        results[name] = {"weights_sha256": file_sha256(weight), "groups": groups}

    summary = {}
    for name, result in results.items():
        summary[name] = {}
        for group, rows in result["groups"].items():
            expected = [r for r in rows if r["expected"] in TARGETS]
            summary[name][group] = {
                "frames": len(rows),
                "truth_objects": sum(r["score"]["truth_count"] for r in rows),
                "matched_hits": sum(r["score"]["matched_hits"] for r in rows),
                "predictions": sum(r["score"]["predictions"] for r in rows),
                "expected_class_hits": sum(bool(r["score"]["expected_class_hit"]) for r in expected),
                "expected_class_frames": len(expected),
                "high_confidence_predictions": sum(r["score"]["high_confidence_predictions"] for r in rows),
            }

    paired = {}
    for arm in protocol["arms"]:
        paired[arm] = {}
        for group in datasets:
            rows = [summary[f"{arm}-{seed}"][group] for seed in protocol["seeds"]]
            paired[arm][group] = {
                "seeds": protocol["seeds"],
                "matched_hits_mean": statistics.mean(r["matched_hits"] for r in rows),
                "matched_hits_stdev": statistics.stdev(r["matched_hits"] for r in rows),
                "expected_class_hits_mean": statistics.mean(r["expected_class_hits"] for r in rows),
                "expected_class_hits_stdev": statistics.stdev(r["expected_class_hits"] for r in rows),
            }
    comparison = {}
    for group in datasets:
        uniform = paired["uniform"][group]
        stratified = paired["stratified"][group]
        comparison[group] = {
            "matched_hits_delta_stratified_minus_uniform": stratified["matched_hits_mean"] - uniform["matched_hits_mean"],
            "expected_class_hits_delta_stratified_minus_uniform": stratified["expected_class_hits_mean"] - uniform["expected_class_hits_mean"],
        }
    comparison["sampling_recipe_passes_paired_rule"] = all(
        value["matched_hits_delta_stratified_minus_uniform"] >= 0
        and value["expected_class_hits_delta_stratified_minus_uniform"] >= 0
        for key, value in comparison.items() if key in datasets
    )

    report = {
        "status": "stratified_recheck_complete",
        "protocol_identity": protocol["identity"],
        "results": results,
        "summary": summary,
        "paired_arm_summary": paired,
        "paired_comparison": comparison,
        "dataset_hash_overlap_with_training": overlap,
        "training_image_count": len(training_hashes),
        "inputs": inputs,
        "training_admitted": False,
        "promotable": False,
        "limits": [
            "Development-only recheck on simulated scenes; not protected validation.",
            "Matrix controls and cross-scene rows are hash-disjoint from the 30-image stratified training pool.",
            "The 19-frame held-target set is excluded because 18 frames overlap the original 18-image training set.",
            "No threshold selection or model promotion.",
        ],
    }
    report["identity"] = object_sha256(report)
    out = base / "stratified-sampling-v1/recheck.json"
    write_json(out, report)
    print(json.dumps({"summary": summary, "overlap": overlap}, indent=2))


if __name__ == "__main__":
    main()
