"""Diagnostic positive-set augmentation using the repaired annotation-mode captures."""
import json
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json


def main():
    from ultralytics import YOLO

    base = ROOT / "data/research/ml_training_recovery_v1"
    mem = base / "memorization-v1"
    repair = base / "simple-annotation-repair-v2"
    out = base / "repaired-positive-v1"
    out.mkdir(exist_ok=False)
    images = out / "images"; labels = out / "labels"
    images.mkdir(); labels.mkdir()
    names = ["transformer", "switchgear", "capacitor_bank", "reactor"]
    protocol = json.loads((mem / "protocol.json").read_text())
    receipt = json.loads((repair / "capture/collection-receipt.json").read_text())
    selected = []
    inputs = {str(mem / "protocol.json"): file_sha256(mem / "protocol.json"), str(repair / "capture/collection-receipt.json"): file_sha256(repair / "capture/collection-receipt.json"), str(repair / "semantic-review.json"): file_sha256(repair / "semantic-review.json"), str(Path(__file__)): file_sha256(Path(__file__))}

    def add_positive(index, row, source):
        src = Path(row["rgb_path"]); inputs[str(src)] = file_sha256(src)
        stem = f"pos-{index:03}"
        Image.open(src).save(images / f"{stem}.png")
        lines = []
        for obj in row["truth"]["objects"]:
            x0, y0, x1, y1 = obj["bbox_xyxy"]
            class_id = names.index(obj["class_name"])
            lines.append(f"{class_id} {(x0+x1)/3840:.10f} {(y0+y1)/2160:.10f} {(x1-x0)/1920:.10f} {(y1-y0)/1080:.10f}")
        (labels / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        selected.append({"kind": "positive", "path": str(images / f"{stem}.png"), "source": source, "source_view_id": row["view_id"]})

    for i, item in enumerate(protocol["selected"]):
        add_positive(i, item["row"], "memorization-v1")
    offset = len(protocol["selected"])
    for i, row in enumerate(receipt["views"]):
        if row.get("status") != "captured" or row.get("expected_class_present") is not True:
            raise ValueError(f"Repaired row is not a captured observable target: {row.get('view_id')}")
        add_positive(offset + i, row, "simple-annotation-repair-v2")
    dataset = out / "dataset.yaml"
    dataset.write_text(f"path: {out}\ntrain: images\nval: images\nnames: {json.dumps(names)}\n")
    train_protocol = {"purpose": "diagnostic_repaired_positive_augmentation", "selected": selected, "positive_count": len(selected), "source_protocol": str(mem / "protocol.json"), "source_repair": str(repair), "training_set_only": True, "training_admitted": False, "promotable": False, "inputs": inputs}
    train_protocol["identity"] = object_sha256(train_protocol)
    write_json(out / "protocol.json", train_protocol)
    baseline = ROOT / "models/equipment/visual-yolo11n-baseline-v2.11-candidate-package-v1/weights/best.pt"
    if file_sha256(baseline) != "820f882a5d375be0a294555c7168d49b6d68c21adce97f00b45fe7b0ebc54836":
        raise ValueError("Unexpected baseline weights")
    model = YOLO(str(baseline))
    model.train(data=str(dataset), epochs=10, imgsz=640, batch=6, nbs=6, device="cpu", workers=0, optimizer="AdamW", lr0=.001, lrf=1, warmup_epochs=0, warmup_bias_lr=0, seed=7, deterministic=True, patience=0, amp=False, mosaic=0, mixup=0, copy_paste=0, degrees=0, translate=0, scale=0, shear=0, perspective=0, flipud=0, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0, project=str(out), name="fit", plots=False, save=True, val=False)
    completion = {"status": "complete", "weights": str(out / "fit/weights/last.pt"), "weights_sha256": file_sha256(out / "fit/weights/last.pt"), "positive_count": len(selected), "promotable": False, "training_admitted": False}
    write_json(out / "completion.json", completion)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
