"""Train a larger development pool after scale-stratified semantic gates."""
import argparse
from collections import Counter
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PIL import Image
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.canonical.plan import read_record
from src.vision.collection.simulator_labels import instance_simulator_label
from scripts.vision.run_recovery_expanded_experiment import area_bucket, expected_object, source_rows as source_rows_66


BASE = ROOT / "data/research/ml_training_recovery_v1"
OUT = BASE / "scale-expansion-v1"
NAMES = ["transformer", "switchgear", "capacitor_bank", "reactor"]
SEEDS = [7, 17, 27]
EPOCHS, SLOTS, BATCH = 10, 60, 6
WEIGHTS = ROOT / "models/equipment/visual-yolo11n-baseline-v2.11-candidate-package-v1/weights/best.pt"


def distance_bin(distance):
    if distance < 10:
        return "near"
    if distance < 15:
        return "mid"
    return "far"


def scale_rows():
    review_path = BASE / "scale-stratified-v1/semantic-review.json"
    review = read_record(review_path)
    if review["accepted"] != 36 or review["held"]:
        raise ValueError("Scale-stratified semantic review is not closed")
    accepted = {row["view_id"] for row in review["frames"]}
    rows = []
    for map_id in ("simple", "medium", "complex"):
        plan = read_record(BASE / f"scale-stratified-v1/{map_id}/plan/plan.json")
        receipt = read_record(BASE / f"scale-stratified-v1/{map_id}/capture/collection-receipt.json")
        selected = {row["view_id"]: row for row in plan["calibration_views"]}
        for row in receipt["views"]:
            if row["view_id"] not in accepted or row.get("expected_class_present") is not True:
                raise ValueError(f"Scale row not accepted: {row['view_id']}")
            anchor, anchor_status = expected_object(row)
            spec = selected[row["view_id"]]
            rows.append({
                **row,
                "source": "scale_stratified",
                "scale": area_bucket(anchor["bbox_xyxy"]),
                "distance": spec["distance"],
                "distance_bin": distance_bin(spec["distance"]),
                "sampling_anchor_status": anchor_status,
            })
    if len(rows) != 36 or len({row["image_sha256"] for row in rows}) != 36:
        raise ValueError("Expected 36 unique scale-stratified images")
    return rows


def all_rows():
    rows = source_rows_66() + scale_rows()
    if len(rows) != 102 or len({row["image_sha256"] for row in rows}) != 102:
        raise ValueError("Expected 102 unique images in the combined pool")
    return rows


def make_schedule(rows, seed):
    rng = random.Random(seed)
    schedule = []
    for _ in range(EPOCHS):
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        schedule.append(indices[:SLOTS])
    return schedule


def exposure(rows, schedule):
    draws = [index for epoch in schedule for index in epoch]
    return {
        "draws": len(draws),
        "unique_frames": len(set(draws)),
        "by_source": dict(Counter(rows[i]["source"] for i in draws)),
        "by_expected_class": dict(Counter(rows[i]["expected_category"] for i in draws)),
        "by_map": dict(Counter(rows[i]["map_id"] for i in draws)),
        "by_scale": dict(Counter(rows[i]["scale"] for i in draws)),
        "by_distance_bin": dict(Counter(rows[i].get("distance_bin", "historical") for i in draws)),
    }


def write_training_files(rows, inputs):
    (OUT / "images").mkdir(parents=True)
    (OUT / "labels").mkdir()
    for index, row in enumerate(rows):
        image_path = OUT / "images" / f"{index:03}.png"
        source_image = Path(row["rgb_path"])
        if file_sha256(source_image) != row["image_sha256"]:
            raise ValueError(f"Image mismatch: {source_image}")
        with Image.open(source_image) as image:
            if image.size != (1920, 1080):
                raise ValueError(f"Unexpected image size: {source_image}")
            image.convert("RGB").save(image_path)
        label_path = OUT / "labels" / f"{index:03}.txt"
        lines = []
        for obj in row["truth"]["objects"]:
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            if obj["class_name"] not in NAMES or not (0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080):
                raise ValueError(f"Invalid truth object in {row['view_id']}")
            lines.append(f"{NAMES.index(obj['class_name'])} {(x1+x2)/3840:.10f} {(y1+y2)/2160:.10f} {(x2-x1)/1920:.10f} {(y2-y1)/1080:.10f}")
        label_path.write_text("\n".join(lines) + "\n")
        row["training_path"] = str(image_path)
        inputs[str(image_path)] = file_sha256(image_path)
        inputs[str(label_path)] = file_sha256(label_path)
    dataset = OUT / "dataset.yaml"
    dataset.write_text(f"path: {OUT}\ntrain: images\nval: images\nnames: {json.dumps(NAMES)}\n")
    inputs[str(dataset)] = file_sha256(dataset)
    return dataset


def prepare():
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        protocol = read_record(protocol_path)
        for path, digest in protocol["inputs"].items():
            if file_sha256(Path(path)) != digest:
                raise ValueError(f"Frozen input changed: {path}")
        return protocol
    rows = all_rows()
    OUT.mkdir(parents=True, exist_ok=False)
    inputs = {str(path): file_sha256(path) for path in [Path(__file__), WEIGHTS]}
    for source_path in (
        BASE / "expanded-stratified-v1/protocol.json",
        BASE / "simple-annotation-repair-v2/semantic-review.json",
        BASE / "stratified-expansion-v1/semantic-review.json",
        BASE / "scale-stratified-v1/manifest.json",
        BASE / "scale-stratified-v1/semantic-review.json",
        BASE / "scale-stratified-v1/dedup-audit.json",
    ):
        inputs[str(source_path)] = file_sha256(source_path)
    for row in rows:
        inputs[str(Path(row["rgb_path"]))] = file_sha256(Path(row["rgb_path"]))
    dataset = write_training_files(rows, inputs)
    schedules = {str(seed): make_schedule(rows, seed) for seed in SEEDS}
    protocol = {
        "status": "frozen_before_training",
        "rows": rows,
        "inputs": inputs,
        "schedules": schedules,
        "exposures": {seed: exposure(rows, value) for seed, value in schedules.items()},
        "epochs": EPOCHS,
        "slots_per_epoch": SLOTS,
        "batch": BATCH,
        "optimizer_steps_per_run": (EPOCHS * SLOTS) // BATCH,
        "seeds": SEEDS,
        "sampling": "uniform",
        "weights": str(WEIGHTS),
        "dataset": str(dataset),
        "comparison_rule": "Paired seed summaries on scale-stratified coverage, prior hash-disjoint controls and cross-scene sets; no best-seed selection.",
        "training_admitted": False,
        "promotable": False,
        "limits": [
            "The 102-image pool remains simulated development data.",
            "The 100-step budget differs from the earlier 66-image 50-step experiment; this is a deliberate larger-budget follow-up.",
            "No protected labels or validation artifacts enter the pool.",
        ],
    }
    protocol["identity"] = object_sha256(protocol)
    write_json(protocol_path, protocol)
    return protocol


def train(seed):
    from torch.utils.data import DataLoader, Sampler
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    import torch
    import ultralytics

    protocol = prepare()
    key = f"uniform-{seed}"
    completion = OUT / key / "completion.json"
    if completion.exists():
        result = read_record(completion)
        if file_sha256(Path(result["weights"])) != result["weights_sha256"]:
            raise ValueError("Completed weights changed")
        return result
    if (OUT / key).exists():
        raise ValueError(f"Incomplete run exists: {OUT / key}")
    paths = [row["training_path"] for row in protocol["rows"]]
    schedule = protocol["schedules"][str(seed)]
    actual = []

    class ScheduledSampler(Sampler):
        def __init__(self, trainer, mapping):
            self.trainer, self.mapping = trainer, mapping

        def __len__(self):
            return SLOTS

        def __iter__(self):
            return iter(self.mapping[paths[index]] for index in schedule[self.trainer.epoch])

    class Trainer(DetectionTrainer):
        step_count = 0

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode != "train":
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            dataset = self.build_dataset(dataset_path, mode, batch_size)
            mapping = {path: index for index, path in enumerate(dataset.im_files)}
            if set(mapping) != set(paths) or batch_size != BATCH:
                raise ValueError("Unexpected training membership or batch size")
            return DataLoader(dataset, batch_size=BATCH, sampler=ScheduledSampler(self, mapping), num_workers=0, collate_fn=dataset.collate_fn, generator=torch.Generator().manual_seed(seed))

        def preprocess_batch(self, batch):
            actual.append({"epoch": self.epoch, "paths": list(batch["im_file"])})
            return super().preprocess_batch(batch)

        def optimizer_step(self):
            self.step_count += 1
            return super().optimizer_step()

    model = YOLO(str(WEIGHTS))
    model.train(
        trainer=Trainer,
        data=protocol["dataset"],
        epochs=EPOCHS,
        imgsz=640,
        batch=BATCH,
        nbs=BATCH,
        device="cpu",
        workers=0,
        optimizer="AdamW",
        lr0=.001,
        lrf=1,
        warmup_epochs=0,
        warmup_bias_lr=0,
        seed=seed,
        deterministic=True,
        patience=0,
        amp=False,
        mosaic=0,
        close_mosaic=0,
        mixup=0,
        copy_paste=0,
        degrees=0,
        translate=0,
        scale=0,
        shear=0,
        perspective=0,
        flipud=0,
        fliplr=0,
        hsv_h=0,
        hsv_s=0,
        hsv_v=0,
        project=str(OUT),
        name=key,
        plots=False,
        save=True,
        val=False,
    )
    expected_steps = protocol["optimizer_steps_per_run"]
    if model.trainer.step_count != expected_steps or len(actual) != expected_steps:
        raise ValueError("Training steps did not match protocol")
    for epoch in range(EPOCHS):
        observed = [path for batch in actual if batch["epoch"] == epoch for path in batch["paths"]]
        if observed != [paths[index] for index in schedule[epoch]]:
            raise ValueError("Observed schedule differs from frozen schedule")
    weights = OUT / key / "weights/last.pt"
    result = {
        "protocol_identity": protocol["identity"],
        "seed": seed,
        "status": "complete",
        "optimizer_steps": model.trainer.step_count,
        "weights": str(weights),
        "weights_sha256": file_sha256(weights),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "training_admitted": False,
        "promotable": False,
    }
    result["identity"] = object_sha256(result)
    write_json(completion, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.seed is None:
        protocol = prepare()
        print(json.dumps({"identity": protocol["identity"], "exposures": protocol["exposures"]}, indent=2))
    else:
        print(json.dumps(train(args.seed), indent=2))
