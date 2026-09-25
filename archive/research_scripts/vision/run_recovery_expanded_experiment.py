"""Step-matched development experiment on the 66-image expanded pool."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PIL import Image
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.canonical.plan import read_record
from src.vision.collection.simulator_labels import instance_simulator_label


BASE = ROOT / "data/research/ml_training_recovery_v1"
OUT = BASE / "expanded-stratified-v1"
NAMES = ["transformer", "switchgear", "capacitor_bank", "reactor"]
SEEDS = [7, 17, 27]
ARMS = ["uniform", "stratified"]
EPOCHS, SLOTS, BATCH = 10, 30, 6
WEIGHTS = ROOT / "models/equipment/visual-yolo11n-baseline-v2.11-candidate-package-v1/weights/best.pt"


def area_bucket(box):
    area = (box[2] - box[0]) * (box[3] - box[1])
    return "small" if area < 40000 else "medium" if area < 180000 else "large"


def expected_object(row):
    label = instance_simulator_label(row["expected_category"], row["expected_object_id"])
    matches = [
        obj for obj in row["truth"]["objects"]
        if obj["class_name"] == row["expected_category"]
        and int(re.search(r"instance-(\d+)-", obj["annotation_id"])[1]) == label
    ]
    if len(matches) == 1:
        return matches[0], "planned_instance_present"
    candidates = [obj for obj in row["truth"]["objects"] if obj["class_name"] == row["expected_category"]]
    if not candidates:
        raise ValueError(f"No observed expected class: {row['view_id']}")
    return max(candidates, key=lambda obj: (obj["bbox_xyxy"][2] - obj["bbox_xyxy"][0]) * (obj["bbox_xyxy"][3] - obj["bbox_xyxy"][1])), "planned_instance_absent_use_observed_class_for_sampling_only"


def make_schedule(rows, arm, seed):
    rng = random.Random(seed)
    if arm == "uniform":
        epochs = []
        for _ in range(EPOCHS):
            selected = list(range(len(rows)))
            rng.shuffle(selected)
            epochs.append(selected[:SLOTS])
        return epochs
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for index, row in enumerate(rows):
        tree[row["expected_category"]][row["map_id"]][row["scale"]].append(index)
    counts = Counter()

    def choose(options, prefix):
        options = sorted(options)
        minimum = min(counts[prefix + (option,)] for option in options)
        value = rng.choice([option for option in options if counts[prefix + (option,)] == minimum])
        counts[prefix + (value,)] += 1
        return value

    epochs = []
    for _ in range(EPOCHS):
        selected = []
        for _ in range(SLOTS):
            category = choose(tree, ("class",))
            map_id = choose(tree[category], ("map", category))
            scale = choose(tree[category][map_id], ("scale", category, map_id))
            options = tree[category][map_id][scale]
            selected.append(options[rng.randrange(len(options))])
        rng.shuffle(selected)
        epochs.append(selected)
    return epochs


def exposure(rows, schedule):
    draws = [index for epoch in schedule for index in epoch]
    return {
        "draws": len(draws),
        "unique_frames": len(set(draws)),
        "by_expected_class": dict(Counter(rows[i]["expected_category"] for i in draws)),
        "by_map": dict(Counter(rows[i]["map_id"] for i in draws)),
        "by_source": dict(Counter(rows[i]["source"] for i in draws)),
        "by_scale": dict(Counter(rows[i]["scale"] for i in draws)),
        "by_stratum": dict(Counter(":".join((rows[i]["expected_category"], rows[i]["map_id"], rows[i]["scale"])) for i in draws)),
    }


def source_rows():
    mem = read_record(BASE / "memorization-v1/protocol.json")
    repair = read_record(BASE / "simple-annotation-repair-v2/capture/collection-receipt.json")
    repair_review = read_record(BASE / "simple-annotation-repair-v2/semantic-review.json")
    expansion_review = read_record(BASE / "stratified-expansion-v1/semantic-review.json")
    expansion_rows = []
    for map_id in ("simple", "medium", "complex"):
        receipt = read_record(BASE / "stratified-expansion-v1" / map_id / "capture/collection-receipt.json")
        expansion_rows.extend(receipt["views"])
    if len(expansion_rows) != expansion_review["accepted"] or expansion_review["held"]:
        raise ValueError("Expansion semantic review mismatch")
    rows = []
    for source, members in [
        ("original", [item["row"] for item in mem["selected"]]),
        ("repaired_simple", repair["views"]),
        ("coverage_expansion", expansion_rows),
    ]:
        if source == "repaired_simple" and {item["view_id"] for item in members} != {item["view_id"] for item in repair_review["frames"] if item["semantic_decision"] == "accepted"}:
            raise ValueError("Repaired semantic review mismatch")
        for row in members:
            image = Path(row["rgb_path"])
            if file_sha256(image) != row["image_sha256"]:
                raise ValueError(f"Image mismatch: {image}")
            anchor, anchor_status = expected_object(row)
            rows.append({**row, "source": source, "scale": area_bucket(anchor["bbox_xyxy"]), "sampling_anchor_status": anchor_status})
    if len(rows) != 66 or len({row["image_sha256"] for row in rows}) != 66:
        raise ValueError("Expected 66 unique images")
    return rows


def prepare():
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        protocol = read_record(protocol_path)
        for path, digest in protocol["inputs"].items():
            if file_sha256(Path(path)) != digest:
                raise ValueError(f"Frozen input changed: {path}")
        return protocol
    rows = source_rows()
    inputs = {str(Path(__file__)): file_sha256(Path(__file__)), str(WEIGHTS): file_sha256(WEIGHTS)}
    for input_path in [BASE / "memorization-v1/protocol.json", BASE / "simple-annotation-repair-v2/capture/collection-receipt.json", BASE / "simple-annotation-repair-v2/semantic-review.json", BASE / "stratified-expansion-v1/semantic-review.json"]:
        inputs[str(input_path)] = file_sha256(input_path)
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "images").mkdir(); (OUT / "labels").mkdir()
    for index, row in enumerate(rows):
        image_path = OUT / "images" / f"{index:03}.png"
        with Image.open(row["rgb_path"]) as image:
            if image.size != (1920, 1080):
                raise ValueError("Unexpected image size")
            image.convert("RGB").save(image_path)
        label_path = OUT / "labels" / f"{index:03}.txt"
        lines = []
        for obj in row["truth"]["objects"]:
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            if not (0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080):
                raise ValueError("Invalid box")
            lines.append(f"{NAMES.index(obj['class_name'])} {(x1+x2)/3840:.10f} {(y1+y2)/2160:.10f} {(x2-x1)/1920:.10f} {(y2-y1)/1080:.10f}")
        label_path.write_text("\n".join(lines) + "\n")
        row["training_path"] = str(image_path)
        inputs[str(image_path)] = file_sha256(image_path)
        inputs[str(label_path)] = file_sha256(label_path)
    dataset = OUT / "dataset.yaml"
    dataset.write_text(f"path: {OUT}\ntrain: images\nval: images\nnames: {json.dumps(NAMES)}\n")
    inputs[str(dataset)] = file_sha256(dataset)
    schedules = {f"{arm}-{seed}": make_schedule(rows, arm, seed) for arm in ARMS for seed in SEEDS}
    strata = sorted({f"{row['expected_category']}:{row['map_id']}:{row['scale']}" for row in rows})
    protocol = {
        "status": "frozen_before_training",
        "rows": rows,
        "inputs": inputs,
        "schedules": schedules,
        "exposures": {key: exposure(rows, schedule) for key, schedule in schedules.items()},
        "observed_strata": strata,
        "epochs": EPOCHS,
        "slots_per_epoch": SLOTS,
        "batch": BATCH,
        "optimizer_steps_per_run": 50,
        "seeds": SEEDS,
        "arms": ARMS,
        "weights": str(WEIGHTS),
        "dataset": str(dataset),
        "comparison_rule": "Paired seed summaries on the new hash-clean expansion set and the prior hash-disjoint controls/cross-scene sets; no best-seed selection.",
        "training_admitted": False,
        "promotable": False,
        "limits": [
            "The 66-image pool remains simulated development data.",
            "Each run uses 50 optimizer steps; repeated exposure is not new diversity.",
            "All original truth boxes are preserved; the sampling anchor is not a label edit.",
        ],
    }
    protocol["identity"] = object_sha256(protocol)
    write_json(protocol_path, protocol)
    return protocol


def train(arm, seed):
    from torch.utils.data import DataLoader, Sampler
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    import torch
    import ultralytics

    protocol = prepare(); key = f"{arm}-{seed}"; run_dir = OUT / key; completion = run_dir / "completion.json"
    if completion.exists():
        result = read_record(completion)
        if file_sha256(Path(result["weights"])) != result["weights_sha256"]:
            raise ValueError("Completed weights changed")
        return result
    if run_dir.exists():
        raise ValueError(f"Incomplete run exists: {run_dir}")
    paths = [row["training_path"] for row in protocol["rows"]]
    schedule = protocol["schedules"][key]
    actual = []

    class ScheduledSampler(Sampler):
        def __init__(self, trainer, mapping): self.trainer, self.mapping = trainer, mapping
        def __len__(self): return SLOTS
        def __iter__(self): return iter(self.mapping[paths[index]] for index in schedule[self.trainer.epoch])

    class Trainer(DetectionTrainer):
        step_count = 0
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode != "train": return super().get_dataloader(dataset_path, batch_size, rank, mode)
            dataset = self.build_dataset(dataset_path, mode, batch_size); mapping = {path: i for i, path in enumerate(dataset.im_files)}
            if set(mapping) != set(paths) or batch_size != BATCH: raise ValueError("Unexpected training membership/batch size")
            return DataLoader(dataset, batch_size=BATCH, sampler=ScheduledSampler(self, mapping), num_workers=0, collate_fn=dataset.collate_fn, generator=torch.Generator().manual_seed(seed))
        def preprocess_batch(self, batch):
            actual.append({"epoch": self.epoch, "paths": list(batch["im_file"])}); return super().preprocess_batch(batch)
        def optimizer_step(self): self.step_count += 1; return super().optimizer_step()

    model = YOLO(str(WEIGHTS))
    model.train(trainer=Trainer, data=protocol["dataset"], epochs=EPOCHS, imgsz=640, batch=BATCH, nbs=BATCH, device="cpu", workers=0, optimizer="AdamW", lr0=.001, lrf=1, warmup_epochs=0, warmup_bias_lr=0, seed=seed, deterministic=True, patience=0, amp=False, mosaic=0, close_mosaic=0, mixup=0, copy_paste=0, degrees=0, translate=0, scale=0, shear=0, perspective=0, flipud=0, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0, project=str(OUT), name=key, plots=False, save=True, val=False)
    if model.trainer.step_count != 50 or len(actual) != 50: raise ValueError("Training steps did not match protocol")
    for epoch in range(EPOCHS):
        observed = [path for batch in actual if batch["epoch"] == epoch for path in batch["paths"]]
        if observed != [paths[index] for index in schedule[epoch]]: raise ValueError("Observed schedule differs")
    weights = run_dir / "weights/last.pt"
    result = {"protocol_identity": protocol["identity"], "arm": arm, "seed": seed, "status": "complete", "optimizer_steps": model.trainer.step_count, "observed_batches": actual, "weights": str(weights), "weights_sha256": file_sha256(weights), "ultralytics": ultralytics.__version__, "torch": torch.__version__, "training_admitted": False, "promotable": False}
    result["identity"] = object_sha256(result); write_json(completion, result); return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--arm", choices=ARMS); parser.add_argument("--seed", type=int, choices=SEEDS); args = parser.parse_args()
    if args.arm is None: print(json.dumps(prepare(), indent=2))
    elif args.seed is None: parser.error("--arm requires --seed")
    else: print(json.dumps({k: v for k, v in train(args.arm, args.seed).items() if k != "observed_batches"}, indent=2))
