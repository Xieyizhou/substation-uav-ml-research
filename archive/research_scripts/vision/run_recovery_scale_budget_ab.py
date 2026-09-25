"""Compare 50-step and 100-step budgets on the frozen 102-image pool."""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.canonical.plan import read_record


BASE = ROOT / "data/research/ml_training_recovery_v1"
SOURCE = BASE / "scale-expansion-v1"
OUT = BASE / "scale-budget-ab-v1"
SEEDS = [7, 17, 27]
EPOCHS, BATCH = 10, 6
WEIGHTS = ROOT / "models/equipment/visual-yolo11n-baseline-v2.11-candidate-package-v1/weights/best.pt"


def prepare():
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        protocol = read_record(protocol_path)
        for path, digest in protocol["inputs"].items():
            if file_sha256(Path(path)) != digest:
                raise ValueError(f"Frozen input changed: {path}")
        return protocol
    source_protocol_path = SOURCE / "protocol.json"
    source = read_record(source_protocol_path)
    inputs = {
        str(Path(__file__)): file_sha256(Path(__file__)),
        str(source_protocol_path): file_sha256(source_protocol_path),
        str(WEIGHTS): file_sha256(WEIGHTS),
        str(SOURCE / "dataset.yaml"): file_sha256(SOURCE / "dataset.yaml"),
    }
    schedules = {}
    for seed in SEEDS:
        rng = random.Random(seed)
        epochs = []
        for _ in range(EPOCHS):
            indices = list(range(len(source["rows"])))
            rng.shuffle(indices)
            epochs.append(indices[:30])
        schedules[str(seed)] = epochs
    protocol = {
        "status": "frozen_before_training",
        "source_protocol_identity": source["identity"],
        "source_dataset": source["dataset"],
        "rows": source["rows"],
        "schedules": schedules,
        "epochs": EPOCHS,
        "slots_per_epoch": 30,
        "batch": BATCH,
        "optimizer_steps_per_run": 50,
        "seeds": SEEDS,
        "budget_comparison": {
            "candidate": "50 optimizer steps",
            "reference": "100 optimizer steps from scale-expansion-v1",
            "reference_completion_paths": [str(SOURCE / f"uniform-{seed}/completion.json") for seed in SEEDS],
        },
        "training_admitted": False,
        "promotable": False,
        "inputs": inputs,
        "limits": [
            "Both budgets use the same frozen 102-image pool and 640px uniform sampling.",
            "The 100-step reference was completed under the source protocol and is only a paired development comparison.",
            "No protected labels or validation artifacts enter this experiment.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=False)
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
    key = f"steps50-{seed}"
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
            return 30

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
        data=protocol["source_dataset"],
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
    if model.trainer.step_count != 50 or len(actual) != 50:
        raise ValueError("Training steps did not match 50-step protocol")
    for epoch in range(EPOCHS):
        observed = [path for batch in actual if batch["epoch"] == epoch for path in batch["paths"]]
        if observed != [paths[index] for index in schedule[epoch]]:
            raise ValueError("Observed schedule differs from frozen schedule")
    weights = OUT / key / "weights/last.pt"
    result = {
        "protocol_identity": protocol["identity"],
        "seed": seed,
        "budget": "steps50",
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
        print(json.dumps({"identity": protocol["identity"], "source_protocol_identity": protocol["source_protocol_identity"]}, indent=2))
    else:
        print(json.dumps(train(args.seed), indent=2))
