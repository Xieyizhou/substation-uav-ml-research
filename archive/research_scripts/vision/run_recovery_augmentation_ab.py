"""Controlled A/B runs for small-object generalization on the 66-image pool."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import file_sha256, object_sha256, write_json
from src.vision.canonical.plan import read_record


BASE = ROOT / "data/research/ml_training_recovery_v1"
SOURCE = BASE / "expanded-stratified-v1"
OUT = BASE / "augmentation-ab-v1"
WEIGHTS = ROOT / "models/equipment/visual-yolo11n-baseline-v2.11-candidate-package-v1/weights/best.pt"
ARMS = {
    "scale_aug": {"imgsz": 640, "scale": 0.25, "cls": 0.5},
    "highres": {"imgsz": 960, "scale": 0.0, "cls": 0.5},
    "cls_weight": {"imgsz": 640, "scale": 0.0, "cls": 0.75},
}
SEEDS = [7, 17, 27]
EPOCHS, SLOTS, BATCH = 10, 30, 6


def prepare():
    path = OUT / "protocol.json"
    if path.exists():
        protocol = read_record(path)
        for source_path, digest in protocol["inputs"].items():
            if file_sha256(Path(source_path)) != digest:
                raise ValueError(f"Frozen input changed: {source_path}")
        return protocol
    source_protocol = read_record(SOURCE / "protocol.json")
    inputs = {str(SOURCE / "protocol.json"): file_sha256(SOURCE / "protocol.json"), str(SOURCE / "dataset.yaml"): file_sha256(SOURCE / "dataset.yaml"), str(WEIGHTS): file_sha256(WEIGHTS), str(Path(__file__)): file_sha256(Path(__file__))}
    for row in source_protocol["rows"]:
        inputs.update({str(row["training_path"]): file_sha256(row["training_path"])})
    protocol = {
        "status": "frozen_before_training",
        "source_protocol_identity": source_protocol["identity"],
        "dataset": source_protocol["dataset"],
        "rows": source_protocol["rows"],
        "schedules": {f"uniform-{seed}": source_protocol["schedules"][f"uniform-{seed}"] for seed in SEEDS},
        "arms": ARMS,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "slots_per_epoch": SLOTS,
        "batch": BATCH,
        "optimizer_steps_per_run": 50,
        "weights": str(WEIGHTS),
        "inputs": inputs,
        "comparison_rule": "Paired seed means on the targeted 24-frame set, 80-frame controls and 25-frame cross-scene set. No best-seed selection.",
        "training_admitted": False,
        "promotable": False,
        "limits": ["One training variable changes per arm.", "All data are simulated development data.", "No threshold selection or promotion."],
    }
    protocol["identity"] = object_sha256(protocol)
    OUT.mkdir(parents=True, exist_ok=False)
    write_json(path, protocol)
    return protocol


def train(arm, seed):
    import torch
    import ultralytics
    from torch.utils.data import DataLoader, Sampler
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    protocol = prepare(); key = f"{arm}-{seed}"; run_dir = OUT / key; completion = run_dir / "completion.json"
    if completion.exists():
        result = read_record(completion)
        if file_sha256(Path(result["weights"])) != result["weights_sha256"]: raise ValueError("Completed weights changed")
        return result
    if run_dir.exists(): raise ValueError(f"Incomplete run exists: {run_dir}")
    paths = [row["training_path"] for row in protocol["rows"]]; schedule = protocol["schedules"][f"uniform-{seed}"]; actual = []

    class ScheduledSampler(Sampler):
        def __init__(self, trainer, mapping): self.trainer, self.mapping = trainer, mapping
        def __len__(self): return SLOTS
        def __iter__(self): return iter(self.mapping[paths[index]] for index in schedule[self.trainer.epoch])

    class Trainer(DetectionTrainer):
        step_count = 0
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode != "train": return super().get_dataloader(dataset_path, batch_size, rank, mode)
            dataset = self.build_dataset(dataset_path, mode, batch_size); mapping = {path: i for i, path in enumerate(dataset.im_files)}
            if set(mapping) != set(paths) or batch_size != BATCH: raise ValueError("Unexpected dataset membership")
            return DataLoader(dataset, batch_size=BATCH, sampler=ScheduledSampler(self, mapping), num_workers=0, collate_fn=dataset.collate_fn, generator=torch.Generator().manual_seed(seed))
        def preprocess_batch(self, batch):
            actual.append({"epoch": self.epoch, "paths": list(batch["im_file"])}); return super().preprocess_batch(batch)
        def optimizer_step(self): self.step_count += 1; return super().optimizer_step()

    cfg = protocol["arms"][arm]
    model = YOLO(str(WEIGHTS))
    model.train(trainer=Trainer, data=protocol["dataset"], epochs=EPOCHS, imgsz=cfg["imgsz"], batch=BATCH, nbs=BATCH, device="cpu", workers=0, optimizer="AdamW", lr0=.001, lrf=1, warmup_epochs=0, warmup_bias_lr=0, seed=seed, deterministic=True, patience=0, amp=False, mosaic=0, close_mosaic=0, mixup=0, copy_paste=0, degrees=0, translate=0, scale=cfg["scale"], shear=0, perspective=0, flipud=0, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0, cls=cfg["cls"], project=str(OUT), name=key, plots=False, save=True, val=False)
    if model.trainer.step_count != 50 or len(actual) != 50: raise ValueError("Training steps did not match protocol")
    for epoch in range(EPOCHS):
        observed = [path for batch in actual if batch["epoch"] == epoch for path in batch["paths"]]
        if observed != [paths[index] for index in schedule[epoch]]: raise ValueError("Observed schedule differs")
    weights = run_dir / "weights/last.pt"
    result = {"protocol_identity": protocol["identity"], "arm": arm, "seed": seed, "config": cfg, "status": "complete", "optimizer_steps": model.trainer.step_count, "observed_batches": actual, "weights": str(weights), "weights_sha256": file_sha256(weights), "ultralytics": ultralytics.__version__, "torch": torch.__version__, "training_admitted": False, "promotable": False}
    result["identity"] = object_sha256(result); write_json(completion, result); return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--arm", choices=sorted(ARMS)); parser.add_argument("--seed", type=int, choices=SEEDS); args = parser.parse_args()
    if args.arm is None: print(json.dumps(prepare(), indent=2))
    elif args.seed is None: parser.error("--arm requires --seed")
    else: print(json.dumps({k: v for k, v in train(args.arm, args.seed).items() if k != "observed_batches"}, indent=2))
