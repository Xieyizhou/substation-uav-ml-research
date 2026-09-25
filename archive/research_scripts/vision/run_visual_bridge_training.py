"""Run one frozen visual-bridge A/B/C/D development training cell."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml.artifacts import file_sha256, object_sha256, write_json
from scripts.vision.prepare_visual_bridge_training import OUT, prepare


def train(arm, seed):
    from torch.utils.data import DataLoader, Sampler
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    import torch
    import ultralytics

    protocol = prepare(); controls = protocol["controls"]
    if seed not in protocol["seeds"] or arm not in protocol["membership"]: raise ValueError("Unknown training cell")
    dataset_spec = protocol["dataset_yamls"][arm]; dataset_path = Path(dataset_spec["path"])
    if file_sha256(dataset_path) != dataset_spec["sha256"]: raise ValueError("Arm dataset YAML changed")
    run_name = f"arm-{arm}-seed-{seed}"; completion = OUT / run_name / "completion.json"
    if completion.exists():
        result = json.loads(completion.read_text())
        if result["protocol_identity"] != protocol["identity"] or file_sha256(Path(result["weights"])) != result["weights_sha256"]:
            raise ValueError("Completed training cell no longer matches frozen protocol")
        return result
    if (OUT / run_name).exists(): raise ValueError(f"Incomplete run exists: {run_name}")
    rows = {row["member_id"]: row for row in protocol["pool_rows"]}
    member_ids = protocol["membership"][arm]["member_ids"]
    paths = [rows[item]["image_path"] for item in member_ids]
    path_for_member = {item: rows[item]["image_path"] for item in member_ids}
    expected = [[path_for_member[item] for item in epoch] for epoch in protocol["schedules"][arm][str(seed)]]
    actual = []

    class ScheduledSampler(Sampler):
        def __init__(self, trainer, mapping): self.trainer, self.mapping = trainer, mapping
        def __len__(self): return controls["slots_per_epoch"]
        def __iter__(self): return iter(self.mapping[path] for path in expected[self.trainer.epoch])

    class Trainer(DetectionTrainer):
        step_count = 0
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode != "train": return super().get_dataloader(dataset_path, batch_size, rank, mode)
            dataset = self.build_dataset(dataset_path, mode, batch_size); mapping = {path: index for index, path in enumerate(dataset.im_files)}
            if set(mapping) != set(paths) or batch_size != controls["batch"]: raise ValueError("Runtime membership or batch differs")
            return DataLoader(dataset, batch_size=controls["batch"], sampler=ScheduledSampler(self, mapping), num_workers=0,
                              collate_fn=dataset.collate_fn, generator=torch.Generator().manual_seed(seed))
        def preprocess_batch(self, batch):
            actual.append({"epoch": self.epoch, "paths": list(batch["im_file"])}); return super().preprocess_batch(batch)
        def optimizer_step(self): self.step_count += 1; return super().optimizer_step()

    model = YOLO(controls["initial_weights"])
    model.train(trainer=Trainer, data=str(dataset_path), epochs=controls["epochs"], imgsz=controls["input_size"],
                batch=controls["batch"], nbs=controls["batch"], device="cpu", workers=0, optimizer=controls["optimizer"],
                lr0=controls["lr0"], lrf=controls["lrf"], warmup_epochs=0, warmup_bias_lr=0, seed=seed,
                deterministic=True, patience=0, amp=False, mosaic=0, close_mosaic=0, mixup=0, copy_paste=0,
                degrees=0, translate=0, scale=0, shear=0, perspective=0, flipud=0, fliplr=0,
                hsv_h=0, hsv_s=0, hsv_v=0, project=str(OUT), name=run_name, plots=False, save=True, val=False)
    if model.trainer.step_count != controls["optimizer_steps"] or len(actual) != controls["optimizer_steps"]:
        raise ValueError("Optimizer-step count differs from protocol")
    for epoch in range(controls["epochs"]):
        observed = [path for batch in actual if batch["epoch"] == epoch for path in batch["paths"]]
        if observed != expected[epoch]: raise ValueError(f"Actual exposure differs in epoch {epoch}")
    weights = OUT / run_name / "weights/last.pt"
    result = {"status": "complete", "arm": arm, "seed": seed, "protocol_identity": protocol["identity"],
              "optimizer_steps": model.trainer.step_count, "observed_draws": sum(len(item["paths"]) for item in actual),
              "exposure_verified": True, "weights": str(weights), "weights_sha256": file_sha256(weights),
              "ultralytics": ultralytics.__version__, "torch": torch.__version__,
              "development_training_only": True, "training_admitted": False, "promotable": False}
    result["identity"] = object_sha256(result); write_json(completion, result); return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--arm", choices=list("ABCD"), required=True)
    parser.add_argument("--seed", type=int, choices=[7, 17, 27], required=True); args = parser.parse_args()
    print(json.dumps(train(args.arm, args.seed), indent=2))
