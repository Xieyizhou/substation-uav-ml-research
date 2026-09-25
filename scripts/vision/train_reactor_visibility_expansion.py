"""Start the explicit source-isolated visibility expansion training.

Training is reachable only after the immutable collection review and all three
real-loader preflights.  The runner uses two CPU workers at a time and keeps
each seed's failure/attempt directory separate.
"""
import argparse
import fcntl
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from src.ml.artifacts import file_sha256, object_sha256
from src.vision.canonical.plan import read_record, write_record
from scripts.vision.prepare_reactor_visibility_training import OUT, SEEDS, STEPS, BATCH, freeze, preflight, _loader

KEYS = tuple(f"visibility-452-{s}" for s in SEEDS)


def _ready():
    p = read_record(OUT / "protocol.json")
    r = read_record(OUT / "entry-ready.json")
    if r["status"] != "ready_for_training_not_started" or r["cells"] != list(KEYS):
        raise ValueError("Training readiness receipt invalid")
    if r["actual_draws_verified"] != STEPS * BATCH * len(SEEDS):
        raise ValueError("Training readiness draw count invalid")
    if any(r[k] for k in ("optimizer_created", "backward_executed", "training_admitted", "promotable")):
        raise ValueError("Readiness has forbidden flags")
    if file_sha256(Path(p["initialization"]["path"])) != p["initialization"]["sha256"]:
        raise ValueError("v2.11 initialization changed")
    return p, r


def _unit_complete(key):
    path = OUT / "training" / key / "completion.json"
    if not path.exists(): return None
    r = read_record(path)
    if r["status"] != "trained_not_evaluated" or r["optimizer_steps"] != STEPS:
        raise ValueError("Reusable endpoint invalid")
    if file_sha256(r["weights"]) != r["weights_sha256"]:
        raise ValueError("Reusable weight hash changed")
    return r


def _worker(key):
    import math
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    p, ready = _ready(); existing = _unit_complete(key)
    if existing: return existing
    seed = int(key.rsplit("-", 1)[1]); root = OUT / "training" / key; root.mkdir(parents=True, exist_ok=True)
    with (root / "unit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = _unit_complete(key)
        if existing: return existing
        attempts = sorted(root.glob("attempt-*")); n = len(attempts) + 1
        if n > 3: raise ValueError("Technical attempt cap")
        attempt = root / f"attempt-{n:03}"; attempt.mkdir()
        listing = Path(p["listings"][key]); data_yaml = attempt / "dataset.yaml"
        data_yaml.write_text(yaml.safe_dump({"path": str(attempt), "train": str(listing), "val": str(listing), "names": p["names"]}))
        seq = p["schedules"][key]; lookup = {r["image_path"]: r["member_id"] for r in p["pool_rows"]}; actual=[]; losses=[]; started = time.monotonic()

        class Trainer(DetectionTrainer):
            step_count = 0
            def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
                if mode != "train": return super().get_dataloader(dataset_path, batch_size, rank, mode)
                if str(dataset_path) != str(listing) or batch_size != BATCH: raise ValueError("Loader configuration drift")
                return _loader(p, key, self)[1]
            def preprocess_batch(self, batch):
                mids = [lookup[x] for x in batch["im_file"]]
                pos = len(actual)
                if mids != seq[pos:pos+BATCH]: raise ValueError("Runtime exposure order drift")
                actual.extend(mids)
                if tuple(batch["img"].shape) != (BATCH, 3, 640, 640): raise ValueError("Runtime tensor shape drift")
                return super().preprocess_batch(batch)
            def optimizer_step(self):
                super().optimizer_step(); self.step_count += 1
                values = self.loss_items
                if hasattr(values, "detach"): values = values.detach().cpu().tolist()
                if isinstance(values, dict): values = {str(k): float(v) for k, v in values.items()}
                else: values = [float(v) for v in values]
                if any(not math.isfinite(float(v)) for v in (values.values() if isinstance(values, dict) else values)):
                    raise ValueError("Non-finite training loss")
                losses.append({"step": self.step_count, "loss_items": values})

        try:
            torch.set_num_threads(4)
            model = YOLO(p["initialization"]["path"])
            model.train(trainer=Trainer, data=str(data_yaml), epochs=STEPS // 10, imgsz=640, batch=BATCH, nbs=BATCH,
                device="cpu", workers=0, optimizer="AdamW", lr0=0.0005, lrf=1.0, warmup_epochs=0,
                warmup_bias_lr=0, seed=seed, deterministic=True, patience=0, amp=False, mosaic=0, close_mosaic=0,
                mixup=0, copy_paste=0, degrees=0, translate=0, scale=0, shear=0, perspective=0,
                flipud=0, fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0, plots=False, save=True, val=False,
                project=str(attempt), name="run")
            steps_done = int(model.trainer.step_count)
            if actual != seq or steps_done != STEPS: raise ValueError("Training exposure/step endpoint mismatch")
            weight = attempt / "run/weights/last.pt"
            if not weight.is_file(): raise ValueError("Missing last.pt")
            exposure = write_record(attempt / "exposure.json", {"key": key, "actual": actual, "draws": len(actual), "optimizer_steps": steps_done,
                "image_exposure": {m: actual.count(m) for m in sorted(set(actual))}, "training_admitted": False, "promotable": False,
                "inputs": {str(x): file_sha256(x) for x in (OUT/"protocol.json", OUT/"entry-ready.json", data_yaml)}})
            exposure_path = attempt / "exposure.json"
            completion = write_record(root / "completion.json", {"status": "trained_not_evaluated", "key": key, "weights": str(weight),
                "weights_sha256": file_sha256(weight), "exposure_path": str(attempt/"exposure.json"), "optimizer_steps": steps_done,
                "wall_seconds": time.monotonic()-started, "loss_curve": losses, "training_admitted": False, "promotable": False,
                "inputs": {str(x): file_sha256(x) for x in (weight, exposure_path, OUT/"protocol.json", OUT/"entry-ready.json")}})
            print("TRAINING_COMPLETE", key, flush=True); return completion
        except BaseException:
            write_record(attempt / "failure.json", {"status": "failed", "key": key, "error": traceback.format_exc(), "training_admitted": False, "promotable": False})
            raise


def run():
    preflight(); _ready(); runner = OUT / "training-runner"; runner.mkdir(exist_ok=True)
    with (runner / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); started = time.monotonic()
        for start in range(0, len(KEYS), 2):
            jobs=[]
            try:
                for key in KEYS[start:start+2]:
                    if _unit_complete(key): continue
                    folder=runner/key; folder.mkdir(exist_ok=True); n=len(list(folder.glob("attempt-*")))+1
                    if n>3: raise ValueError("Runner attempt cap")
                    attempt=folder/f"attempt-{n:03}";attempt.mkdir();log=(attempt/"log.txt").open("x")
                    proc=subprocess.Popen([sys.executable,"-u","-m","scripts.vision.train_reactor_visibility_expansion","--worker",key],cwd=Path(__file__).resolve().parents[2],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    jobs.append((proc,log,key,attempt)); print("TRAINING_STARTED",key,proc.pid,"THREADS=4",flush=True)
                for proc,_,key,_ in jobs:
                    proc.wait(timeout=14400)
                    if proc.returncode: raise RuntimeError("Worker failed "+key)
            except BaseException:
                err=traceback.format_exc()
                for proc,_,key,attempt in jobs:
                    if proc.poll() is None:
                        proc.terminate()
                        try: proc.wait(timeout=3)
                        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                    if not (attempt/"failure.json").exists(): write_record(attempt/"failure.json",{"status":"failed","key":key,"error":err,"process_cleanup_complete":proc.poll() is not None})
                raise
            finally:
                for proc,log,_,_ in jobs:
                    if proc.poll() is None: proc.terminate()
                    log.close()
        completed=[_unit_complete(k) for k in KEYS]
        write_record(runner/"completion.json",{"status":"three_units_complete_evaluation_pending","training_started":True,"wall_seconds":time.monotonic()-started,
            "inputs":{str(OUT/"training"/k/"completion.json"):file_sha256(OUT/"training"/k/"completion.json") for k in KEYS}})


if __name__ == "__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--worker",choices=KEYS);ap.add_argument("--train",action="store_true");a=ap.parse_args()
    if a.worker: _worker(a.worker)
    elif a.train: run()
    else: preflight(); print("PREFLIGHT_ONLY_NO_TRAINING")
