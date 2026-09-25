"""Explicit entry for the hard-negative rehearsal diagnostic.

The underlying worker is the already-reviewed visibility training worker, with
only its versioned output, schedule, step count and cell names rebound here.
No historical file is overwritten and the default command is preflight-only.
"""
import argparse
import fcntl
import subprocess
import sys
import time
import traceback
from pathlib import Path

from src.ml.artifacts import file_sha256
from src.vision.canonical.plan import read_record, write_record
from scripts.vision import train_reactor_visibility_expansion as worker_impl
from scripts.vision.prepare_negative_rehearsal_control import OUT, KEYS, STEPS, BATCH, preflight


def _configure():
    worker_impl.OUT = OUT
    worker_impl.KEYS = KEYS
    worker_impl.STEPS = STEPS
    worker_impl.BATCH = BATCH


def worker(key):
    _configure()
    return worker_impl._worker(key)


def train():
    ready = preflight(); _configure()
    if ready["status"] != "ready_for_training_not_started":
        raise ValueError("Training readiness receipt invalid")
    runner = OUT / "training-runner"; runner.mkdir(exist_ok=True)
    with (runner / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        for start in range(0, len(KEYS), 2):
            jobs = []
            try:
                for key in KEYS[start:start + 2]:
                    cp = OUT / "training" / key / "completion.json"
                    if cp.exists():
                        read_record(cp); continue
                    folder = runner / key; folder.mkdir(exist_ok=True)
                    attempts = sorted(folder.glob("attempt-*")); n = len(attempts) + 1
                    if n > 3: raise ValueError(f"Runner attempt cap exceeded: {key}")
                    attempt = folder / f"attempt-{n:03}"; attempt.mkdir()
                    log = (attempt / "log.txt").open("x")
                    proc = subprocess.Popen([sys.executable, "-u", "-m", "scripts.vision.train_negative_rehearsal_control", "--worker", key],
                                            cwd=Path(__file__).resolve().parents[2], stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    jobs.append((proc, log, key, attempt)); print("TRAINING_STARTED", key, proc.pid, "THREADS=4", flush=True)
                for proc, _, key, _ in jobs:
                    proc.wait(timeout=21600)
                    if proc.returncode: raise RuntimeError(f"Training worker failed: {key}")
            except BaseException:
                error = traceback.format_exc()
                for proc, _, key, attempt in jobs:
                    if proc.poll() is None:
                        proc.terminate()
                        try: proc.wait(timeout=3)
                        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                    if not (attempt / "failure.json").exists():
                        write_record(attempt / "failure.json", {"status": "failed", "key": key, "error": error,
                                                                   "process_cleanup_complete": proc.poll() is not None,
                                                                   "training_admitted": False, "promotable": False})
                raise
            finally:
                for proc, log, _, _ in jobs:
                    if proc.poll() is None: proc.terminate()
                    log.close()
        completions = {k: OUT / "training" / k / "completion.json" for k in KEYS}
        if not all(p.exists() for p in completions.values()):
            raise ValueError("Training did not produce all endpoints")
        write_record(runner / "completion.json", {"status": "three_units_complete_evaluation_pending",
                                                   "training_started": True,
                                                   "wall_seconds": time.monotonic() - started,
                                                   "inputs": {str(p): file_sha256(p) for p in completions.values()},
                                                   "training_admitted": False, "promotable": False})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--worker", choices=KEYS); ap.add_argument("--train", action="store_true"); args = ap.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.train:
        train()
    else:
        preflight(); print("PREFLIGHT_ONLY_NO_TRAINING")
