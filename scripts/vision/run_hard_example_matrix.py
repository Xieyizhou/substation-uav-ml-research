#!/usr/bin/env python3
"""Run the v2.1 hard-example flight matrix sequentially and fail closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _wait_for_launcher_exit(timeout=300):
    pid_path = ROOT / ".runtime/px4_launcher.pid"
    deadline = time.monotonic() + timeout
    while pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, 0)
        except (FileNotFoundError, ProcessLookupError, ValueError):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"project launcher PID {pid} did not exit")
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=ROOT / "config/perception/visual_hard_examples_v2_1_matrix.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/research/visual_hard_examples_v2_1")
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    rows = matrix["runs"][: args.max_runs] if args.max_runs is not None else matrix["runs"]
    results = []
    for row in rows:
        revision = row.get("run_revision", matrix["run_revision"])
        run_id = f"{row['map_id']}-{row['seed']}-{row['split']}-{revision}"
        run_root = args.output / "runs" / run_id
        receipt_path = run_root / "run-receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") == "complete":
                results.append(receipt)
                continue
            raise SystemExit(f"blocked evidence already exists for {run_id}; use a new run ID")
        _wait_for_launcher_exit()
        command = [str(ROOT / ".venv/bin/python"), "scripts/vision/run_hard_example_flight.py", "--map-id", row["map_id"], "--seed", str(row["seed"]), "--split", row["split"], "--frames", str(row.get("frames", matrix["frames_per_run"])), "--emit-stride", str(row.get("emit_stride", matrix["emit_stride"])), "--scheduler", row.get("scheduler", "active_utility"), "--output", str(run_root)]
        if row.get("collection_start_delay_s") is not None:
            command.extend(["--collection-start-delay", str(row["collection_start_delay_s"])])
        if row.get("visual_route"):
            command.extend(["--visual-route", row["visual_route"]])
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode:
            raise SystemExit(f"hard-example matrix stopped at {run_id}")
        results.append(json.loads(receipt_path.read_text()))
        time.sleep(float(matrix.get("cooldown_s", 0)))
    report = {"schema_version":1,"matrix_id":matrix["matrix_id"],"status":"complete" if len(results)==len(rows) else "partial","run_count":len(results),"run_identities":[row["identity"] for row in results]}
    report["identity"] = hashlib.sha256(_canonical(report).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "matrix-receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
