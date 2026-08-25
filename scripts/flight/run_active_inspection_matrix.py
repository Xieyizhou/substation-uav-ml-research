#!/usr/bin/env python3
"""Run one manifest-selected live flight with owned-process cleanup only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--map-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--diagnostics-output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    map_row = next(row for row in manifest["maps"] if row["map_id"] == args.map_id)
    scheduler = next((name for name, ids in map_row["runs"].items() if args.run_id in ids), None)
    if scheduler is None:
        raise SystemExit("run-id is not registered for map-id")
    output = ROOT / manifest["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT), "MPLCONFIGDIR": "/tmp/matplotlib-cache"}
    launcher_pid_path = ROOT / ".runtime/px4_launcher.pid"
    launcher_pid_before = launcher_pid_path.read_text().strip() if launcher_pid_path.exists() else None
    owned_launcher_pid = None
    simulator_log = (output / f"{args.run_id}-simulator.log").open("w")
    simulator = subprocess.Popen(
        [str(ROOT / ".venv/bin/python"), "main.py", "map", "start", args.map_id, "--vehicle-model", "x500_research", "--display-mode", "headless"],
        cwd=ROOT, env=env, stdout=simulator_log, stderr=subprocess.STDOUT,
    )
    try:
        time.sleep(70)
        launcher_pid_after = launcher_pid_path.read_text().strip() if launcher_pid_path.exists() else None
        if launcher_pid_after and launcher_pid_after != launcher_pid_before:
            owned_launcher_pid = int(launcher_pid_after)
        elif simulator.poll() is not None:
            raise RuntimeError("map launcher exited before RGB-D probe")
        probe = subprocess.run([
            str(ROOT / ".venv/bin/python"), "scripts/flight/probe_rgbd_native.py",
            "--pairs", "100", "--timeout", "45", "--output", str(output / f"{args.run_id}-prearm.json"),
        ], cwd=ROOT, env=env)
        if probe.returncode:
            return probe.returncode
        with (output / f"{args.run_id}-console.log").open("w") as console:
            flight_command = [
                str(ROOT / ".venv/bin/python"), "main.py", "astar", "fly", "--compact-output",
                "--runtime-mode", "active_semantic_inspection", "--active-inspection-trial", manifest["policy"],
                "--inspection-scheduler", scheduler, "--equipment-model", manifest["model"],
                "--enable-perception", "--perception-source", "gazebo_lidar_2d", "--risk-model", "geometric",
                "--risk-fusion", "safety_max", "--risk-action", "stop_and_land", "--enable-local-replan",
                "--replan-mode", "active", "--max-speed", "1.0", "--altitude", "1.5", "--sensor-startup-timeout", "30",
                "--visual-mission-events", str(output / f"{args.run_id}-events.jsonl"),
            ]
            if args.diagnostics_output:
                flight_command.extend(["--active-inspection-diagnostics", str(args.diagnostics_output)])
            flight = subprocess.run(flight_command, cwd=ROOT, env=env, stdout=console, stderr=subprocess.STDOUT)
        return flight.returncode
    finally:
        if owned_launcher_pid is not None:
            try:
                os.kill(owned_launcher_pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            for _ in range(150):
                try:
                    os.kill(owned_launcher_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(.2)
        if simulator.poll() is None:
            simulator.send_signal(signal.SIGINT)
            try:
                simulator.wait(timeout=30)
            except subprocess.TimeoutExpired:
                simulator.terminate()
                simulator.wait(timeout=10)
        simulator_log.close()
        time.sleep(15)


if __name__ == "__main__":
    sys.exit(main())
