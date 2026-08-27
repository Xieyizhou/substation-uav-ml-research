#!/usr/bin/env python3
"""Run one non-acceptance flight and collect split-isolated hard examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "config/perception/visual_hard_examples_v2_1.json"
MODEL = ROOT / "models/equipment/visual-yolo11n-baseline-v2-package/weights/best.pt"
POLICY = ROOT / "config/perception/active_inspection_qualification_v1.json"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _events(path):
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _wait_event(path, event_type, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(row.get("event") == event_type or row.get("event_type") == event_type for row in _events(path)):
            return
        if process.poll() is not None:
            raise RuntimeError(f"flight exited before {event_type}")
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for {event_type}")


def _wait_any_event(path, event_types, process, timeout):
    deadline = time.monotonic() + timeout
    expected = set(event_types)
    while time.monotonic() < deadline:
        if any((row.get("event") or row.get("event_type")) in expected for row in _events(path)):
            return
        if process.poll() is not None:
            raise RuntimeError(f"flight exited before one of {sorted(expected)}")
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for one of {sorted(expected)}")


def _wait_event_fields(path, event_type, process, timeout, **fields):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for row in _events(path):
            observed = row.get("event") or row.get("event_type")
            if observed == event_type and all(row.get(key) == value for key, value in fields.items()):
                return
        if process.poll() is not None:
            raise RuntimeError(f"flight exited before {event_type} matching {fields}")
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for {event_type} matching {fields}")


def _terminate(process, timeout=30):
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=10)


def _terminate_group(process, timeout=180):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        process.wait(timeout=30)


def _wait_for_previous_launcher(pid_path, timeout=300):
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


def run(args):
    protocol = json.loads(args.protocol.read_text())
    expected_split = "development" if args.seed in protocol["development_seeds"] else "validation" if args.seed in protocol["validation_seeds"] else None
    if expected_split != args.split:
        raise ValueError("seed is not registered for the requested split")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT), "GZ_IP": "127.0.0.1", "GZ_PARTITION": "substation_uav", "MPLCONFIGDIR": "/tmp/matplotlib-cache"}
    events = args.output / "flight-events.jsonl"
    launcher_pid_path = ROOT / ".runtime/px4_launcher.pid"
    _wait_for_previous_launcher(launcher_pid_path)
    launcher_pid_before = launcher_pid_path.read_text().strip() if launcher_pid_path.exists() else None
    launcher = flight = None
    owned_launcher_pid = None
    stage = "starting"
    error = None
    probe_rc = collector_rc = flight_rc = None
    simulator_log = (args.output / "simulator.log").open("w")
    flight_log = (args.output / "flight.log").open("w")
    try:
        launcher = subprocess.Popen([str(ROOT / ".venv/bin/python"), "main.py", "map", "start", args.map_id, "--vehicle-model", "x500_research", "--display-mode", "headless"], cwd=ROOT, env=env, stdout=simulator_log, stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(args.startup_wait)
        launcher_pid_after = launcher_pid_path.read_text().strip() if launcher_pid_path.exists() else None
        if launcher_pid_after and launcher_pid_after != launcher_pid_before:
            owned_launcher_pid = int(launcher_pid_after)
        if launcher.poll() is not None:
            raise RuntimeError("map launcher exited before pre-arm")
        stage = "prearm"
        probe = subprocess.run([str(ROOT / ".venv/bin/python"), "scripts/flight/probe_rgbd_native.py", "--pairs", "100", "--timeout", "45", "--output", str(args.output / "prearm.json")], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        probe_rc = probe.returncode
        if probe_rc:
            raise RuntimeError("RGB-D pre-arm probe failed")
        stage = "flight"
        if args.visual_route is None:
            command = [str(ROOT / ".venv/bin/python"), "main.py", "astar", "fly", "--compact-output", "--runtime-mode", "active_semantic_inspection", "--active-inspection-trial", str(POLICY), "--inspection-scheduler", args.scheduler, "--equipment-model", str(MODEL), "--enable-perception", "--perception-source", "gazebo_lidar_2d", "--risk-model", "geometric", "--risk-fusion", "safety_max", "--risk-action", "stop_and_land", "--enable-local-replan", "--replan-mode", "active", "--max-speed", "1.0", "--altitude", "1.5", "--sensor-startup-timeout", "30", "--visual-mission-events", str(events)]
        else:
            obstacle = ROOT / {
                "simple": "config/substation_obstacles.json",
                "medium": "config/maps/substation_medium.json",
                "complex": "config/maps/substation_complex.json",
            }[args.map_id]
            command = [str(ROOT / ".venv/bin/python"), "main.py", "task", "run", "fly_round_trip", "--", "--obstacle-config", str(obstacle), "--visual-route", str(args.visual_route), "--return-home", "--visual-mission-events", str(events)]
        flight = subprocess.Popen(command, cwd=ROOT, env=env, stdout=flight_log, stderr=subprocess.STDOUT)
        _wait_any_event(events, ("takeoff_completed", "waypoint_started"), flight, 60)
        collection_trigger = "takeoff_or_waypoint"
        if args.visual_route is not None:
            _wait_event_fields(events, "yaw_settled", flight, 240, waypoint_name="complete_distant")
            collection_trigger = "yaw_settled:complete_distant"
        if args.collection_start_delay:
            deadline = time.monotonic() + args.collection_start_delay
            while time.monotonic() < deadline:
                if flight.poll() is not None:
                    raise RuntimeError("qualification flight exited before delayed collection")
                time.sleep(min(0.5, deadline - time.monotonic()))
        stage = "collecting"
        collector_log = (args.output / "collector.log").open("w")
        try:
            collector = subprocess.run([str(ROOT / ".venv/bin/python"), "scripts/vision/collect_hard_examples.py", "--map-id", args.map_id, "--seed", str(args.seed), "--split", args.split, "--frames", str(args.frames), "--output", str(args.output / "collection"), "--emit-stride", str(args.emit_stride), "--timeout", "45"], cwd=ROOT, env=env, stdout=collector_log, stderr=subprocess.STDOUT, timeout=args.collection_timeout)
            collector_rc = collector.returncode
        finally:
            collector_log.close()
        if collector_rc:
            raise RuntimeError("hard-example collector failed")
        stage = "landing"
        flight_rc = flight.wait(timeout=args.flight_timeout)
        if flight_rc:
            raise RuntimeError("qualification flight failed")
        rows = _events(events)
        if not any(row.get("event") == "landing_confirmed" or row.get("event_type") == "landing_confirmed" for row in rows):
            raise RuntimeError("landing was not confirmed")
        if any(row.get("event") == "collision" or row.get("event_type") == "collision" for row in rows):
            raise RuntimeError("collision event recorded")
        stage = "complete"
    except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired, ValueError) as exception:
        error = f"{type(exception).__name__}: {exception}"
        stage = "blocked"
    finally:
        _terminate(flight)
        _terminate_group(launcher)
        if owned_launcher_pid is not None:
            try: os.kill(owned_launcher_pid, signal.SIGINT)
            except ProcessLookupError: pass
            for _ in range(150):
                try: os.kill(owned_launcher_pid, 0)
                except ProcessLookupError: break
                time.sleep(0.2)
        simulator_log.close(); flight_log.close()
    receipt = {"schema_version":1,"protocol_id":protocol["protocol_id"],"status":stage,"map_id":args.map_id,"seed":args.seed,"split":args.split,"scheduler":args.scheduler if args.visual_route is None else "target_centered_route","visual_route":str(args.visual_route) if args.visual_route is not None else None,"requested_frames":args.frames,"collection_trigger":locals().get("collection_trigger"),"collection_start_delay_s":args.collection_start_delay,"probe_returncode":probe_rc,"collector_returncode":collector_rc,"flight_returncode":flight_rc,"landing_confirmed":any(row.get("event") == "landing_confirmed" or row.get("event_type") == "landing_confirmed" for row in _events(events)),"error":error}
    collection_receipt = args.output / "collection/collection-receipt.json"
    receipt["collection_identity"] = json.loads(collection_receipt.read_text())["identity"] if collection_receipt.exists() else None
    receipt["identity"] = hashlib.sha256(_canonical(receipt).encode()).hexdigest()
    (args.output / "run-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if stage == "complete" else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-id", choices=("simple", "medium", "complex"), required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("development", "validation"), required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--emit-stride", type=int, default=6)
    parser.add_argument("--scheduler", choices=("fixed_serpentine", "nearest_target_first", "active_utility"), default="active_utility")
    parser.add_argument("--visual-route", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--startup-wait", type=float, default=70)
    parser.add_argument("--collection-timeout", type=float, default=180)
    parser.add_argument("--collection-start-delay", type=float, default=0)
    parser.add_argument("--flight-timeout", type=float, default=420)
    args = parser.parse_args()
    if min(args.frames, args.emit_stride) <= 0 or args.collection_start_delay < 0:
        parser.error("frames and emit-stride must be positive; collection delay must be non-negative")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
