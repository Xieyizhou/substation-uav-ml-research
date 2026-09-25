#!/usr/bin/env python3
"""Run one isolated background pilot using real RGB-D and simulator truth."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ml.artifacts import object_sha256
from src.vision.collection.gazebo_truth import GazeboTruthSource
from scripts.vision.collect_hard_examples import collect
from scripts.vision.run_hard_example_flight import (
    _events, _terminate, _wait_any_event, _wait_event_fields,
    _wait_for_previous_launcher,
)

PARTITION = "substation_background_calibration"


def load_scene(scene_id, package=None):
    package = Path(package or ROOT / "data/research/background_calibration_v1").resolve()
    if (package / "PAUSED.json").exists():
        raise ValueError("Background package paused: excluded from product-scope collection")
    manifest = json.loads((package / "manifest.json").read_text())
    identity = manifest.pop("identity")
    if identity != object_sha256(manifest) or manifest.get("development_only") is not True:
        raise ValueError("Invalid development package")
    scene = next(s for s in manifest["scenes"] if s["scene_id"] == scene_id)
    paths = {}
    for key, record in scene["files"].items():
        path = (ROOT / record["path"]).resolve()
        if not path.is_relative_to(package.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Scene artifact identity mismatch")
        paths[key] = path
    return identity, scene, paths


async def truth_gate():
    source = GazeboTruthSource(width=1920, height=1080, topic="auto")
    try:
        await source.start()
        async for event in source.events(timeout_s=45):
            record = event.truth.to_record()
            if not event.truth.valid or event.truth.objects:
                raise ValueError("Background pre-arm requires real valid empty equipment truth")
            return record
    finally:
        await source.stop()
    raise RuntimeError("No simulator truth received")


def stop_owned_group(process):
    if process is None:
        return
    # The group was created by this adapter, not discovered by process name.
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            continue
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--package", type=Path, default=ROOT / "data/research/background_calibration_v1")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--collect-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.package = args.package.resolve()
    identity, scene, paths = load_scene(args.scene_id, args.package)
    frames = int(scene.get("collection_frames", 60))
    if not 1 <= frames <= 90:
        raise ValueError("Background collection must be bounded to 1..90 frames")
    os.environ.update({"GZ_IP": "127.0.0.1", "GZ_PARTITION": PARTITION})
    if args.collect_only:
        # Use the existing collector's implementation, not a fake canonical ID.
        # Its legacy protocol field describes the serialization version; the
        # parent run receipt binds the actual scene package and seed.
        return asyncio.run(collect(SimpleNamespace(output=args.output, map_id=args.scene_id, seed=scene["seed"], split="development", frames=frames, emit_stride=30, timeout=45)))
    args.output.mkdir(parents=True, exist_ok=False)
    _wait_for_previous_launcher(ROOT / ".runtime/px4_launcher.pid")
    env = {**os.environ, "PYTHONPATH": str(ROOT), "MPLCONFIGDIR": "/tmp/matplotlib-cache"}
    events = args.output / "flight-events.jsonl"
    launcher = flight = None
    stage = "startup"
    error = None
    probe_rc = collector_rc = flight_rc = None
    with (args.output / "simulator.log").open("w") as simlog, (args.output / "flight.log").open("w") as flightlog:
        try:
            launcher = subprocess.Popen([sys.executable, "scripts/vision/start_background_calibration_scene.py", args.scene_id, "--package", str(args.package)], cwd=ROOT, env=env, stdout=simlog, stderr=subprocess.STDOUT, start_new_session=True)
            time.sleep(140)
            if launcher.poll() is not None:
                raise RuntimeError("Background launcher exited before pre-arm")
            stage = "prearm"
            with (args.output / "prearm.log").open("w") as log:
                probe_rc = subprocess.run([sys.executable, "scripts/flight/probe_rgbd_native.py", "--pairs", "100", "--timeout", "45", "--output", str(args.output / "prearm.json")], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=100).returncode
            if probe_rc:
                raise RuntimeError("RGB-D pre-arm gate failed")
            truth = asyncio.run(asyncio.wait_for(truth_gate(), timeout=60))
            with (args.output / "prearm-truth.json").open("x") as stream:
                json.dump(truth, stream, indent=2, sort_keys=True)
            stage = "flight"
            command = [sys.executable, "scripts/flight/run_task.py", "run", "fly_round_trip", "--", "--obstacle-config", str(paths["obstacles.json"]), "--visual-route", str(paths["route.json"]), "--return-home", "--visual-mission-events", str(events)]
            flight = subprocess.Popen(command, cwd=ROOT, env=env, stdout=flightlog, stderr=subprocess.STDOUT)
            _wait_any_event(events, ("takeoff_completed", "waypoint_started"), flight, 60)
            _wait_event_fields(events, "yaw_settled", flight, 240, waypoint_name="complete_distant")
            stage = "collecting"
            with (args.output / "collector.log").open("w") as log:
                collector_rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--scene-id", args.scene_id, "--package", str(args.package), "--output", str(args.output / "collection"), "--collect-only"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180).returncode
            if collector_rc:
                raise RuntimeError("Background RGB-D/truth collector failed")
            stage = "landing"
            flight_rc = flight.wait(timeout=420)
            rows = _events(events)
            kinds = {r.get("event_type") or r.get("event") for r in rows}
            if flight_rc or "landing_confirmed" not in kinds or "collision" in kinds:
                raise RuntimeError("Flight/landing safety evidence failed")
            stage = "complete"
        except (OSError, RuntimeError, ValueError, TimeoutError, subprocess.TimeoutExpired) as exc:
            error = f"{type(exc).__name__}: {exc}"
            stage = "blocked"
        finally:
            try:
                _terminate(flight, timeout=180)
            except (OSError, subprocess.TimeoutExpired) as exc:
                stage = "blocked"
                error = f"{error or ''}; flight cleanup failed: {exc}"
            finally:
                if flight is None or any((r.get("event_type") or r.get("event")) == "landing_confirmed" for r in _events(events)):
                    stop_owned_group(launcher)
                else:
                    stage = "blocked"
                    error = f"{error or ''}; landing unconfirmed: simulator retained for recovery"
    cp = args.output / "collection/collection-receipt.json"
    collection = json.loads(cp.read_text()) if cp.exists() else {}
    receipt = {"schema_version": 1, "protocol_id": "background-calibration-v1", "status": stage, "map_id": args.scene_id, "scene_package_identity": identity, "scene_files": scene["files"], "route_identity": scene["route_identity"], "seed": scene["seed"], "split": "development", "requested_frames": 60, "collection_trigger": "yaw_settled:complete_distant", "probe_returncode": probe_rc, "collector_returncode": collector_rc, "flight_returncode": flight_rc, "landing_confirmed": any((r.get("event_type") or r.get("event")) == "landing_confirmed" for r in _events(events)), "collection_identity": collection.get("identity"), "legacy_collection_protocol_id": collection.get("protocol_id"), "transport_partition": PARTITION, "training_started": False, "error": error}
    receipt["requested_frames"] = frames
    receipt["scene_package_path"] = str(args.package)
    receipt["identity"] = object_sha256(receipt)
    with (args.output / "run-receipt.json").open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0 if stage == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
