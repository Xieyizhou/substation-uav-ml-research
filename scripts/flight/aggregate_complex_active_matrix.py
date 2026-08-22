#!/usr/bin/env python3
"""Aggregate the canonical nine-run Complex live inspection matrix."""

import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/sandbox/active_rgbd_formal"
RUNS = {
    "fixed_serpentine": [
        "fixed-serpentine-run-1-v5", "fixed-serpentine-run-2-v5", "fixed-serpentine-run-3-v5",
    ],
    "nearest_target_first": [
        "nearest-target-run-1-retry2", "nearest-target-run-2-v4", "nearest-target-run-3-v4",
    ],
    "active_utility": [
        "active-utility-ambiguity-v4-qualification", "active-utility-run-2-v4", "active-utility-run-3-v4",
    ],
}
TELEMETRY = {
    "fixed-serpentine-run-1-v5": ROOT / "data/logs/astar_20260822_025511.csv",
    "fixed-serpentine-run-2-v5": ROOT / "data/logs/astar_20260822_030006.csv",
    "fixed-serpentine-run-3-v5": ROOT / "data/logs/astar_20260822_030458.csv",
    "nearest-target-run-1-retry2": ROOT / "data/logs/astar_20260822_041755.csv",
}


def identity(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def telemetry_metrics(console_text, run_id):
    match = re.search(r"Telemetry log saved to (.+\.csv)", console_text)
    path = TELEMETRY.get(run_id)
    if match:
        path = Path(match.group(1).strip())
    if path is None:
        return {"distance_m": None, "duration_s": None}
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        return {"distance_m": 0.0, "duration_s": 0.0}
    def pick(row, names):
        for name in names:
            if name in row and row[name] not in (None, ""):
                return float(row[name])
        return None
    points = []
    times = []
    for row in rows:
        east = pick(row, ("east_m", "position_east_m", "local_east_m"))
        north = pick(row, ("north_m", "position_north_m", "local_north_m"))
        stamp = pick(row, ("elapsed_s", "timestamp_s", "time_s"))
        if east is not None and north is not None:
            points.append((east, north))
        if stamp is not None:
            times.append(stamp)
    distance = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    duration = max(times) - min(times) if times else None
    return {"distance_m": round(distance, 6), "duration_s": duration}


def aggregate_run(strategy, name):
    prearm = json.loads((OUT / f"{name}-prearm.json").read_text())
    event_rows = events(OUT / f"{name}-events.jsonl")
    console_path = OUT / f"{name}-console.log"
    console = console_path.read_text() if console_path.exists() else ""
    receipt = json.loads((ROOT / "outputs/sandbox/active_rgbd_runtime" / f"{name}-events" / "rgb-depth-runtime-receipt.json").read_text())
    names = [row.get("event") or row.get("event_type") for row in event_rows]
    metrics = telemetry_metrics(console, name)
    completed_event = "mission_completed" in names
    landed_event = "landing_confirmed" in names
    return {
        "run_id": name,
        "strategy": strategy,
        "prearm_ready": (
            prearm.get("acceptance", {}).get("status", prearm.get("status")) == "ready"
            and not prearm.get("acceptance", {}).get("blocked_reasons", prearm.get("blocked_reasons", []))
        ),
        "process_completed": completed_event or ("Done." in console and "Flight error:" not in console),
        "landing_confirmed": landed_event or "Drone has landed." in console,
        "safety_failed": "DangerObstacleDetected" in console or "Flight error:" in console or "mission_failed" in names,
        "pairing_rate": receipt["pairing"].get("rgb_pairing_success_rate"),
        "skew_p95_ms": receipt["pairing"].get("skew_ms", {}).get("p95"),
        "registered_targets": sum(name == "target_registered" for name in names),
        "completed_targets": sum(name in {"target_completed", "target_inspected"} for name in names),
        "classes": sorted({track.get("class_name") for track in receipt.get("tracks", []) if track.get("class_name")}),
        **metrics,
    }


def main():
    runs = [aggregate_run(strategy, name) for strategy, names in RUNS.items() for name in names]
    grouped = {}
    for strategy in RUNS:
        selected = [run for run in runs if run["strategy"] == strategy]
        distances = [run["distance_m"] for run in selected if run["distance_m"] is not None]
        grouped[strategy] = {
            "run_count": len(selected),
            "all_safe_and_landed": all(run["prearm_ready"] and run["process_completed"] and run["landing_confirmed"] and not run["safety_failed"] for run in selected),
            "mean_distance_m": round(sum(distances) / len(distances), 6) if distances else None,
            "mean_registered_targets": round(sum(run["registered_targets"] for run in selected) / len(selected), 3),
            "mean_completed_targets": round(sum(run["completed_targets"] for run in selected) / len(selected), 3),
        }
    fixed, active = grouped["fixed_serpentine"], grouped["active_utility"]
    improvement = None
    if fixed["mean_distance_m"] and active["mean_distance_m"] is not None:
        improvement = (fixed["mean_distance_m"] - active["mean_distance_m"]) / fixed["mean_distance_m"]
    checks = {
        "nine_runs_present": len(runs) == 9,
        "all_prearm_ready_safe_and_landed": all(value["all_safe_and_landed"] for value in grouped.values()),
        "pairing_rate": all((run["pairing_rate"] or 0) >= .95 for run in runs),
        "skew_p95": all(run["skew_p95_ms"] is not None and run["skew_p95_ms"] <= 33.334 for run in runs),
        "efficiency_improvement": improvement is not None and improvement >= .15,
    }
    report = {"schema_version": 1, "world": "substation_complex", "evidence_level": "gazebo_live_rgbd", "runs": runs, "strategies": grouped,
              "active_distance_improvement": improvement, "checks": checks,
              "status": "passed" if all(checks.values()) else "blocked"}
    report["artifact_identity"] = identity(report)
    output = OUT / "complex-active-matrix-report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "checks": checks,
        "active_distance_improvement": improvement,
        "strategy_distance_m": {key: value["mean_distance_m"] for key, value in grouped.items()},
        "strategy_target_counts": {key: {"registered": value["mean_registered_targets"], "completed": value["mean_completed_targets"]} for key, value in grouped.items()},
        "failed_runs": [run["run_id"] for run in runs if not (
            run["prearm_ready"] and run["process_completed"] and run["landing_confirmed"] and not run["safety_failed"]
        )],
        "artifact_identity": report["artifact_identity"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
