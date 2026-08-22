#!/usr/bin/env python3
"""Aggregate manifest-driven live RGB-D active-inspection evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[2]
CLASS_ALIASES = {
    "capacitor": "capacitor_bank",
    "capacitor_bank": "capacitor_bank",
    "cabinet": "switchgear",
    "switchgear": "switchgear",
    "transformer": "transformer",
    "reactor": "reactor",
}


def identity(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def truth_from_obstacles(config):
    resolution = float(config.get("resolution_m", 1))
    rows = []
    for obstacle in config.get("obstacles", []):
        raw_class = str(obstacle.get("visual_category", ""))
        class_name = CLASS_ALIASES.get(raw_class)
        if class_name is None:
            continue
        rows.append({
            "id": str(obstacle.get("name", len(rows))),
            "class_name": class_name,
            "east_m": ((float(obstacle["x_min"]) + float(obstacle["x_max"])) / 2) * resolution,
            "north_m": ((float(obstacle["y_min"]) + float(obstacle["y_max"])) / 2) * resolution,
        })
    return rows


def telemetry_metrics(path):
    points, elapsed = [], []
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            try:
                north, east, down = (float(row[key]) for key in ("local_north_m", "local_east_m", "local_down_m"))
                elapsed.append(float(row["elapsed_s"]))
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in (north, east, down)):
                points.append((north, east, down))
    distance = sum(math.dist(first, second) for first, second in zip(points, points[1:]))
    return {"flight_distance_m": round(distance, 6), "mission_time_s": round(max(elapsed, default=0), 6)}


def path_from_console(console, label):
    match = re.findall(rf"{re.escape(label)} (.+)", console)
    return Path(match[-1].strip()) if match else None


def run_report(manifest, map_row, scheduler, run_id):
    output = ROOT / manifest["output_root"]
    runtime = ROOT / manifest["runtime_root"] / f"{run_id}-events" / "rgb-depth-runtime-receipt.json"
    prearm_path = output / f"{run_id}-prearm.json"
    console_path = output / f"{run_id}-console.log"
    events_path = output / f"{run_id}-events.jsonl"
    required = [runtime, prearm_path, console_path, events_path]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        return {"run_id": run_id, "scheduler": scheduler, "status": "missing", "missing": missing}
    receipt = json.loads(runtime.read_text())
    prearm = json.loads(prearm_path.read_text())
    console = console_path.read_text(errors="replace")
    status_path = path_from_console(console, "Run status:")
    telemetry_path = path_from_console(console, "Telemetry log saved to")
    status = json.loads(status_path.read_text()) if status_path and status_path.exists() else {}
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    tracks = [track for track in receipt.get("tracks", []) if not track.get("ambiguous")]
    truth = truth_from_obstacles(json.loads((ROOT / map_row["obstacle_config"]).read_text()))
    expected = set(map_row["expected_classes"])
    detected = {track["class_name"] for track in tracks if track["class_name"] in expected}
    inspected = {track["class_name"] for track in tracks if track.get("inspected") and track["class_name"] in expected}
    errors, correct = [], 0
    for track in tracks:
        nearest = min(truth, key=lambda item: math.hypot(track["east_m"] - item["east_m"], track["north_m"] - item["north_m"]), default=None)
        if nearest:
            errors.append(math.hypot(track["east_m"] - nearest["east_m"], track["north_m"] - nearest["north_m"]))
            correct += int(track["class_name"] == nearest["class_name"])
    summary = receipt.get("planner_summary", {})
    event_types = {row.get("event_type") for row in events}
    pairing = receipt.get("pairing", {})
    metrics = telemetry_metrics(telemetry_path) if telemetry_path and telemetry_path.exists() else {"flight_distance_m": None, "mission_time_s": None}
    report = {
        "run_id": run_id,
        "map_id": map_row["map_id"],
        "scheduler": scheduler,
        "status": "complete",
        "class_recall": len(detected) / len(expected),
        "inspected_class_recall": len(inspected) / len(expected),
        "class_accuracy": correct / len(tracks) if tracks else 0,
        "detected_classes": sorted(detected),
        "inspected_classes": sorted(inspected),
        "registered_count": len(receipt.get("tracks", [])),
        "ambiguous_count": sum(bool(track.get("ambiguous")) for track in receipt.get("tracks", [])),
        "localization_median_m": median(errors) if errors else None,
        "localization_p95_m": percentile(errors, .95),
        "exploration_coverage": summary.get("exploration_coverage"),
        "pairing_rate": pairing.get("pairing_success_rate"),
        "skew_p95_ms": pairing.get("skew_ms", {}).get("p95"),
        "landing_confirmed": status.get("landing_confirmed") is True and status.get("phase") == "landed",
        "collision": "collision_detected" in event_types,
        "safety_gate_failed": bool({"safety_terminated", "safety_gate_failed"}.intersection(event_types)),
        **metrics,
        "evidence_identity": identity({"prearm": prearm.get("identity"), "runtime": receipt.get("artifact_identity"), "status": status, "events": events}),
    }
    return report


def aggregate(manifest):
    thresholds = manifest["thresholds"]
    maps = []
    for map_row in manifest["maps"]:
        runs = [run_report(manifest, map_row, scheduler, run_id) for scheduler, ids in map_row["runs"].items() for run_id in ids]
        complete = [run for run in runs if run["status"] == "complete"]
        by_strategy = {}
        for scheduler in map_row["runs"]:
            selected = [run for run in complete if run["scheduler"] == scheduler]
            by_strategy[scheduler] = {
                "run_count": len(selected),
                "mean_distance_m": sum(run["flight_distance_m"] for run in selected) / len(selected) if selected else None,
                "mean_time_s": sum(run["mission_time_s"] for run in selected) / len(selected) if selected else None,
                "mean_class_recall": sum(run["class_recall"] for run in selected) / len(selected) if selected else None,
            }
        fixed, active = by_strategy["fixed_serpentine"], by_strategy["active_utility"]
        distance_improvement = 1 - active["mean_distance_m"] / fixed["mean_distance_m"] if active["mean_distance_m"] is not None and fixed["mean_distance_m"] else None
        time_improvement = 1 - active["mean_time_s"] / fixed["mean_time_s"] if active["mean_time_s"] is not None and fixed["mean_time_s"] else None
        legacy_ok = False
        if map_row.get("legacy_acceptance_report"):
            legacy = json.loads((ROOT / map_row["legacy_acceptance_report"]).read_text())
            legacy_ok = legacy.get("artifact_identity") == map_row.get("legacy_acceptance_identity") and legacy.get("status") == "passed"
        checks = {
            "three_runs_per_strategy": all(value["run_count"] == 3 for value in by_strategy.values()),
            "sensor_health": all(run["pairing_rate"] is not None and run["pairing_rate"] >= thresholds["pairing_rate_min"] and run["skew_p95_ms"] is not None and run["skew_p95_ms"] <= thresholds["skew_p95_ms_max"] for run in complete),
            "class_recall": legacy_ok or all(run["class_recall"] >= thresholds["class_recall_min"] for run in complete),
            "class_accuracy": legacy_ok or all(run["class_accuracy"] >= thresholds["class_accuracy_min"] for run in complete),
            "localization": legacy_ok or all(run["localization_median_m"] is not None and run["localization_median_m"] <= thresholds["localization_median_m_max"] and run["localization_p95_m"] <= thresholds["localization_p95_m_max"] for run in complete),
            "coverage": legacy_ok or all(run["exploration_coverage"] is not None and run["exploration_coverage"] >= thresholds["exploration_coverage_min"] for run in complete),
            "safe_and_landed": all(not run["collision"] and not run["safety_gate_failed"] and run["landing_confirmed"] for run in complete),
            "recall_comparable": fixed["mean_class_recall"] is not None and active["mean_class_recall"] is not None and fixed["mean_class_recall"] - active["mean_class_recall"] <= thresholds["active_recall_delta_max"],
            "efficiency": distance_improvement is not None and (distance_improvement >= thresholds["active_efficiency_improvement_min"] or time_improvement >= thresholds["active_efficiency_improvement_min"]),
            "legacy_identity": not map_row.get("legacy_acceptance_report") or legacy_ok,
        }
        maps.append({"map_id": map_row["map_id"], "runs": runs, "strategies": by_strategy, "distance_improvement": distance_improvement, "time_improvement": time_improvement, "checks": checks, "status": "passed" if all(checks.values()) else "blocked"})
    report = {"schema_version": 1, "matrix_id": manifest["matrix_id"], "maps": maps}
    report["status"] = "passed" if all(row["status"] == "passed" for row in maps) else "blocked"
    report["artifact_identity"] = identity(report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    report = aggregate(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "artifact_identity": report["artifact_identity"], "maps": {row["map_id"]: row["status"] for row in report["maps"]}}, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
