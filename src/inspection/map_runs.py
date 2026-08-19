"""Read-only map-run and trajectory presentation."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from src.maps.sandbox_contracts import SandboxMap
from src.maps.sandbox_store import SandboxMapStore
from src.sandbox.map_recording_dataset import audit_sandbox_recording


def _json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _run_root(config, record):
    if not record or not record.get("run_root"):
        return None
    root = Path(record["run_root"]).resolve()
    root.relative_to(config.sandbox_map_runs_root.resolve())
    return root


def _trajectory(path, map_value, limit=1200):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    step = max(1, len(rows) // limit)
    points = []
    for row in rows[::step]:
        try:
            points.append({
                "east_m": round(map_value.start_east_m + float(row["local_east_m"]), 4),
                "north_m": round(map_value.start_north_m + float(row["local_north_m"]), 4),
                "yaw_deg": float(row["yaw_deg"]),
                "elapsed_s": float(row["elapsed_s"]),
                "phase": row["phase"],
            })
        except (KeyError, TypeError, ValueError):
            continue
    return points


def _live(record, map_value):
    if not record:
        return None
    value = dict(record)
    position = dict(value.get("position") or {})
    if position.get("east_m") is not None:
        position["east_m"] += map_value.start_east_m
    if position.get("north_m") is not None:
        position["north_m"] += map_value.start_north_m
    value["position"] = position
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(
            value["timestamp_utc"]
        )).total_seconds()
    except (KeyError, TypeError, ValueError):
        age = None
    value["age_s"] = round(age, 3) if age is not None else None
    value["stale"] = age is None or age > 2.0
    return value


def _present_run(config, root, summary, *, active):
    if root is None:
        return {"active": False, "latest": None, "history": []}
    store = SandboxMapStore(config.sandbox_maps_root)
    revision = store.revision_root(summary["map_id"], summary["revision_id"])
    map_value = SandboxMap.from_record(_json(revision / "map.json"))
    route = _json(revision / "routes" / f"{summary['mission_id']}.json")
    history = []
    for path in sorted(config.sandbox_map_runs_root.glob("*/summary.json"), reverse=True)[:20]:
        value = _json(path)
        if value:
            history.append(value)
    recording_audit = None
    recording = root / "recording"
    if recording.is_dir():
        try:
            recording_audit = audit_sandbox_recording(recording)
        except (OSError, TypeError, ValueError) as error:
            recording_audit = {"valid": False, "error": str(error)}
    return {
        "active": active,
        "latest": summary,
        "map": map_value.to_record(),
        "route": route,
        "live": _live(_json(root / "live_telemetry.json"), map_value),
        "trajectory": _trajectory(root / "telemetry.csv", map_value),
        "recording_audit": recording_audit,
        "history": history,
    }


def map_runs_summary(config):
    pointer = _json(config.sandbox_map_runs_root / "active.json")
    latest = pointer or _json(config.sandbox_map_runs_root / "latest.json")
    root = _run_root(config, latest)
    summary = _json(root / "summary.json", latest) if root else None
    return _present_run(config, root, summary, active=pointer is not None)


def inspect_map_run(config, run_id):
    root = config.sandbox_map_run(run_id)
    summary = _json(root / "summary.json")
    if not summary:
        raise ValueError("sandbox map run was not found")
    return _present_run(config, root, summary, active=False)


def register_map_run(config, run_id, dataset_id=None):
    root = config.sandbox_map_run(run_id)
    summary = _json(root / "summary.json")
    if not summary or summary.get("state") != "complete":
        raise ValueError("only a complete custom-map run can be registered")
    recording = root / "recording"
    if dataset_id is None:
        dataset_id = f"sandbox-{summary['recording_id'][-40:]}"
    from src.sandbox.map_recording_dataset import register_sandbox_recording

    return register_sandbox_recording(
        recording, config.workbench_datasets_root, dataset_id
    )
