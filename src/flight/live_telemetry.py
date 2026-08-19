"""Run-local telemetry snapshots for the desktop flight console."""

from __future__ import annotations

import json
import os
from pathlib import Path


def configured_snapshot_path():
    value = os.environ.get("UAV_LIVE_TELEMETRY_PATH", "").strip()
    return Path(value) if value else None


def _value(message, name):
    if message is None:
        return None
    value = getattr(message, name, None)
    return round(float(value), 6) if value is not None else None


def snapshot_record(now, start_time, phase, target, position, velocity, attitude, latest):
    return {
        "snapshot_schema_version": 1,
        "timestamp_utc": now.isoformat(),
        "elapsed_s": round((now - start_time).total_seconds(), 3),
        "phase": phase.get("phase", "unknown"),
        "route_direction": phase.get("route_direction", "none"),
        "position": {
            "north_m": _value(position, "north_m"),
            "east_m": _value(position, "east_m"),
            "down_m": _value(position, "down_m"),
        },
        "velocity": {
            "north_m_s": _value(velocity, "north_m_s"),
            "east_m_s": _value(velocity, "east_m_s"),
            "down_m_s": _value(velocity, "down_m_s"),
        },
        "yaw_deg": _value(attitude, "yaw_deg"),
        "target": {
            "name": target.get("name", ""),
            "north_m": target.get("north_m"),
            "east_m": target.get("east_m"),
            "down_m": target.get("down_m"),
        },
        "connected": latest.get("connected"),
        "armed": latest.get("armed"),
        "in_air": latest.get("in_air"),
    }


def write_snapshot(path, record):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
