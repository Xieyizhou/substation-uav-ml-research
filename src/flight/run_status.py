"""Atomic machine-readable flight outcome records."""

import json
from datetime import datetime, timezone

from src.flight.flight_config import LOG_DIR, display_path


def status_path_for_log(log_path):
    return log_path.with_suffix(".status.json")


def write_run_status(log_path, status, phase, message="", landing_confirmed=None):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    status_path = status_path_for_log(log_path)
    payload = {
        "run_id": log_path.stem.replace("astar_", "as_", 1),
        "status": status,
        "phase": phase,
        "message": message,
        "landing_confirmed": landing_confirmed,
        "log_path": str(display_path(log_path)),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary_path = status_path.with_suffix(status_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
    temporary_path.replace(status_path)
    return status_path
