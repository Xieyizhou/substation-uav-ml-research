"""Persist and resolve per-map A* destination target selections."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.maps.map_catalog import (
    PROJECT_ROOT,
    current_map,
    list_maps,
    map_by_id,
    project_path,
)


TARGET_STATE_PATH = PROJECT_ROOT / ".runtime" / "selected_targets.json"
FLIGHT_PID_PATH = PROJECT_ROOT / ".runtime" / "flight.pid"


class TargetCatalogError(ValueError):
    """Raised when a target selection or target catalog entry is invalid."""


def targets_for_map(entry=None):
    """Return the target presets for a map, preserving catalog order."""
    entry = current_map() if entry is None else entry
    return list(entry["targets"])


def target_by_id(entry, target_id):
    """Return one target preset belonging to the supplied map entry."""
    for target in targets_for_map(entry):
        if target["id"] == target_id:
            return target
    available = ", ".join(target["id"] for target in targets_for_map(entry))
    raise TargetCatalogError(
        f"Unknown target {target_id!r} for map {entry['id']!r}. "
        f"Available targets: {available}"
    )


def _load_state(state_path):
    state_path = Path(state_path)
    if not state_path.exists():
        return {}
    try:
        state = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    selections = state.get("selected_targets") if isinstance(state, dict) else None
    return selections if isinstance(selections, dict) else {}


def selected_target_id(entry=None, state_path: Path = TARGET_STATE_PATH):
    """Return the saved target ID for a map or its configured default."""
    entry = current_map() if entry is None else entry
    selected = _load_state(state_path).get(entry["id"])
    valid_ids = {target["id"] for target in targets_for_map(entry)}
    return selected if selected in valid_ids else entry["default_target"]


def current_target(entry=None, state_path: Path = TARGET_STATE_PATH):
    """Return the selected target preset for a map."""
    entry = current_map() if entry is None else entry
    return target_by_id(entry, selected_target_id(entry, state_path))


def select_target(
    target_id,
    map_id=None,
    state_path: Path = TARGET_STATE_PATH,
):
    """Atomically persist one target selection for one map."""
    entry = current_map() if map_id is None else map_by_id(map_id)
    target = target_by_id(entry, target_id)
    state_path = Path(state_path)
    selections = _load_state(state_path)
    selections[entry["id"]] = target["id"]
    payload = {
        "selected_targets": selections,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
    temporary_path.replace(state_path)
    return entry, target


def map_for_obstacle_config(config_path):
    """Return the catalog map paired with a config path, otherwise None."""
    resolved = Path(config_path).resolve()
    for entry in list_maps():
        if project_path(entry["obstacle_config"]) == resolved:
            return entry
    return None


def selected_target_for_config(config_path, state_path: Path = TARGET_STATE_PATH):
    """Resolve the selected target when a config belongs to a catalog map."""
    entry = map_for_obstacle_config(config_path)
    if entry is None:
        return None
    return current_target(entry, state_path)


def target_field_value(target, field):
    """Return one target field in a shell-friendly representation."""
    if field not in target:
        available = ", ".join(sorted(target))
        raise TargetCatalogError(
            f"Unknown target field {field!r}. Available fields: {available}"
        )
    value = target[field]
    if field == "cell":
        return ",".join(str(int(item)) for item in value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def flight_runner_pid(pid_path: Path = FLIGHT_PID_PATH):
    """Return the live project-managed flight runner PID, otherwise None."""
    try:
        pid = int(Path(pid_path).read_text().strip())
    except (FileNotFoundError, OSError, ValueError):
        return None
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid
    return pid
