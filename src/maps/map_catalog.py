"""Load the map catalog and persist a project-local active map selection."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = PROJECT_ROOT / "config" / "maps" / "catalog.json"
STATE_PATH = PROJECT_ROOT / ".runtime" / "selected_map.json"
PX4_PID_PATH = PROJECT_ROOT / ".runtime" / "px4_launcher.pid"
MAP_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
REQUIRED_FIELDS = {
    "id",
    "display_name",
    "difficulty",
    "difficulty_label",
    "description",
    "world_name",
    "world_file",
    "obstacle_config",
    "spawn_pose",
    "default_target",
    "targets",
}


class MapCatalogError(ValueError):
    """Raised when the map catalog or active selection is invalid."""


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as error:
        raise MapCatalogError(f"Map catalog file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise MapCatalogError(f"Invalid JSON in {path}: {error}") from error


def _validate_project_relative_path(value, field, entry_id):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise MapCatalogError(
            f"Map {entry_id!r} field {field!r} must be a project-relative path"
        )
    resolved = (PROJECT_ROOT / path).resolve()
    if PROJECT_ROOT not in resolved.parents:
        raise MapCatalogError(f"Map {entry_id!r} path escapes the project: {value}")
    return resolved


def _validate_entry(entry, seen_ids, validate_paths):
    if not isinstance(entry, dict):
        raise MapCatalogError("Every map catalog entry must be an object")
    missing = sorted(REQUIRED_FIELDS - set(entry))
    if missing:
        raise MapCatalogError(f"Map entry is missing fields: {missing}")

    entry_id = entry["id"]
    if not isinstance(entry_id, str) or not MAP_ID_PATTERN.fullmatch(entry_id):
        raise MapCatalogError(f"Invalid map id: {entry_id!r}")
    if entry_id in seen_ids:
        raise MapCatalogError(f"Duplicate map id: {entry_id}")
    seen_ids.add(entry_id)

    if not isinstance(entry["difficulty"], int) or not 1 <= entry["difficulty"] <= 5:
        raise MapCatalogError(f"Map {entry_id!r} difficulty must be an integer from 1 to 5")
    spawn_pose = entry["spawn_pose"]
    if not isinstance(spawn_pose, list) or len(spawn_pose) != 6:
        raise MapCatalogError(f"Map {entry_id!r} spawn_pose must contain six numbers")
    try:
        entry["spawn_pose"] = [float(value) for value in spawn_pose]
    except (TypeError, ValueError) as error:
        raise MapCatalogError(f"Map {entry_id!r} spawn_pose must contain numbers") from error

    for field in ("world_file", "obstacle_config"):
        resolved = _validate_project_relative_path(entry[field], field, entry_id)
        if validate_paths and not resolved.is_file():
            raise MapCatalogError(f"Map {entry_id!r} file not found: {resolved}")

    targets = entry["targets"]
    if not isinstance(targets, list) or not targets:
        raise MapCatalogError(f"Map {entry_id!r} targets must be a non-empty list")
    target_ids = set()
    for target in targets:
        if not isinstance(target, dict):
            raise MapCatalogError(f"Map {entry_id!r} target entries must be objects")
        missing_target_fields = {
            "id",
            "display_name",
            "description",
            "cell",
        } - set(target)
        if missing_target_fields:
            raise MapCatalogError(
                f"Map {entry_id!r} target is missing fields: "
                f"{sorted(missing_target_fields)}"
            )
        target_id = target["id"]
        if not isinstance(target_id, str) or not MAP_ID_PATTERN.fullmatch(target_id):
            raise MapCatalogError(
                f"Map {entry_id!r} has invalid target id: {target_id!r}"
            )
        if target_id in target_ids:
            raise MapCatalogError(
                f"Map {entry_id!r} has duplicate target id: {target_id}"
            )
        target_ids.add(target_id)
        cell = target["cell"]
        if (
            not isinstance(cell, list)
            or len(cell) != 2
            or any(not isinstance(value, int) for value in cell)
        ):
            raise MapCatalogError(
                f"Map {entry_id!r} target {target_id!r} cell must contain two integers"
            )
    if entry["default_target"] not in target_ids:
        raise MapCatalogError(
            f"Map {entry_id!r} default_target must refer to one of its targets"
        )


def load_catalog(path: Path = CATALOG_PATH, validate_paths: bool = True):
    """Return a validated catalog dictionary."""
    catalog = _read_json(Path(path))
    if not isinstance(catalog, dict) or not isinstance(catalog.get("maps"), list):
        raise MapCatalogError("Map catalog must contain a 'maps' list")
    if not catalog["maps"]:
        raise MapCatalogError("Map catalog cannot be empty")

    seen_ids = set()
    for entry in catalog["maps"]:
        _validate_entry(entry, seen_ids, validate_paths)
    if catalog.get("default_map") not in seen_ids:
        raise MapCatalogError("default_map must refer to a catalog entry")
    return catalog


def list_maps(catalog_path: Path = CATALOG_PATH):
    catalog = load_catalog(catalog_path)
    return sorted(catalog["maps"], key=lambda entry: (entry["difficulty"], entry["id"]))


def map_by_id(map_id, catalog_path: Path = CATALOG_PATH):
    catalog = load_catalog(catalog_path)
    for entry in catalog["maps"]:
        if entry["id"] == map_id:
            return entry
    available = ", ".join(entry["id"] for entry in catalog["maps"])
    raise MapCatalogError(f"Unknown map {map_id!r}. Available maps: {available}")


def selected_map_id(
    catalog_path: Path = CATALOG_PATH,
    state_path: Path = STATE_PATH,
):
    catalog = load_catalog(catalog_path)
    state_path = Path(state_path)
    if not state_path.exists():
        return catalog["default_map"]
    try:
        state = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return catalog["default_map"]
    selected = state.get("map_id") if isinstance(state, dict) else None
    valid_ids = {entry["id"] for entry in catalog["maps"]}
    return selected if selected in valid_ids else catalog["default_map"]


def current_map(
    catalog_path: Path = CATALOG_PATH,
    state_path: Path = STATE_PATH,
):
    return map_by_id(selected_map_id(catalog_path, state_path), catalog_path)


def select_map(
    map_id,
    catalog_path: Path = CATALOG_PATH,
    state_path: Path = STATE_PATH,
):
    """Atomically persist and return the selected map entry."""
    entry = map_by_id(map_id, catalog_path)
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "map_id": entry["id"],
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary_path = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
    temporary_path.replace(state_path)
    return entry


def project_path(relative_path):
    return (PROJECT_ROOT / relative_path).resolve()


def spawn_pose_text(entry):
    return ",".join(f"{float(value):g}" for value in entry["spawn_pose"])


def field_value(entry, field):
    if field not in entry:
        available = ", ".join(sorted(entry))
        raise MapCatalogError(f"Unknown field {field!r}. Available fields: {available}")
    value = entry[field]
    if field == "spawn_pose":
        return spawn_pose_text(entry)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def px4_launcher_pid(pid_path: Path = PX4_PID_PATH):
    """Return the live project-managed PX4 launcher PID, otherwise None."""
    pid_path = Path(pid_path)
    try:
        value = pid_path.read_text().strip()
        pid = int(value)
    except (FileNotFoundError, OSError, ValueError):
        return None
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        # A permissions failure still proves that the PID exists. Treat it as
        # live so map switching fails closed instead of risking a world/config
        # mismatch in a restricted shell.
        return pid
    return pid
