"""Deterministic qualification obstacles for active-replanning coverage."""

from __future__ import annotations

import json
from pathlib import Path

from src.maps.map_catalog import project_path
from src.ml.artifacts import object_sha256, write_json
from src.ml.domain_randomization import materialize_planner_config, materialize_world
from src.planner.astar_grid import astar
from src.planner.obstacle_config import build_obstacle_map, inflate_cells


UNMAPPED_ROUTE_BLOCKER_V1 = "unmapped_route_blocker_v1"
CONTROLLED_SENSOR_DEFAULTS = {
    "lidar_noise_stddev_m": 0.0,
    "lidar_dropout_probability": 0.0,
    "sensor_outage_probability": 0.0,
    "sensor_outage_duration_s": 0.0,
}


def _route(config):
    obstacle_map = build_obstacle_map(config)
    return astar(
        tuple(config["start_cell"]),
        tuple(config["goal_cell"]),
        obstacle_map["inflated_blocking_cells"],
        int(config["width"]),
        int(config["height"]),
    ), obstacle_map


def _detour_exists(config, obstacle_map, cell):
    blocked = set(obstacle_map["inflated_blocking_cells"])
    blocked.update(
        inflate_cells(
            {cell}, int(config["width"]), int(config["height"]), 1
        )
    )
    try:
        astar(
            tuple(config["start_cell"]), tuple(config["goal_cell"]), blocked,
            int(config["width"]), int(config["height"]),
        )
    except ValueError:
        return False
    return True


def route_blocker_specification(planner_path):
    """Place a simulator-only obstacle on a reachable route segment."""
    config = json.loads(Path(planner_path).read_text(encoding="utf-8"))
    path, obstacle_map = _route(config)
    lower, upper = max(3, len(path) // 4), min(len(path) - 3, len(path) * 3 // 4)
    candidates = path[lower:upper]
    cell = next(
        (candidate for candidate in candidates if _detour_exists(config, obstacle_map, candidate)),
        None,
    )
    if cell is None:
        raise ValueError("qualification route has no safe detour around a blocker")
    resolution = float(config.get("resolution_m", 1.0))
    return {
        "id": "qualification_route_blocker",
        "east_m": (cell[0] + 0.5) * resolution,
        "north_m": (cell[1] + 0.5) * resolution,
        "size_m": min(0.7, resolution * 0.7),
        "height_m": 2.2,
        "grid_cell": list(cell),
        "profile": UNMAPPED_ROUTE_BLOCKER_V1,
    }


def _stabilize_challenge_manifest(manifest):
    changed = bool(manifest.get("unknown_obstacles"))
    manifest["unknown_obstacles"] = []
    for name, value in CONTROLLED_SENSOR_DEFAULTS.items():
        changed = changed or float(manifest.get(name, value)) != value
        manifest[name] = value
    return changed


def apply_scenario_profile(manifest, planner_path, profile):
    """Add a profile once; return true when rematerialization is required."""
    if not profile:
        return False
    if profile != UNMAPPED_ROUTE_BLOCKER_V1:
        raise ValueError(f"unsupported qualification scenario profile: {profile}")
    if _stabilize_challenge_manifest(manifest):
        return True
    if manifest.get("unmapped_obstacles"):
        return False
    manifest["unmapped_obstacles"] = [route_blocker_specification(planner_path)]
    manifest["expected_capabilities"] = [
        "dynamic_threat_detection",
        "local_replan_attempt",
        "local_replan_success",
        "active_route_replacement",
        "safe_mission_completion",
    ]
    return True


def _set_target_and_validate(planner_path, entry, target_id):
    planner_path = Path(planner_path)
    config = json.loads(planner_path.read_text(encoding="utf-8"))
    target = next(
        (item for item in entry["targets"] if item["id"] == target_id), None
    )
    if target is None:
        raise ValueError(f"{entry['id']} has no target {target_id!r}")
    config["goal_cell"] = list(target["cell"])
    path, _ = _route(config)
    if not path:
        raise ValueError("qualification route is empty")
    write_json(planner_path, config)


def _rehash_manifest(manifest):
    value = {key: item for key, item in manifest.items() if key != "config_hash"}
    manifest["config_hash"] = object_sha256(value)


def materialize_reachable_scenario(
    manifest, entry, target_id, world_path, scenario_manifest, oracle_planner,
    scenario_profile=None,
):
    """Materialize a reachable world, optionally with a qualification profile."""
    adjustments = manifest.setdefault("feasibility_adjustments", [])
    relaxed_equipment = False
    while True:
        _rehash_manifest(manifest)
        materialize_world(
            project_path(entry["world_file"]), world_path, manifest,
            report_path=scenario_manifest,
        )
        report = json.loads(Path(scenario_manifest).read_text(encoding="utf-8"))
        materialize_planner_config(
            project_path(entry["obstacle_config"]), oracle_planner, report
        )
        try:
            _set_target_and_validate(oracle_planner, entry, target_id)
            if apply_scenario_profile(manifest, oracle_planner, scenario_profile):
                marker = f"applied_profile:{scenario_profile}"
                if marker not in adjustments:
                    adjustments.append(marker)
                continue
            return
        except ValueError:
            if manifest["unknown_obstacles"]:
                removed = manifest["unknown_obstacles"].pop()
                adjustments.append(f"removed_unreachable:{removed['id']}")
                continue
            if not relaxed_equipment:
                manifest["equipment_position_jitter_m"] = 0.0
                manifest["equipment_scale"] = 1.0
                adjustments.append("restored_baseline_equipment_geometry")
                relaxed_equipment = True
                continue
            raise
