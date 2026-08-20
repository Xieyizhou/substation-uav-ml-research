"""Deterministic runtime-blocker scenarios and benchmark acceptance."""

from __future__ import annotations

import json
from pathlib import Path

from src.maps.map_catalog import map_by_id, project_path
from src.ml.artifacts import object_sha256, write_json
from src.planner.astar_grid import astar
from src.planner.obstacle_config import build_obstacle_map, inflate_cells


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / "config/planning/dynamic_replanning_benchmark_v1.json"
PHASE_EVENT_CHAIN = (
    "dynamic_blocker_spawned",
    "dynamic_blocker_detected",
    "dynamic_replan_decided",
    "dynamic_replan_hovered",
    "dynamic_replan_route_ready",
    "dynamic_replan_waypoints_accepted",
    "dynamic_replan_resumed",
    "mission_completed",
)


def load_dynamic_spec(path=DEFAULT_SPEC):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("dynamic_replanning_benchmark_schema_version") != 1:
        raise ValueError("unsupported dynamic replanning benchmark schema")
    if len(value.get("maps", [])) != 3 or len(value.get("injection_phases", [])) != 4:
        raise ValueError("dynamic benchmark requires three maps and four phases")
    return value


def dynamic_replanning_matrix(path=DEFAULT_SPEC):
    spec = load_dynamic_spec(path)
    return [
        {
            "scenario_id": f"dynamic-{map_id}-{phase['id']}-{spec['seed'] + index}",
            "map_id": map_id,
            "target_id": "top_right",
            "seed": spec["seed"] + index,
            "condition": "geometric_lidar",
            "injection_phase": phase["id"],
        }
        for index, (map_id, phase) in enumerate(
            (map_id, phase)
            for map_id in spec["maps"]
            for phase in spec["injection_phases"]
        )
    ]


def _base_route(config):
    obstacle_map = build_obstacle_map(config)
    path = astar(
        tuple(config["start_cell"]), tuple(config["goal_cell"]),
        obstacle_map["inflated_blocking_cells"],
        int(config["width"]), int(config["height"]),
    )
    return path, obstacle_map


def _candidate_indices(path, fraction):
    preferred = round((len(path) - 1) * float(fraction))
    return sorted(range(3, len(path) - 3), key=lambda index: (abs(index - preferred), index))


def _alternative_path(config, obstacle_map, path, index, phase, blocker):
    route_direction = phase["route_direction"]
    lead = int(blocker["trigger_lead_cells"])
    trigger_index = max(0, index - lead) if route_direction == "outbound" else min(len(path) - 1, index + lead)
    target = tuple(config["goal_cell"] if route_direction == "outbound" else config["start_cell"])
    blocked = set(obstacle_map["inflated_blocking_cells"])
    blocked.update(inflate_cells(
        {path[index]}, int(config["width"]), int(config["height"]),
        int(blocker["inflation_cells"]),
    ))
    blocked -= {tuple(config["start_cell"]), tuple(config["goal_cell"])}
    try:
        alternative = astar(
            path[trigger_index], target, blocked,
            int(config["width"]), int(config["height"]),
        )
    except ValueError:
        return None
    return trigger_index, alternative


def materialize_dynamic_scenario(
    map_id, phase_id, output_path, spec_path=DEFAULT_SPEC, planner_path=None
):
    spec = load_dynamic_spec(spec_path)
    phase = next((item for item in spec["injection_phases"] if item["id"] == phase_id), None)
    if phase is None or map_id not in spec["maps"]:
        raise ValueError("unsupported dynamic replanning map or phase")
    entry = map_by_id(map_id)
    source = Path(planner_path) if planner_path is not None else project_path(
        entry["obstacle_config"]
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    path, obstacle_map = _base_route(config)
    selected = None
    for index in _candidate_indices(path, phase["path_fraction"]):
        alternative = _alternative_path(config, obstacle_map, path, index, phase, spec["blocker"])
        if alternative is not None:
            selected = (index, *alternative)
            break
    if selected is None:
        raise ValueError(f"{map_id}/{phase_id} has no safe runtime blocker detour")
    index, trigger_index, alternative = selected
    resolution = float(config.get("resolution_m", 1.0))
    cell, trigger = path[index], path[trigger_index]
    offset_east, offset_north = map(float, entry["spawn_pose"][:2])
    payload = {
        "dynamic_replanning_scenario_schema_version": 1,
        "map_id": map_id,
        "world_name": entry["world_name"],
        "resolution_m": resolution,
        "phase": phase_id,
        "route_direction": phase["route_direction"],
        "blocker": {
            "id": f"dynamic_blocker_{map_id}_{phase_id}",
            "grid_cell": list(cell),
            "east_m": (cell[0] + 0.5) * resolution,
            "north_m": (cell[1] + 0.5) * resolution,
            "world_east_m": offset_east + (cell[0] + 0.5) * resolution,
            "world_north_m": offset_north + (cell[1] + 0.5) * resolution,
            "size_m": float(spec["blocker"]["size_m"]),
            "height_m": float(spec["blocker"]["height_m"]),
        },
        "trigger": {
            "grid_cell": list(trigger),
            "radius_m": float(spec["blocker"]["trigger_radius_m"]),
        },
        "evidence": {
            "original_path_length": len(path),
            "blocker_path_index": index,
            "alternative_path_length": len(alternative),
            "original_route_invalidated": cell in set(path),
            "safe_alternative_exists": bool(alternative),
        },
        "required_event_chain": list(PHASE_EVENT_CHAIN),
    }
    payload["dynamic_replanning_scenario_identity_sha256"] = object_sha256(payload)
    write_json(output_path, payload)
    return payload


def event_chain_report(events, required=PHASE_EVENT_CHAIN):
    names = [event.get("event_type") for event in events]
    cursor, missing = -1, []
    for name in required:
        try:
            cursor = names.index(name, cursor + 1)
        except ValueError:
            missing.append(name)
    return {"complete": not missing, "missing_events": missing, "observed_events": names}


def dynamic_benchmark_acceptance(rows, spec_path=DEFAULT_SPEC):
    spec = load_dynamic_spec(spec_path)
    limits = spec["acceptance"]
    count = len(rows)
    if count != 12:
        return {"passed": False, "reasons": [f"expected 12 runs, found {count}"]}
    rate = lambda name: sum(float(row.get(name, 0)) for row in rows) / count
    totals = lambda name: sum(float(row.get(name, 0)) for row in rows)
    checks = {
        "successful_replan_rate": rate("successful_replan"),
        "route_switch_correctness": rate("route_switch_correct"),
        "mission_completion_rate": rate("mission_success"),
        "landing_success_rate": rate("landing_success"),
        "false_replan_rate": rate("false_replan"),
        "collision_count": totals("collision_count"),
        "safety_failure_count": totals("safety_failure_count"),
        "event_chain_completion_rate": rate("event_chain_complete"),
    }
    reasons = []
    for name in ("successful_replan_rate", "route_switch_correctness", "mission_completion_rate", "landing_success_rate"):
        if checks[name] < limits[f"{name}_min"]:
            reasons.append(f"{name} below minimum")
    if checks["false_replan_rate"] > limits["false_replan_rate_max"]:
        reasons.append("false_replan_rate above maximum")
    for name in ("collision_count", "safety_failure_count"):
        if checks[name] > limits[f"{name}_max"]:
            reasons.append(f"{name} above maximum")
    if checks["event_chain_completion_rate"] < 1.0:
        reasons.append("one or more runs lack the required blocker/replan event chain")
    return {"passed": not reasons, "reasons": reasons, "metrics": checks}
