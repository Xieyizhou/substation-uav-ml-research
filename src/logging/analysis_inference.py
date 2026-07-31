"""Experiment classification and obstacle-map inference for log analysis."""

from pathlib import Path

from src.planner.obstacle_config import build_obstacle_map, get_resolution_altitude, load_obstacle_config
from src.logging.log_io import display_path, resolve_project_path
from src.logging.metrics import (
    bool_from_value,
    first_active_value,
    first_value,
    perception_enabled_mask,
    safe_max,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def infer_return_home_enabled(df):
    configured = first_value(df, "return_home_enabled")
    if configured is not None:
        return bool_from_value(configured)
    if "route_direction" in df.columns:
        return bool((df["route_direction"] == "return").any())
    return None


def infer_local_replan_enabled(df):
    if "replan_count" in df.columns and safe_max(df, "replan_count"):
        return True
    if "replan_triggered" in df.columns and df["replan_triggered"].map(bool_from_value).fillna(False).any():
        return True
    if "replan_route_replaced" in df.columns and df["replan_route_replaced"].map(bool_from_value).fillna(False).any():
        return True
    if "active_replan_count" in df.columns and safe_max(df, "active_replan_count"):
        return True
    return False


def infer_experiment_stage(df):
    perception_enabled = bool(perception_enabled_mask(df).any()) if "perception_enabled" in df.columns else False
    local_replan_enabled = infer_local_replan_enabled(df)
    replan_mode = str(first_value(df, "replan_mode") or "log_only").strip().lower()
    if local_replan_enabled and replan_mode == "active":
        return "active_replan"
    if local_replan_enabled and replan_mode == "log_only":
        return "replan_log_only"
    if perception_enabled:
        return "perception_response"
    return "static_astar"


def infer_experiment_type(df, stage):
    risk_action = first_active_value(df, "risk_action") or first_value(df, "risk_action")
    if stage == "static_astar":
        return "astar_only_perception_disabled"
    if stage == "active_replan":
        return "active_local_replan"
    if stage == "replan_log_only":
        return "local_replan_log_only"
    if risk_action == "slow_down":
        return "perception_slow_down"
    if risk_action == "log_only":
        return "perception_log_only"
    return "perception_response"


def load_analysis_obstacles(args, df, warnings):
    config_path = resolve_project_path(args.obstacle_config)
    if config_path is None:
        inferred_path = PROJECT_ROOT / "config" / "substation_obstacles.json"
        map_name = first_value(df, "map_name")
        if inferred_path.exists() and map_name:
            try:
                inferred_config = load_obstacle_config(inferred_path)
                if inferred_config.get("map_name") == map_name:
                    config_path = inferred_path
                    print(f"Inferred obstacle config: {display_path(config_path)}")
            except Exception as error:
                warnings.append(f"Could not infer obstacle config: {error}")

    if config_path is None:
        warnings.append("No obstacle config provided; obstacle avoidance could not be validated in analysis.")
        return None, None, set(), None, None

    config = load_obstacle_config(config_path)
    log_altitude_m = first_value(df, "altitude_m")
    _, config_altitude_m = get_resolution_altitude(config)
    altitude_m = float(log_altitude_m) if log_altitude_m is not None else config_altitude_m
    obstacle_map = build_obstacle_map(config, flight_altitude_m=altitude_m)
    obstacles = obstacle_map["inflated_blocking_cells"]
    resolution_m, _ = get_resolution_altitude(config)
    return config_path, config, obstacles, resolution_m, obstacle_map
