"""Log path resolution and dataframe preparation helpers for A* analysis."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "data" / "logs"


def find_newest_astar_log():
    csv_files = list(LOG_DIR.glob("astar_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No astar_*.csv files found in {LOG_DIR}")
    return max(csv_files, key=lambda path: path.stat().st_mtime)


def resolve_log_path(log_arg):
    if log_arg is None:
        return find_newest_astar_log(), True
    return log_arg.expanduser().resolve(), False


def resolve_project_path(path):
    if path is None:
        return None
    path = path.expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def run_id_from_log(log_path):
    if log_path.stem.startswith("astar_"):
        return "as_" + log_path.stem.removeprefix("astar_")
    return log_path.stem


def display_path(path):
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path.resolve()


def prepare_dataframe(log_path):
    df = pd.read_csv(log_path)
    if "elapsed_s" not in df.columns:
        raise KeyError("CSV file does not contain an elapsed_s column")

    numeric_columns = [
        "elapsed_s",
        "target_north_m",
        "target_east_m",
        "target_down_m",
        "local_north_m",
        "local_east_m",
        "local_down_m",
        "error_north_m",
        "error_east_m",
        "error_down_m",
        "horizontal_error_m",
        "velocity_north_m_s",
        "velocity_east_m_s",
        "velocity_down_m_s",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "battery_percent",
        "grid_width",
        "grid_height",
        "resolution_m",
        "altitude_m",
        "nearest_obstacle_distance_m",
        "nearest_obstacle_bearing_deg",
        "detected_obstacle_count",
        "detection_range_m",
        "detection_fov_deg",
        "warning_distance_m",
        "danger_distance_m",
        "replan_start_grid_x",
        "replan_start_grid_y",
        "replan_goal_grid_x",
        "replan_goal_grid_y",
        "replan_path_length",
        "dynamic_blocked_cell_count",
        "replan_count",
        "active_replan_count",
        "active_replan_path_length",
        "sensor_frequency_hz",
        "sensor_frame_age_s",
        "sensor_dropped_frames",
        "inference_latency_ms",
        "costmap_version",
        "equipment_detection_count",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in [
        "phase",
        "route_direction",
        "target_name",
        "planner_name",
        "map_name",
        "perception_enabled",
        "detected_obstacle",
        "perception_risk_level",
        "nearest_obstacle_name",
        "nearest_obstacle_layer",
        "risk_action",
        "replan_mode",
        "replan_triggered",
        "replan_success",
        "replan_route_replaced",
        "perception_source",
        "sensor_healthy",
        "risk_model_id",
        "equipment_model_id",
        "equipment_classes",
    ]:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str)

    if "local_down_m" in df.columns:
        df["actual_altitude_m"] = -df["local_down_m"]

    return df
