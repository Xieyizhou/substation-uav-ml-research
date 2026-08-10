"""CSV telemetry formatting for A* flight experiments.

The flight loop samples MAVSDK telemetry, perception state, and local replan
state. This module defines the stable CSV schema and converts runtime objects
into rows for later analysis by `python main.py report analyze` and the
stage summary tools.
"""

import json


TELEMETRY_CSV_HEADER = [
    "timestamp_utc",
    "elapsed_s",
    "phase",
    "route_direction",
    "target_name",
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
    "flight_mode",
    "is_armed",
    "grid_start",
    "grid_goal",
    "grid_width",
    "grid_height",
    "planner_name",
    "map_name",
    "resolution_m",
    "altitude_m",
    "return_home_enabled",
    "perception_enabled",
    "detected_obstacle",
    "perception_risk_level",
    "nearest_obstacle_name",
    "nearest_obstacle_layer",
    "nearest_obstacle_distance_m",
    "nearest_obstacle_bearing_deg",
    "detected_obstacle_count",
    "detection_range_m",
    "detection_fov_deg",
    "warning_distance_m",
    "danger_distance_m",
    "risk_action",
    "perception_risk_value",
    "perception_closest_obstacle",
    "perception_distance_m",
    "perception_in_detection_range",
    "perception_in_fov",
    "perception_suggested_action",
    "perception_reason",
    "perception_source",
    "sensor_healthy",
    "sensor_frequency_hz",
    "sensor_frame_age_s",
    "sensor_dropped_frames",
    "risk_model_id",
    "inference_latency_ms",
    "model_inference_latency_ms",
    "truth_risk_level",
    "geometric_risk_level",
    "ml_risk_level",
    "risk_confidence",
    "truth_traversability_json",
    "predicted_traversability_json",
    "truth_direction_deg",
    "predicted_direction_deg",
    "costmap_version",
    "equipment_model_id",
    "equipment_detection_count",
    "equipment_classes",
    "replan_mode",
    "replan_triggered",
    "replan_success",
    "replan_start_grid_x",
    "replan_start_grid_y",
    "replan_goal_grid_x",
    "replan_goal_grid_y",
    "replan_path_length",
    "dynamic_blocked_cell_count",
    "replan_count",
    "replan_route_replaced",
    "active_replan_count",
    "active_replan_path_length",
]


def _value_or_blank(message, attribute):
    if message is None:
        return ""
    return getattr(message, attribute, "")


def _flight_mode_name(flight_mode):
    if flight_mode is None:
        return ""
    return getattr(flight_mode, "name", str(flight_mode))


def _battery_percent_or_blank(battery):
    if battery is None:
        return ""
    remaining_percent = getattr(battery, "remaining_percent", None)
    if remaining_percent is None:
        return ""
    return remaining_percent * 100


def replan_csv_values(replan_state):
    """Return CSV-ready local-replan fields for one telemetry sample.

    Args:
        replan_state: Mutable state owned by the flight loop. Some fields are
            one-shot event flags that the logger captures before they are reset.
    """
    return [
        replan_state.get("replan_mode", "log_only"),
        "true" if replan_state.get("replan_triggered") else "false",
        replan_state.get("replan_success", ""),
        replan_state.get("replan_start_grid_x", ""),
        replan_state.get("replan_start_grid_y", ""),
        replan_state.get("replan_goal_grid_x", ""),
        replan_state.get("replan_goal_grid_y", ""),
        replan_state.get("replan_path_length", ""),
        replan_state.get("dynamic_blocked_cell_count", ""),
        replan_state.get("replan_count", 0),
        "true" if replan_state.get("replan_route_replaced") else "false",
        replan_state.get("active_replan_count", 0),
        replan_state.get("active_replan_path_length", ""),
    ]


def perception_csv_values(perception_config, detection):
    """Return CSV-ready perception fields for one telemetry sample.

    Args:
        perception_config: CLI-derived perception settings.
        detection: Structured perception state, or `None` for legacy callers.

    Returns:
        A list matching the perception slice of `TELEMETRY_CSV_HEADER`.
    """
    enabled = bool(perception_config.get("enabled"))
    range_m = perception_config.get("detection_range_m", "")
    fov_deg = perception_config.get("detection_fov_deg", "")
    warning_m = perception_config.get("warning_distance_m", "")
    danger_m = perception_config.get("danger_distance_m", "")
    risk_action = perception_config.get("risk_action", "log_only")
    source = perception_config.get("source", "map_baseline")
    equipment_model = perception_config.get("equipment_model") or ""

    if detection is None:
        return [
            "true" if enabled else "false",
            "false",
            "disabled" if not enabled else "clear",
            "",
            "",
            "",
            "",
            0,
            range_m,
            fov_deg,
            warning_m,
            danger_m,
            risk_action,
            0,
            "",
            "",
            "false",
            "false",
            "none",
            "perception disabled" if not enabled else "local position unavailable",
            source,
            "false" if enabled else "",
            "",
            "",
            "",
            perception_config.get("risk_model", ""),
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            equipment_model,
            0,
            "",
        ]

    state_enabled = bool(detection.get("enabled", enabled))
    risk_level = detection.get("risk_level", "clear")
    closest_name = detection.get("closest_obstacle_name", "")
    closest_distance_m = detection.get("closest_obstacle_distance_m")
    in_detection_range = bool(detection.get("in_detection_range", False))
    in_fov = bool(detection.get("in_fov", False))
    state_values = [
        detection.get("risk_value", 0),
        closest_name,
        closest_distance_m if closest_distance_m is not None else "",
        "true" if in_detection_range else "false",
        "true" if in_fov else "false",
        detection.get("suggested_action", "none"),
        detection.get("risk_reason", ""),
        detection.get("sensor_source", source),
        (
            "true"
            if detection.get("sensor_healthy", source == "map_baseline")
            else "false"
        ),
        detection.get("sensor_frequency_hz", ""),
        detection.get("sensor_frame_age_s", ""),
        detection.get("sensor_dropped_frames", ""),
        detection.get("risk_model_id", perception_config.get("risk_model", "")),
        detection.get("inference_latency_ms", ""),
        detection.get("model_inference_latency_ms", ""),
        detection.get("truth_risk_level", ""),
        detection.get("geometric_risk_level", ""),
        detection.get("ml_risk_level", ""),
        detection.get("risk_confidence", ""),
        json.dumps(detection.get("truth_traversability", []), separators=(",", ":")),
        json.dumps(
            detection.get("predicted_traversability", []), separators=(",", ":")
        ),
        detection.get("truth_direction_deg", ""),
        detection.get("predicted_direction_deg", ""),
        getattr(detection.get("costmap"), "version", ""),
        detection.get("equipment_model_id", equipment_model),
        len(detection.get("equipment_detections", [])),
        ",".join(
            sorted(
                {
                    item.get("class_name", "")
                    for item in detection.get("equipment_detections", [])
                    if item.get("class_name")
                }
            )
        ),
    ]

    nearest = detection["nearest_obstacle"]
    if nearest is None:
        return [
            "true" if state_enabled else "false",
            "false",
            "clear" if state_enabled else "disabled",
            "",
            "",
            "",
            "",
            0,
            range_m,
            fov_deg,
            warning_m,
            danger_m,
            risk_action,
            *state_values,
        ]

    return [
        "true" if state_enabled else "false",
        "true",
        risk_level,
        nearest["obstacle_name"],
        nearest["obstacle_layer"],
        nearest["distance_m"],
        nearest["bearing_deg_relative"],
        len(detection["detected_obstacles"]),
        range_m,
        fov_deg,
        warning_m,
        danger_m,
        risk_action,
        *state_values,
    ]


def build_telemetry_log_row(
    now,
    start_time,
    phase_state,
    target,
    position,
    velocity,
    attitude,
    battery,
    flight_mode,
    armed,
    planner_info,
    error,
    perception_values,
    replan_values,
):
    """Build one telemetry CSV row from flight, planner, perception, and replan data."""
    return [
        now.isoformat(),
        round((now - start_time).total_seconds(), 3),
        phase_state["phase"],
        phase_state.get("route_direction", "none"),
        target["name"],
        target["north_m"] if target["name"] else "",
        target["east_m"] if target["name"] else "",
        target["down_m"] if target["name"] else "",
        _value_or_blank(position, "north_m"),
        _value_or_blank(position, "east_m"),
        _value_or_blank(position, "down_m"),
        error["north_m"] if error else "",
        error["east_m"] if error else "",
        error["down_m"] if error else "",
        error["horizontal_m"] if error else "",
        _value_or_blank(velocity, "north_m_s"),
        _value_or_blank(velocity, "east_m_s"),
        _value_or_blank(velocity, "down_m_s"),
        _value_or_blank(attitude, "roll_deg"),
        _value_or_blank(attitude, "pitch_deg"),
        _value_or_blank(attitude, "yaw_deg"),
        _battery_percent_or_blank(battery),
        _flight_mode_name(flight_mode),
        armed if armed is not None else "",
        planner_info["grid_start"],
        planner_info["grid_goal"],
        planner_info["grid_width"],
        planner_info["grid_height"],
        planner_info["planner_name"],
        planner_info["map_name"],
        planner_info["resolution_m"],
        planner_info["altitude_m"],
        planner_info["return_home_enabled"],
        *perception_values,
        *replan_values,
    ]
