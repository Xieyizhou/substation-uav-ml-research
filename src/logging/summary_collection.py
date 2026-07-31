"""Collect one normalized summary row from each analyzed run."""

from src.logging.output_registry import RUN_STAGES, STAGE_DIR_NAMES, display_path, get_runs_dir
from src.logging.summary_values import (
    approximate_distance_traveled,
    choose_status,
    commanded_speed,
    completed_or_failed,
    count_csv_true,
    count_risk_rows,
    csv_rows,
    first_csv_value,
    first_non_empty,
    last_csv_float,
    load_json,
    max_csv_float,
    mean_csv_float,
    measured_max_horizontal_speed,
    parse_bool,
    parse_summary_bullets,
    planned_path_length,
    read_text,
    slow_down_event_count,
    warning_note,
    waypoint_reached_count,
)

def collect_run(output_dir):
    manifest = load_json(output_dir / "manifest.json")
    metadata = load_json(output_dir / "run_metadata.json")
    summary_text = read_text(output_dir / "summary.md")
    summary = parse_summary_bullets(summary_text)
    collision = manifest.get("collision_report", {})
    height = manifest.get("height_aware_planning", {})
    perception = manifest.get("perception_summary", {})
    replan = manifest.get("replan_summary", {}) or {}
    active_replan = manifest.get("active_replan_route_replacement_summary", {}) or {}
    run_status = manifest.get("run_status") or {}

    run_id = first_non_empty(manifest.get("run_id"), summary.get("run_id"), output_dir.name)
    source_log = first_non_empty(manifest.get("source_log"), summary.get("source_log"))
    rows = csv_rows(source_log)

    speed, speed_source = commanded_speed(rows)
    if speed is None:
        speed = measured_max_horizontal_speed(rows)
        speed_source = "measured_velocity"

    raw_collision = first_non_empty(
        collision.get("raw_physical_collision_detected"),
        summary.get("raw_physical_collision_detected"),
    )
    inflated_entry = first_non_empty(
        collision.get("inflated_safety_buffer_entry_detected"),
        summary.get("inflated_safety_buffer_entry_detected"),
    )
    obstacle_collision = first_non_empty(
        collision.get("obstacle_collision_detected"),
        summary.get("obstacle_collision_detected"),
    )
    if obstacle_collision is None:
        obstacle_collision = bool(parse_bool(raw_collision) or parse_bool(inflated_entry))

    final_error = first_non_empty(
        summary.get("final_horizontal_error"),
        last_csv_float(rows, "horizontal_error_m"),
    )

    inferred_stage = first_non_empty(metadata.get("stage"), infer_stage_from_output_dir(output_dir))
    result = {
        "stage": inferred_stage,
        "stage_folder": STAGE_DIR_NAMES.get(inferred_stage),
        "experiment_type": metadata.get("experiment_type"),
        "run_id": run_id,
        "source_log": display_path(source_log),
        "risk_action": first_non_empty(
            perception.get("risk_action"),
            summary.get("risk_action"),
            first_csv_value(rows, "risk_action"),
        ),
        "map_name": first_non_empty(
            manifest.get("map_name"),
            summary.get("map_name"),
            first_csv_value(rows, "map_name"),
        ),
        "altitude_m": first_non_empty(
            height.get("altitude_m"),
            summary.get("height_aware_altitude"),
            summary.get("max_altitude"),
            first_csv_value(rows, "altitude_m"),
        ),
        "max_speed_m_s": speed,
        "speed_source": speed_source,
        "return_home_enabled": first_non_empty(
            summary.get("return_home_enabled"),
            first_csv_value(rows, "return_home_enabled"),
        ),
        "duration_s": first_non_empty(
            manifest.get("duration_s"),
            perception.get("total_duration_s"),
            summary.get("duration"),
            summary.get("total_duration_s"),
            max_csv_float(rows, "elapsed_s"),
        ),
        "warning_sample_ratio": first_non_empty(
            perception.get("warning_sample_ratio"),
            summary.get("warning_sample_ratio"),
        ),
        "danger_sample_ratio": first_non_empty(
            perception.get("danger_sample_ratio"),
            summary.get("danger_sample_ratio"),
        ),
        "percent_time_warning": first_non_empty(
            perception.get("percent_time_warning"),
            summary.get("percent_time_warning"),
        ),
        "percent_time_danger": first_non_empty(
            perception.get("percent_time_danger"),
            summary.get("percent_time_danger"),
        ),
        "minimum_nearest_obstacle_distance_m": first_non_empty(
            perception.get("minimum_nearest_obstacle_distance_m"),
            summary.get("minimum_nearest_obstacle_distance_m"),
            summary.get("minimum_nearest_obstacle_distance"),
        ),
        "mean_nearest_obstacle_distance_m": first_non_empty(
            perception.get("mean_nearest_obstacle_distance_m"),
            summary.get("mean_nearest_obstacle_distance_m"),
        ),
        "median_nearest_obstacle_distance_m": first_non_empty(
            perception.get("median_nearest_obstacle_distance_m"),
            summary.get("median_nearest_obstacle_distance_m"),
        ),
        "max_horizontal_error_m": first_non_empty(
            summary.get("max_horizontal_error"),
            max_csv_float(rows, "horizontal_error_m"),
        ),
        "mean_horizontal_error_m": first_non_empty(
            summary.get("mean_horizontal_error"),
            mean_csv_float(rows, "horizontal_error_m"),
        ),
        "final_horizontal_error_m": final_error,
        "approximate_min_clearance_m": first_non_empty(
            collision.get("approximate_min_clearance_m"),
            summary.get("approximate_min_clearance_m"),
        ),
        "raw_physical_collision_detected": raw_collision,
        "inflated_safety_buffer_entry_detected": inflated_entry,
        "obstacle_collision_detected": obstacle_collision,
        "active_route_replacement_count": first_non_empty(
            active_replan.get("active_route_replacement_count"),
            replan.get("active_route_replacement_count"),
            summary.get("active_route_replacement_count"),
            max_csv_float(rows, "active_replan_count"),
            count_csv_true(rows, "replan_route_replaced"),
        ),
        "total_replan_attempts": first_non_empty(
            replan.get("total_replan_attempts"),
            max_csv_float(rows, "replan_count"),
        ),
        "successful_replan_attempts": first_non_empty(
            replan.get("successful_replan_attempts"),
            count_csv_true(rows, "replan_success"),
        ),
        "safety_buffer_violation_count": first_non_empty(
            collision.get("inflated_buffer_entry_points"),
            summary.get("inflated_buffer_entry_points"),
        ),
        "perception_risk_detection_count": first_non_empty(
            perception.get("samples_with_detections"),
            count_risk_rows(rows),
        ),
        "slow_down_event_count": slow_down_event_count(
            rows,
            first_non_empty(
                perception.get("risk_action"),
                summary.get("risk_action"),
                first_csv_value(rows, "risk_action"),
            ),
        ),
        "planned_path_length_m": planned_path_length(rows),
        "actual_traveled_distance_m": approximate_distance_traveled(rows),
        "notes": first_non_empty(
            run_status.get("message") if run_status else None,
            warning_note(manifest.get("warnings")),
        ),
        "max_active_replan_path_length": first_non_empty(
            active_replan.get("max_active_replan_path_length"),
            replan.get("max_active_replan_path_length"),
            summary.get("max_active_replan_path_length"),
            summary.get("active_replan_path_length"),
            max_csv_float(rows, "active_replan_path_length"),
        ),
        "waypoints_reached": waypoint_reached_count(summary_text),
    }
    result["status"] = choose_status(
        result["raw_physical_collision_detected"],
        result["inflated_safety_buffer_entry_detected"],
        result["final_horizontal_error_m"],
        manifest.get("run_status"),
    )
    result["completed_or_failed"] = completed_or_failed(result["status"])
    result["final_status"] = result["status"]
    return result


def infer_stage_from_output_dir(output_dir):
    parts = set(output_dir.parts)
    for stage_name, folder_name in STAGE_DIR_NAMES.items():
        if folder_name in parts:
            return stage_name
    return None


def find_stage_analysis_dirs(stage_name):
    runs_dir = get_runs_dir(stage_name)
    if not runs_dir.exists():
        return []
    dirs = []
    for path in sorted(runs_dir.glob("as_*")):
        if not path.is_dir():
            continue
        if (path / "manifest.json").exists() or (path / "summary.md").exists():
            dirs.append(path)
    return dirs
