"""Markdown sections shared by A* analysis reports."""

from src.logging.metrics import format_value


def markdown_transition_table(rows):
    if not rows:
        return ["No waypoint target data was available."]
    lines = [
        "| route_direction | target_name | first seen (s) | last seen (s) | min error (m) | reached | reached time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['route_direction']} | "
            f"{row['target_name']} | "
            f"{row['first_time_s']:.2f} | "
            f"{row['last_time_s']:.2f} | "
            f"{format_value(row['min_horizontal_error_m'])} | "
            f"{'yes' if row['reached'] else 'no'} | "
            f"{format_value(row['reached_time_s'])} |"
        )
    return lines


def markdown_replan_summary(summary):
    lines = [
        "## Local Replan Summary",
        "",
    ]
    if summary is None:
        lines.append("- No local replan columns were found in this log.")
        return lines

    lines.extend(
        [
            f"- replan_mode: {summary.get('replan_mode') or 'N/A'}",
            f"- total_replan_attempts: {summary['total_replan_attempts']}",
            f"- successful_replan_attempts: {summary['successful_replan_attempts']}",
            f"- max_replan_path_length: {format_value(summary['max_replan_path_length'])}",
            f"- max_dynamic_blocked_cell_count: {format_value(summary['max_dynamic_blocked_cell_count'])}",
        ]
    )
    if summary.get("active_route_replacement_count") is not None:
        lines.extend(
            [
                f"- active_route_replacement_count: {summary['active_route_replacement_count']}",
                f"- max_active_replan_path_length: {format_value(summary['max_active_replan_path_length'])}",
            ]
        )
    elif summary.get("replan_mode") == "log_only":
        lines.append(
            "- Local replanning was log-only; analyzed replans did not replace active flight waypoints."
        )
    return lines


def markdown_active_replan_route_replacement_summary(summary):
    lines = [
        "## Active Replan Route Replacement Summary",
        "",
    ]
    if summary is None:
        lines.append("- No active replan route replacement columns were found in this log.")
        return lines

    lines.extend(
        [
            f"- active_route_replacement_count: {summary['active_route_replacement_count']}",
            f"- first_active_route_replacement_time_s: {format_value(summary['first_active_route_replacement_time_s'], ' s')}",
            f"- target_before_replacement: {summary['target_before_replacement'] or 'N/A'}",
            f"- first_replanned_target_after_replacement: {summary['first_replanned_target_after_replacement'] or 'N/A'}",
            f"- active_replan_path_length: {format_value(summary['active_replan_path_length'])}",
        ]
    )
    if summary.get("replan_mode") != "active":
        lines.append("- Active route replacement was not enabled for this run.")
    elif summary["active_route_replacement_count"] == 0:
        lines.append("- No active route replacement event was recorded.")
    return lines


def markdown_active_replan_target_validation(validation):
    """Render the machine-readable target-switching result for summary.md."""
    sequence = validation.get("post_replan_unique_target_sequence") or []

    def yes_no(value):
        if value is None:
            return "N/A"
        return "yes" if value else "no"

    return [
        "## Active Replan Target Validation",
        "",
        f"- pre_replan_target_name: {validation.get('pre_replan_target_name') or 'N/A'}",
        f"- first_replanned_target_name: {validation.get('first_replanned_target_name') or 'N/A'}",
        f"- first_replanned_target_elapsed_s: {format_value(validation.get('first_replanned_target_elapsed_s'), ' s')}",
        f"- post_replan_unique_target_sequence: {', '.join(sequence) if sequence else 'N/A'}",
        f"- post_replan_old_wp_target_count: {validation.get('post_replan_old_wp_target_count', 0)}",
        f"- rwp_sequence_contiguous: {yes_no(validation.get('rwp_sequence_contiguous'))}",
        f"- original_goal_reached: {yes_no(validation.get('original_goal_reached'))}",
        f"- mission_completed: {yes_no(validation.get('mission_completed'))}",
        f"- active_replan_target_switching_status: {validation['active_replan_target_switching_status']}",
        f"- active_replan_target_switching_notes: {validation['active_replan_target_switching_notes']}",
    ]


def generated_file_description(path):
    descriptions = {
        "summary.md": "human-readable A* analysis report.",
        "manifest.json": "machine-readable analysis metadata.",
        "run_metadata.json": "stage classification and experiment metadata.",
        "traj.png": "clean trajectory overview showing actual path, planned A* waypoints, start, and final/end markers.",
        "traj_with_obstacles.png": "obstacle validation plot showing raw footprints, inflated planning cells, trajectory, and collision/buffer entry points when present.",
        "error.png": "horizontal waypoint tracking error over time.",
        "alt.png": "altitude over time.",
        "vel.png": "NED velocity over time.",
        "yaw.png": "yaw over time.",
        "target_timeline.png": "waypoint target over time.",
        "perception_timeline.png": "nearest simulated obstacle distance over time for active perception samples.",
        "detection_count_over_time.png": "number of simulated obstacle detections at each active perception sample.",
        "perception_risk_timeline.png": "clear/detected/warning/danger risk level over time.",
        "collision_points.csv": "raw physical collision and inflated safety-buffer entry samples.",
        "collision_zoom.png": "zoomed obstacle validation plot around collision/buffer entry samples.",
    }
    return descriptions.get(path.name)


def markdown_generated_file_list(paths):
    if not paths:
        return ["- None"]
    lines = []
    for path in paths:
        description = generated_file_description(path)
        if description:
            lines.append(f"- `{path.name}`: {description}")
        else:
            lines.append(f"- `{path.name}`")
    return lines


def markdown_perception_summary(summary):
    if not summary["available"]:
        return [
            "## Perception Summary",
            "",
            "- perception_enabled: no",
            "- No perception columns were found in this log.",
        ]

    most_frequent = summary["most_frequent_obstacle_names"] or ["N/A"]
    warning_frequent = summary["most_frequent_warning_obstacle_names"] or ["N/A"]
    danger_frequent = summary["most_frequent_danger_obstacle_names"] or ["N/A"]
    risk_note = (
        "- No warning or danger perception samples were recorded."
        if summary["warning_samples"] == 0 and summary["danger_samples"] == 0
        else "- Warning/danger samples were recorded; inspect `perception_risk_timeline.png` and `traj_with_obstacles.png`."
    )
    return [
        "## Perception Summary",
        "",
        f"- perception_enabled: {'yes' if summary['perception_enabled'] else 'no'}",
        f"- total_duration_s: {format_value(summary['total_duration_s'])}",
        f"- detection_range_m: {format_value(summary['detection_range_m'])}",
        f"- detection_fov_deg: {format_value(summary['detection_fov_deg'])}",
        f"- warning_distance_m: {format_value(summary['warning_distance_m'])}",
        f"- danger_distance_m: {format_value(summary['danger_distance_m'])}",
        f"- risk_action: {summary['risk_action'] or 'N/A'}",
        f"- perception_source: {summary['perception_source'] or 'N/A'}",
        f"- sensor_healthy_ratio: {format_value(summary['sensor_healthy_ratio'])}",
        f"- sensor_frequency_p50_hz: {format_value(summary['sensor_frequency_p50_hz'], ' Hz')}",
        f"- sensor_frequency_p95_hz: {format_value(summary['sensor_frequency_p95_hz'], ' Hz')}",
        f"- sensor_frame_age_p50_ms: {format_value(summary['sensor_frame_age_p50_ms'], ' ms')}",
        f"- sensor_frame_age_p95_ms: {format_value(summary['sensor_frame_age_p95_ms'], ' ms')}",
        f"- sensor_dropped_frames_max: {format_value(summary['sensor_dropped_frames_max'])}",
        f"- risk_model_id: {summary['risk_model_id'] or 'N/A'}",
        f"- inference_latency_p50_ms: {format_value(summary['inference_latency_p50_ms'], ' ms')}",
        f"- inference_latency_p95_ms: {format_value(summary['inference_latency_p95_ms'], ' ms')}",
        f"- costmap_version_max: {format_value(summary['costmap_version_max'])}",
        f"- equipment_model_id: {summary['equipment_model_id'] or 'N/A'}",
        f"- equipment_detection_count_max: {format_value(summary['equipment_detection_count_max'])}",
        f"- equipment_classes: {', '.join(summary['equipment_classes']) or 'N/A'}",
        f"- total_detection_samples: {summary['total_detection_samples']}",
        f"- clear_sample_count: {summary['clear_sample_count']}",
        f"- detected_sample_count: {summary['detected_sample_count']}",
        f"- warning_sample_count: {summary['warning_sample_count']}",
        f"- danger_sample_count: {summary['danger_sample_count']}",
        f"- clear_sample_ratio: {format_value(summary['clear_sample_ratio'])}",
        f"- detected_sample_ratio: {format_value(summary['detected_sample_ratio'])}",
        f"- warning_sample_ratio: {format_value(summary['warning_sample_ratio'])}",
        f"- danger_sample_ratio: {format_value(summary['danger_sample_ratio'])}",
        f"- time_in_clear_s: {format_value(summary['time_in_clear_s'], ' s')}",
        f"- time_in_detected_s: {format_value(summary['time_in_detected_s'], ' s')}",
        f"- time_in_warning_s: {format_value(summary['time_in_warning_s'], ' s')}",
        f"- time_in_danger_s: {format_value(summary['time_in_danger_s'], ' s')}",
        f"- percent_time_clear: {format_value(summary['percent_time_clear'], '%')}",
        f"- percent_time_detected: {format_value(summary['percent_time_detected'], '%')}",
        f"- percent_time_warning: {format_value(summary['percent_time_warning'], '%')}",
        f"- percent_time_danger: {format_value(summary['percent_time_danger'], '%')}",
        f"- samples_with_detections: {summary['samples_with_detections']}",
        f"- first detection time: {format_value(summary['first_detection_time_s'], ' s')}",
        f"- first warning time: {format_value(summary['first_warning_time_s'], ' s')}",
        f"- first danger time: {format_value(summary['first_danger_time_s'], ' s')}",
        f"- nearest obstacle ever detected: {summary['nearest_obstacle_ever_detected'] or 'N/A'}",
        f"- minimum_nearest_obstacle_distance_m: {format_value(summary['minimum_nearest_obstacle_distance_m'], ' m')}",
        f"- mean_nearest_obstacle_distance_m: {format_value(summary['mean_nearest_obstacle_distance_m'], ' m')}",
        f"- median_nearest_obstacle_distance_m: {format_value(summary['median_nearest_obstacle_distance_m'], ' m')}",
        f"- most frequently detected obstacle names: {', '.join(most_frequent)}",
        f"- most frequent warning obstacle names: {', '.join(warning_frequent)}",
        f"- most frequent danger obstacle names: {', '.join(danger_frequent)}",
        "",
        "Interpretation:",
        "",
        "- Sample ratios and percent-time metrics are normalized so runs with different durations can be compared more fairly.",
        "- `detected` means an obstacle is within sensor range but not immediately risky.",
        "- `warning` means the drone is close enough to require caution.",
        "- `danger` means the drone is very close and should trigger emergency behavior in future versions.",
        risk_note,
    ]
