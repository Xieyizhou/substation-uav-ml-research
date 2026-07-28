"""CSV, Markdown, evaluation-table, and output-index writers."""

import csv
from datetime import datetime, timezone

from src.logging.evaluation_schema import EVALUATION_COLUMNS
from src.logging.output_registry import (
    OUTPUT_ROOT,
    RUN_STAGES,
    STAGE_DIR_NAMES,
    STAGE_LABELS,
    display_path,
    get_comparisons_dir,
    get_summary_paths,
)
from src.logging.summary_values import (
    FINAL_ERROR_PASS_THRESHOLD_M,
    SUMMARY_COLUMNS,
    first_non_empty,
    format_bool,
    format_float,
    parse_float,
)

def normalized_csv_row(row):
    normalized = {}
    for column in SUMMARY_COLUMNS:
        value = row.get(column)
        if (
            column.endswith("_m")
            or column.endswith("_s")
            or column.endswith("_ratio")
            or column.startswith("percent_time_")
            or column == "max_speed_m_s"
            or column == "max_active_replan_path_length"
        ):
            normalized[column] = format_float(value)
        elif column.endswith("_detected") or column == "return_home_enabled":
            normalized[column] = format_bool(value)
        elif column in {"waypoints_reached", "active_route_replacement_count"}:
            normalized[column] = "" if value is None else str(value)
        else:
            normalized[column] = "" if value is None else str(value)
    return normalized


def normalized_evaluation_row(row):
    normalized = {}
    for column in EVALUATION_COLUMNS:
        value = row.get(column)
        if column.endswith("_m") or column.endswith("_s"):
            normalized[column] = format_float(value)
        elif column.endswith("_count"):
            parsed = parse_float(value)
            normalized[column] = "" if parsed is None else str(int(parsed))
        else:
            normalized[column] = "" if value is None else str(value)
    return normalized


def write_csv(rows, summary_csv):
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalized_csv_row(row))


def write_evaluation_csv(rows, evaluation_csv):
    evaluation_csv.parent.mkdir(parents=True, exist_ok=True)
    with evaluation_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=EVALUATION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalized_evaluation_row(row))


def markdown_value(row, column):
    value = normalized_csv_row(row).get(column, "")
    return value if value != "" else "N/A"


def evaluation_markdown_value(row, column):
    value = normalized_evaluation_row(row).get(column, "")
    return value if value != "" else "unavailable"


def write_evaluation_markdown(rows, stage_name, evaluation_csv, evaluation_md):
    created_at = datetime.now(timezone.utc).isoformat()
    stage_label = STAGE_LABELS.get(stage_name, stage_name)
    lines = [
        f"# A* Experiment Evaluation: {stage_label}",
        "",
        f"Generated: `{created_at}`",
        "",
        "This table is the formal evaluation layer for this stage. Missing metrics are shown as `unavailable` instead of estimated.",
        "",
    ]
    if not rows:
        lines.append("No analyzed runs were found for this stage.")
    else:
        lines.extend(
            [
                "| run_id | stage | experiment type | completed/failed | flight time (s) | planned path (m) | actual distance (m) | min obstacle distance (m) | buffer violations | risk detections | slow_down events | replan attempts | successful replans | active replacements | final status | notes |",
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                f"{evaluation_markdown_value(row, 'run_id')} | "
                f"{evaluation_markdown_value(row, 'stage_folder')} | "
                f"{evaluation_markdown_value(row, 'experiment_type')} | "
                f"{evaluation_markdown_value(row, 'completed_or_failed')} | "
                f"{evaluation_markdown_value(row, 'total_flight_time_s')} | "
                f"{evaluation_markdown_value(row, 'planned_path_length_m')} | "
                f"{evaluation_markdown_value(row, 'actual_traveled_distance_m')} | "
                f"{evaluation_markdown_value(row, 'minimum_distance_to_obstacle_m')} | "
                f"{evaluation_markdown_value(row, 'safety_buffer_violation_count')} | "
                f"{evaluation_markdown_value(row, 'perception_risk_detection_count')} | "
                f"{evaluation_markdown_value(row, 'slow_down_event_count')} | "
                f"{evaluation_markdown_value(row, 'local_replan_attempt_count')} | "
                f"{evaluation_markdown_value(row, 'successful_local_replan_count')} | "
                f"{evaluation_markdown_value(row, 'active_route_replacement_count')} | "
                f"{evaluation_markdown_value(row, 'final_status')} | "
                f"{evaluation_markdown_value(row, 'notes')} |"
            )
    lines.extend(["", f"CSV output: `{display_path(evaluation_csv)}`"])
    evaluation_md.write_text("\n".join(lines) + "\n")


def to_evaluation_row(row):
    return {
        "run_id": row.get("run_id"),
        "stage": row.get("stage"),
        "stage_folder": row.get("stage_folder") or STAGE_DIR_NAMES.get(row.get("stage")),
        "experiment_type": row.get("experiment_type"),
        "completed_or_failed": row.get("completed_or_failed"),
        "total_flight_time_s": row.get("duration_s"),
        "planned_path_length_m": row.get("planned_path_length_m"),
        "actual_traveled_distance_m": row.get("actual_traveled_distance_m"),
        "minimum_distance_to_obstacle_m": row.get("minimum_nearest_obstacle_distance_m"),
        "safety_buffer_violation_count": row.get("safety_buffer_violation_count"),
        "perception_risk_detection_count": row.get("perception_risk_detection_count"),
        "slow_down_event_count": row.get("slow_down_event_count"),
        "local_replan_attempt_count": row.get("total_replan_attempts"),
        "successful_local_replan_count": row.get("successful_replan_attempts"),
        "active_route_replacement_count": row.get("active_route_replacement_count"),
        "final_status": row.get("final_status") or row.get("status"),
        "notes": row.get("notes"),
    }


def average_metric(rows, column):
    values = [parse_float(row.get(column)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def grouped_by_risk_action(rows):
    groups = {}
    for row in rows:
        risk_action = first_non_empty(row.get("risk_action"), "N/A")
        groups.setdefault(str(risk_action), []).append(row)
    return dict(sorted(groups.items(), key=lambda item: item[0]))


def risk_action_group_stats(rows):
    metrics = [
        "warning_sample_ratio",
        "danger_sample_ratio",
        "percent_time_warning",
        "percent_time_danger",
        "minimum_nearest_obstacle_distance_m",
        "mean_nearest_obstacle_distance_m",
        "duration_s",
    ]
    return {
        risk_action: {
            "run_count": len(group_rows),
            **{metric: average_metric(group_rows, metric) for metric in metrics},
        }
        for risk_action, group_rows in grouped_by_risk_action(rows).items()
    }


def format_average(value):
    return "N/A" if value is None else f"{value:.3f}"


def append_risk_action_comparison(lines, rows):
    stats = risk_action_group_stats(rows)
    lines.extend(["", "## Risk Action Comparison", ""])
    if not stats:
        lines.append("No risk_action groups were available.")
        return

    lines.extend(
        [
            "| risk_action | runs | avg warning_sample_ratio | avg danger_sample_ratio | avg percent_time_warning | avg percent_time_danger | avg min obstacle distance (m) | avg mean obstacle distance (m) | avg duration (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for risk_action, values in stats.items():
        lines.append(
            "| "
            f"{risk_action} | "
            f"{values['run_count']} | "
            f"{format_average(values['warning_sample_ratio'])} | "
            f"{format_average(values['danger_sample_ratio'])} | "
            f"{format_average(values['percent_time_warning'])} | "
            f"{format_average(values['percent_time_danger'])} | "
            f"{format_average(values['minimum_nearest_obstacle_distance_m'])} | "
            f"{format_average(values['mean_nearest_obstacle_distance_m'])} | "
            f"{format_average(values['duration_s'])} |"
        )

    log_only = stats.get("log_only")
    slow_down = stats.get("slow_down")
    lines.extend(["", "Interpretation:", ""])
    if not log_only or not slow_down:
        lines.append(
            "- Need both `log_only` and `slow_down` analyzed runs for direct risk-action interpretation."
        )
        return

    slow_danger_ratio = slow_down.get("danger_sample_ratio")
    log_danger_ratio = log_only.get("danger_sample_ratio")
    slow_danger_time = slow_down.get("percent_time_danger")
    log_danger_time = log_only.get("percent_time_danger")
    if (
        slow_danger_ratio is not None
        and log_danger_ratio is not None
        and slow_danger_ratio > log_danger_ratio
    ) or (
        slow_danger_time is not None
        and log_danger_time is not None
        and slow_danger_time > log_danger_time
    ):
        lines.append(
            "- `slow_down` has a higher danger ratio or higher percent_time_danger than `log_only`; slowing down may cause the drone to spend longer near obstacles without changing its path."
        )

    slow_min_distance = slow_down.get("minimum_nearest_obstacle_distance_m")
    log_min_distance = log_only.get("minimum_nearest_obstacle_distance_m")
    clearance_improved = (
        slow_min_distance is not None
        and log_min_distance is not None
        and slow_min_distance > log_min_distance
    )
    if clearance_improved:
        lines.append("- `slow_down` increases minimum obstacle distance, so it may improve clearance.")

    slow_duration = slow_down.get("duration_s")
    log_duration = log_only.get("duration_s")
    if (
        slow_duration is not None
        and log_duration is not None
        and slow_duration > log_duration
        and not clearance_improved
    ):
        lines.append(
            "- `slow_down` increases duration but does not improve minimum clearance; future work should add local replanning or safer waypoint generation."
        )


def write_markdown(rows, stage_name, summary_csv, summary_md):
    created_at = datetime.now(timezone.utc).isoformat()
    stage_label = STAGE_LABELS.get(stage_name, stage_name)
    lines = [
        f"# A* Experiment Benchmark Summary: {stage_label}",
        "",
        f"Generated: `{created_at}`",
        "",
        "Status rules:",
        "",
        f"- `PASS`: no raw physical collision, no inflated buffer entry, and final horizontal error <= {FINAL_ERROR_PASS_THRESHOLD_M:.1f} m.",
        "- `WARNING`: inflated safety buffer was entered, final error is missing/high, or the run needs review.",
        "- `FAIL`: raw physical collision was detected.",
        "",
    ]

    if not rows:
        lines.extend(
            [
                "No analyzed A* runs were found.",
                "",
                "Run `python main.py report analyze --obstacle-config config/substation_obstacles.json` first.",
            ]
        )
    else:
        lines.extend(
            [
                "| status | run_id | risk_action | map | alt (m) | speed (m/s) | speed source | return home | duration (s) | warning ratio | danger ratio | % warning | % danger | min obstacle dist (m) | mean obstacle dist (m) | median obstacle dist (m) | raw collision | buffer entry | active replans | max active replan path | waypoints reached |",
                "|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                f"{markdown_value(row, 'status')} | "
                f"{markdown_value(row, 'run_id')} | "
                f"{markdown_value(row, 'risk_action')} | "
                f"{markdown_value(row, 'map_name')} | "
                f"{markdown_value(row, 'altitude_m')} | "
                f"{markdown_value(row, 'max_speed_m_s')} | "
                f"{markdown_value(row, 'speed_source')} | "
                f"{markdown_value(row, 'return_home_enabled')} | "
                f"{markdown_value(row, 'duration_s')} | "
                f"{markdown_value(row, 'warning_sample_ratio')} | "
                f"{markdown_value(row, 'danger_sample_ratio')} | "
                f"{markdown_value(row, 'percent_time_warning')} | "
                f"{markdown_value(row, 'percent_time_danger')} | "
                f"{markdown_value(row, 'minimum_nearest_obstacle_distance_m')} | "
                f"{markdown_value(row, 'mean_nearest_obstacle_distance_m')} | "
                f"{markdown_value(row, 'median_nearest_obstacle_distance_m')} | "
                f"{markdown_value(row, 'raw_physical_collision_detected')} | "
                f"{markdown_value(row, 'inflated_safety_buffer_entry_detected')} | "
                f"{markdown_value(row, 'active_route_replacement_count')} | "
                f"{markdown_value(row, 'max_active_replan_path_length')} | "
                f"{markdown_value(row, 'waypoints_reached')} |"
            )
        append_risk_action_comparison(lines, rows)

    lines.extend(
        [
            "",
            "CSV output:",
            "",
            f"- `{display_path(summary_csv)}`",
        ]
    )
    summary_md.write_text("\n".join(lines) + "\n")


def write_output_index(stage_counts):
    created_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Outputs",
        "",
        f"Generated: `{created_at}`",
        "",
        "New analysis outputs are stage-scoped. Do not write new `as_*` run folders directly under `outputs/`.",
        "",
        "## Stages",
        "",
        "| stage | runs | summaries |",
        "|---|---:|---|",
    ]
    for stage_name in RUN_STAGES:
        paths = get_summary_paths(stage_name)
        summary_links = (
            f"`{display_path(paths['csv'])}`, `{display_path(paths['md'])}`, "
            f"`{display_path(paths['evaluation_csv'])}`, `{display_path(paths['evaluation_md'])}`"
        )
        lines.append(
            f"| `{stage_name}` ({STAGE_LABELS.get(stage_name, stage_name)}) | "
            f"{stage_counts.get(stage_name, 0)} | {summary_links} |"
        )
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            f"- Cross-stage comparison outputs: `{display_path(get_comparisons_dir())}`",
            "- Use `python main.py report compare` for intentional landmark comparisons across stages.",
        ]
    )
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines) + "\n")
