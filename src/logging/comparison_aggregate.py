"""Aggregate metrics across all valid experiment runs."""

import csv
import statistics
from datetime import datetime, timezone

from src.logging.comparison_landmark import normalized_value, stage_rows
from src.logging.comparison_schema import AGGREGATE_COLUMNS, INCLUDED_RUN_COLUMNS
from src.logging.output_registry import RUN_STAGES, STAGE_DIR_NAMES, display_path
from src.logging.summarize_experiments import parse_float


def numeric_values(rows, column):
    values = [parse_float(row.get(column)) for row in rows]
    return [value for value in values if value is not None]


def mean_value(rows, column):
    values = numeric_values(rows, column)
    return statistics.mean(values) if values else None


def std_value(rows, column):
    values = numeric_values(rows, column)
    return statistics.stdev(values) if len(values) > 1 else None


def min_value(rows, column):
    values = numeric_values(rows, column)
    return min(values) if values else None


def max_value(rows, column):
    values = numeric_values(rows, column)
    return max(values) if values else None


def total_value(rows, column):
    values = numeric_values(rows, column)
    return sum(values) if values else None


def aggregate_stage_row(rows, stage_name):
    rows_for_stage = stage_rows(rows, stage_name)
    return {
        "stage": stage_name,
        "stage_folder": STAGE_DIR_NAMES[stage_name],
        "run_count": len(rows_for_stage),
        "completed_count": sum(
            1 for row in rows_for_stage if str(row.get("completed_or_failed", "")).lower() == "completed"
        ),
        "pass_count": sum(1 for row in rows_for_stage if str(row.get("final_status", "")).upper() == "PASS"),
        "mean_total_flight_time_s": mean_value(rows_for_stage, "total_flight_time_s"),
        "std_total_flight_time_s": std_value(rows_for_stage, "total_flight_time_s"),
        "min_total_flight_time_s": min_value(rows_for_stage, "total_flight_time_s"),
        "max_total_flight_time_s": max_value(rows_for_stage, "total_flight_time_s"),
        "mean_planned_path_length_m": mean_value(rows_for_stage, "planned_path_length_m"),
        "mean_actual_traveled_distance_m": mean_value(rows_for_stage, "actual_traveled_distance_m"),
        "mean_minimum_distance_to_obstacle_m": mean_value(rows_for_stage, "minimum_distance_to_obstacle_m"),
        "total_safety_buffer_violation_count": total_value(rows_for_stage, "safety_buffer_violation_count"),
        "mean_perception_risk_detection_count": mean_value(rows_for_stage, "perception_risk_detection_count"),
        "mean_slow_down_event_count": mean_value(rows_for_stage, "slow_down_event_count"),
        "mean_local_replan_attempt_count": mean_value(rows_for_stage, "local_replan_attempt_count"),
        "mean_successful_local_replan_count": mean_value(rows_for_stage, "successful_local_replan_count"),
        "mean_active_route_replacement_count": mean_value(rows_for_stage, "active_route_replacement_count"),
    }


def aggregate_rows(rows):
    return [aggregate_stage_row(rows, stage_name) for stage_name in RUN_STAGES]


def aggregate_csv_value(column, value):
    if value is None:
        return ""
    if column in {"stage", "stage_folder"}:
        return str(value)
    if column in {"run_count", "completed_count", "pass_count", "total_safety_buffer_violation_count"}:
        return str(int(value))
    return f"{float(value):.3f}"


def write_aggregate_csv(rows, output_path):
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=AGGREGATE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: aggregate_csv_value(column, row.get(column)) for column in AGGREGATE_COLUMNS}
            )


def write_included_runs_csv(rows, output_path):
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=INCLUDED_RUN_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: normalized_value(column, row.get(column)) for column in INCLUDED_RUN_COLUMNS}
            )


def aggregate_md_value(row, column):
    value = aggregate_csv_value(column, row.get(column))
    return value if value != "" else "n/a"


def write_aggregate_markdown(rows, output_path, csv_path, included_runs_path, missing_stages, min_runs_per_stage):
    created_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Aggregate Cross-Stage Comparison",
        "",
        f"Generated: `{created_at}`",
        "",
        "This summary uses all valid analyzed runs found under the four official stage folders.",
        "",
        f"Minimum requested runs per stage: `{min_runs_per_stage}`",
        "",
    ]
    if missing_stages:
        lines.extend(
            [
                "Aggregate comparison incomplete: missing required stages or not enough runs.",
                "",
                "Stages below the requested run count:",
                "",
            ]
        )
        for stage_name in missing_stages:
            lines.append(f"- `{STAGE_DIR_NAMES[stage_name]}`")
        lines.append("")
    lines.extend(
        [
            "| stage | run count | completed | pass | mean flight time (s) | std flight time (s) | min flight time (s) | max flight time (s) | mean planned path (m) | mean actual distance (m) | mean min obstacle distance (m) | total buffer violations | mean risk detections | mean slow_down events | mean replan attempts | mean successful replans | mean active replacements |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{aggregate_md_value(row, 'stage_folder')} | "
            f"{aggregate_md_value(row, 'run_count')} | "
            f"{aggregate_md_value(row, 'completed_count')} | "
            f"{aggregate_md_value(row, 'pass_count')} | "
            f"{aggregate_md_value(row, 'mean_total_flight_time_s')} | "
            f"{aggregate_md_value(row, 'std_total_flight_time_s')} | "
            f"{aggregate_md_value(row, 'min_total_flight_time_s')} | "
            f"{aggregate_md_value(row, 'max_total_flight_time_s')} | "
            f"{aggregate_md_value(row, 'mean_planned_path_length_m')} | "
            f"{aggregate_md_value(row, 'mean_actual_traveled_distance_m')} | "
            f"{aggregate_md_value(row, 'mean_minimum_distance_to_obstacle_m')} | "
            f"{aggregate_md_value(row, 'total_safety_buffer_violation_count')} | "
            f"{aggregate_md_value(row, 'mean_perception_risk_detection_count')} | "
            f"{aggregate_md_value(row, 'mean_slow_down_event_count')} | "
            f"{aggregate_md_value(row, 'mean_local_replan_attempt_count')} | "
            f"{aggregate_md_value(row, 'mean_successful_local_replan_count')} | "
            f"{aggregate_md_value(row, 'mean_active_route_replacement_count')} |"
        )
    lines.extend(
        [
            "",
            f"CSV output: `{display_path(csv_path)}`",
            f"Included runs: `{display_path(included_runs_path)}`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")
