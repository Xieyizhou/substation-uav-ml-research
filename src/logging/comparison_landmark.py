"""Write marker-selected landmark comparison reports."""

import csv
from datetime import datetime, timezone

from src.logging.comparison_collection import valid_stage_counts
from src.logging.comparison_schema import COMPARISON_COLUMNS, REQUIRED_STAGES
from src.logging.evaluation_schema import EVALUATION_COLUMNS
from src.logging.output_registry import RUN_STAGES, STAGE_DIR_NAMES, display_path
from src.logging.summarize_experiments import normalized_evaluation_row, parse_float


def write_status(output_path, rows, missing_stages, min_runs_per_stage, generated):
    created_at = datetime.now(timezone.utc).isoformat()
    counts = valid_stage_counts(rows)
    lines = [
        "# Cross-Stage Comparison Status",
        "",
        f"Generated: `{created_at}`",
        "",
        f"Minimum analyzed runs per stage: `{min_runs_per_stage}`",
        "",
        "| stage | analyzed selected runs | status |",
        "|---|---:|---|",
    ]
    for stage_name in REQUIRED_STAGES:
        count = counts.get(stage_name, 0)
        status = "ok" if stage_name not in missing_stages else "missing"
        lines.append(f"| `{STAGE_DIR_NAMES[stage_name]}` | {count} | {status} |")

    lines.extend(["", "## Result", ""])
    if generated:
        lines.append("Comparison outputs were generated.")
    else:
        lines.append(
            "Comparison outputs were skipped because at least one required stage "
            "does not have enough selected analyzed runs."
        )
        lines.extend(["", "Missing stages:", ""])
        for stage_name in missing_stages:
            lines.append(f"- `{STAGE_DIR_NAMES[stage_name]}`")
        lines.extend(
            [
                "",
                "Existing `comparison_summary.csv` and `comparison_summary.md` files were left unchanged.",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n")


def normalized_value(column, value):
    if column in EVALUATION_COLUMNS:
        return normalized_evaluation_row({column: value}).get(column, "")
    return "" if value is None else str(value)


def write_csv(rows, output_path):
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: normalized_value(column, row.get(column)) for column in COMPARISON_COLUMNS}
            )


def md_value(row, column):
    value = normalized_value(column, row.get(column))
    return value if value != "" else "n/a"


def interpretation(rows):
    if not rows:
        return ["- No marker-selected landmark runs were found."]
    lines = [
        "- This is an intentional cross-stage comparison; ordinary stage summaries remain separate.",
    ]
    baseline = next((row for row in rows if row.get("experiment_type") == "astar_only_perception_disabled"), None)
    slow_down = next((row for row in rows if row.get("experiment_type") == "perception_slow_down"), None)
    active = next((row for row in rows if row.get("experiment_type") == "active_local_replan"), None)
    if baseline and slow_down:
        b_clearance = parse_float(baseline.get("minimum_distance_to_obstacle_m"))
        s_clearance = parse_float(slow_down.get("minimum_distance_to_obstacle_m"))
        if b_clearance is not None and s_clearance is not None:
            relation = "higher" if s_clearance > b_clearance else "not higher"
            lines.append(f"- Slow-down minimum obstacle distance is {relation} than the static baseline.")
    if active:
        replacements = parse_float(active.get("active_route_replacement_count"))
        if replacements and replacements > 0:
            lines.append("- The active replan landmark recorded route replacement, so it changed the outbound route rather than only logging a possible replan.")
        else:
            lines.append("- The active replan landmark did not record a route replacement in the available metrics.")
    if any(row.get("completed_or_failed") == "failed" for row in rows):
        lines.append("- At least one selected run is marked failed and should be treated as a failure case.")
    return lines


def stage_rows(rows, stage_name):
    folder_name = STAGE_DIR_NAMES.get(stage_name)
    return [row for row in rows if row.get("stage") == stage_name or row.get("stage_folder") == folder_name]


def write_markdown(rows, output_path, csv_path):
    created_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Cross-Experiment Evaluation",
        "",
        f"Generated: `{created_at}`",
        "",
        "Selected landmark runs are grouped by canonical stage. Missing metrics are shown as `n/a`.",
    ]
    for stage_name in RUN_STAGES:
        folder_name = STAGE_DIR_NAMES[stage_name]
        lines.extend(["", f"## {folder_name}", ""])
        rows_for_stage = stage_rows(rows, stage_name)
        if not rows_for_stage:
            lines.append("No selected landmark run found for this stage.")
            continue
        lines.extend(
            [
                "| run_id | experiment type | completed/failed | flight time (s) | planned path (m) | actual distance (m) | min obstacle distance (m) | buffer violations | risk detections | slow_down events | replan attempts | successful replans | active replacements | final status | notes |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in rows_for_stage:
            lines.append(
                "| "
                f"{md_value(row, 'run_id')} | "
                f"{md_value(row, 'experiment_type')} | "
                f"{md_value(row, 'completed_or_failed')} | "
                f"{md_value(row, 'total_flight_time_s')} | "
                f"{md_value(row, 'planned_path_length_m')} | "
                f"{md_value(row, 'actual_traveled_distance_m')} | "
                f"{md_value(row, 'minimum_distance_to_obstacle_m')} | "
                f"{md_value(row, 'safety_buffer_violation_count')} | "
                f"{md_value(row, 'perception_risk_detection_count')} | "
                f"{md_value(row, 'slow_down_event_count')} | "
                f"{md_value(row, 'local_replan_attempt_count')} | "
                f"{md_value(row, 'successful_local_replan_count')} | "
                f"{md_value(row, 'active_route_replacement_count')} | "
                f"{md_value(row, 'final_status')} | "
                f"{md_value(row, 'notes')} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            *interpretation(rows),
            "",
            "## Selected Runs",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- `{row['marker']}` -> `{row['run_dir']}`")
    lines.extend(["", f"CSV output: `{display_path(csv_path)}`"])
    output_path.write_text("\n".join(lines) + "\n")
